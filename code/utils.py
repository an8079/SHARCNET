# utils.py
import torch
import torch.nn.functional as F
import numpy as np
import networkx as nx
import random
from collections import defaultdict
import math

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (  # 新增 recall_score 和 precision_score
    roc_auc_score, average_precision_score,
    f1_score, accuracy_score, balanced_accuracy_score,
    recall_score, precision_score
)
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

try:
    from GraphRicciCurvature.OllivierRicci import OllivierRicci

    RICCI_CURVATURE_LIB_AVAILABLE = True
except ImportError:
    RICCI_CURVATURE_LIB_AVAILABLE = False
    print("警告: 'GraphRicciCurvature' 库未安装。如果使用Ricci曲率增强，程序将无法运行。")
    print("请尝试安装: pip install GraphRicciCurvature")


# --- set_global_seed, normalize_adjacency_matrix, networkx_to_torch_sparse_adj ---
# --- ricci_curvature_graph_augmentation, compute_ricci_curvature_edges ---
# --- mask_features ---
# --- get_pseudo_labels_and_hcn_lcn ---
# --- InfoNCELoss, TrueTSCGCWeightedTCL ---
# --- split_graph_edges ---
# (这些函数与上一轮答复中的 utils.py 基本一致，为简洁省略，请确保它们存在)
# (请将上一轮答复中的 utils.py 对应函数粘贴到此处)
def set_global_seed(seed: int):
    random.seed(seed);
    np.random.seed(seed);
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(
        seed); torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    print(f"全局随机种子已设置为: {seed}")


def normalize_adjacency_matrix(adj: torch.sparse.FloatTensor, add_self_loops: bool = True) -> torch.sparse.FloatTensor:
    num_nodes = adj.size(0)
    if num_nodes == 0: return adj.clone()
    if add_self_loops:
        eye_indices = torch.arange(num_nodes, device=adj.device).unsqueeze(0).repeat(2, 1)
        eye_values = torch.ones(num_nodes, device=adj.device)
        eye = torch.sparse_coo_tensor(eye_indices, eye_values, adj.size(), device=adj.device)
        adj_processed = (adj + eye).coalesce()
    else:
        adj_processed = adj.coalesce()
    row_sum = torch.sparse.sum(adj_processed, dim=1).to_dense()
    d_inv_sqrt = torch.pow(row_sum, -0.5);
    d_inv_sqrt[torch.isinf(d_inv_sqrt) | torch.isnan(d_inv_sqrt)] = 0.
    indices = adj_processed.indices();
    values = adj_processed.values()
    if indices.numel() == 0: return adj_processed
    src_norm = d_inv_sqrt[indices[0]];
    dst_norm = d_inv_sqrt[indices[1]]
    normalized_values = values * src_norm * dst_norm
    return torch.sparse_coo_tensor(indices, normalized_values, adj_processed.size()).coalesce()


def networkx_to_torch_sparse_adj(graph: nx.Graph, node_map: dict = None,
                                 device: str = 'cpu') -> torch.sparse.FloatTensor:
    if node_map is None: node_list = sorted(list(graph.nodes())); node_map = {node: i for i, node in
                                                                              enumerate(node_list)}
    num_nodes = len(node_map)
    if num_nodes == 0: return torch.sparse_coo_tensor(torch.empty((2, 0), dtype=torch.long), [], (0, 0), device=device)
    try:
        adj_scipy = nx.to_scipy_sparse_array(graph, nodelist=list(node_map.keys()), format='coo', dtype=np.float32)
    except AttributeError:
        adj_scipy = nx.to_scipy_sparse_matrix(graph, nodelist=list(node_map.keys()), format='coo',
                                              dtype=np.float32)  # Older NetworkX
    row = torch.from_numpy(adj_scipy.row.astype(np.int64)).to(torch.long);
    col = torch.from_numpy(adj_scipy.col.astype(np.int64)).to(torch.long)
    edge_index = torch.stack([row, col], dim=0)
    values = torch.from_numpy(adj_scipy.data.astype(np.float32))
    return torch.sparse_coo_tensor(edge_index, values, (num_nodes, num_nodes), device=device).coalesce()


def compute_ricci_curvature_edges(graph: nx.Graph, alpha: float = 0.5) -> dict:
    if not RICCI_CURVATURE_LIB_AVAILABLE: print("错误: GraphRicciCurvature库不可用..."); return {}
    if graph.number_of_edges() == 0: return {}
    print(f"计算Ricci曲率 (alpha={alpha})...");
    orc = OllivierRicci(graph, alpha=alpha, verbose="ERROR")
    try:
        orc.compute_ricci_curvature()
    except Exception as e:
        print(f"Ricci曲率计算时发生错误: {e}"); return {}
    curvature_dict = {tuple(sorted((u, v))): data['ricciCurvature'] for u, v, data in orc.G.edges(data=True) if
                      'ricciCurvature' in data}
    print(f"Ricci曲率计算完成。处理了 {len(curvature_dict)} 条边。");
    return curvature_dict


def ricci_curvature_graph_augmentation(graph: nx.Graph, alpha: float = 0.5, cutoff_min: float = -0.3,
                                       method: str = 'remove') -> nx.Graph:
    aug_graph = graph.copy();
    edge_curvatures = compute_ricci_curvature_edges(aug_graph, alpha)
    if not edge_curvatures: print("警告:未能计算Ricci曲率..."); return aug_graph
    if method == 'remove':
        removed_edges = [edge for edge, curvature in edge_curvatures.items() if curvature < cutoff_min]
        aug_graph.remove_edges_from(removed_edges);
        print(f"RCGA ('remove'): 移除了 {len(removed_edges)} 条边 (曲率 < {cutoff_min})")
    elif method == 'weight':
        nx.set_edge_attributes(aug_graph, values=0, name='ricci_weight')
        for edge, curvature in edge_curvatures.items():
            if aug_graph.has_edge(*edge): aug_graph.edges[edge]['ricci_weight'] = curvature
        print(f"RCGA ('weight'): 为边添加了 'ricci_weight' 属性。")
    else:
        raise ValueError(f"未知的Ricci增强方法: {method}")
    return aug_graph


def mask_features(features: torch.Tensor, mask_rate: float) -> torch.Tensor:
    if mask_rate == 0.0 or mask_rate is None: return features.clone()
    masked_features = features.clone();
    num_nodes, feat_dim = features.shape
    if feat_dim == 0: return masked_features
    for i in range(num_nodes):
        mask = torch.rand(feat_dim, device=features.device) < mask_rate
        masked_features[i, mask] = 0.0
    return masked_features


@torch.no_grad()
def get_pseudo_labels_and_hcn_lcn(embeddings: torch.Tensor, num_clusters: int, hcn_ratio_per_cluster: float,
                                  device: str = 'cpu'):
    if embeddings.shape[0] < 2 or num_clusters <= 0:
        print(f"警告 (SCL): 节点数 ({embeddings.shape[0]}) 或簇数 ({num_clusters}) 不足。跳过伪标签。")
        return None, None, None, None
    actual_num_clusters = min(num_clusters, embeddings.shape[0])
    if actual_num_clusters < 2 and embeddings.shape[0] >= 2: actual_num_clusters = 2
    if embeddings.shape[0] < actual_num_clusters:
        print(f"警告 (SCL): 调整后簇数({actual_num_clusters})仍大于等于样本数({embeddings.shape[0]})。跳过。")
        return None, None, None, None
    embeddings_cpu = embeddings.cpu().numpy()
    kmeans = KMeans(n_clusters=actual_num_clusters, random_state=0, n_init='auto', verbose=0).fit(embeddings_cpu)
    initial_pseudo_labels = torch.from_numpy(kmeans.labels_).to(device)
    initial_centroids = torch.from_numpy(kmeans.cluster_centers_).to(device).float()
    dist_to_centroids_sq = torch.cdist(embeddings, initial_centroids, p=2).pow(2)
    q_membership = (1.0 + dist_to_centroids_sq).pow(-1.0)
    sum_q_for_norm = torch.sum(q_membership, dim=1, keepdim=True).clamp(min=1e-9)  # 修正：clamp放这里
    q_membership = q_membership / sum_q_for_norm
    modified_centroids = torch.zeros_like(initial_centroids)
    for k_idx in range(actual_num_clusters):
        weights_k_all_nodes = q_membership[:, k_idx]
        weighted_sum_embeds_all = torch.sum(embeddings * weights_k_all_nodes.unsqueeze(1), dim=0)
        sum_weights_all = torch.sum(weights_k_all_nodes)
        if sum_weights_all > 1e-6:
            modified_centroids[k_idx] = weighted_sum_embeds_all / sum_weights_all
        else:
            modified_centroids[k_idx] = initial_centroids[k_idx]
    dist_to_modified_centroids_sq = torch.cdist(embeddings, modified_centroids, p=2).pow(2)
    final_q_membership = (1.0 + dist_to_modified_centroids_sq).pow(-1.0)
    sum_final_q_for_norm = torch.sum(final_q_membership, dim=1, keepdim=True).clamp(min=1e-9)  # 修正：clamp放这里
    final_q_membership = final_q_membership / sum_final_q_for_norm
    final_pseudo_labels = torch.argmax(final_q_membership, dim=1)
    hcn_node_indices_list, lcn_node_indices_list = [], []
    for k_idx in range(actual_num_clusters):
        nodes_in_k_mask = (final_pseudo_labels == k_idx)
        if not torch.any(nodes_in_k_mask): continue
        membership_to_k_for_cluster_nodes = final_q_membership[nodes_in_k_mask, k_idx]
        num_nodes_in_k = membership_to_k_for_cluster_nodes.shape[0]
        num_hcn_in_k = max(1, int(num_nodes_in_k * hcn_ratio_per_cluster))
        _, sorted_indices_in_k_local = torch.sort(membership_to_k_for_cluster_nodes, descending=True)
        global_indices_in_cluster_k = torch.where(nodes_in_k_mask)[0]
        hcn_local_idx_for_k = sorted_indices_in_k_local[:num_hcn_in_k]
        lcn_local_idx_for_k = sorted_indices_in_k_local[num_hcn_in_k:]
        if hcn_local_idx_for_k.numel() > 0: hcn_node_indices_list.extend(
            global_indices_in_cluster_k[hcn_local_idx_for_k].tolist())
        if lcn_local_idx_for_k.numel() > 0: lcn_node_indices_list.extend(
            global_indices_in_cluster_k[lcn_local_idx_for_k].tolist())
    return (final_pseudo_labels, torch.tensor(list(set(hcn_node_indices_list)), dtype=torch.long, device=device),
            torch.tensor(list(set(lcn_node_indices_list)), dtype=torch.long, device=device), modified_centroids)


class InfoNCELoss(torch.nn.Module):
    def __init__(self, temperature: float = 0.1):
        super(InfoNCELoss, self).__init__(); self.temperature = temperature

    def forward(self, anchor_embeds: torch.Tensor, positive_embeds: torch.Tensor, negative_embeds: torch.Tensor = None,
                all_candidates_for_neg: torch.Tensor = None):
        anchor_embeds_norm = F.normalize(anchor_embeds, p=2, dim=1);
        positive_embeds_norm = F.normalize(positive_embeds, p=2, dim=1)
        pos_sim = torch.sum(anchor_embeds_norm * positive_embeds_norm, dim=1) / self.temperature
        if anchor_embeds.shape[0] == 0: return torch.tensor(0.0, device=anchor_embeds.device)
        if negative_embeds is not None and negative_embeds.numel() > 0:
            neg_sim = torch.matmul(anchor_embeds_norm, F.normalize(negative_embeds, p=2, dim=1).t()) / self.temperature
            logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
        elif all_candidates_for_neg is not None and all_candidates_for_neg.numel() > 0:
            neg_sim = torch.matmul(anchor_embeds_norm,
                                   F.normalize(all_candidates_for_neg, p=2, dim=1).t()) / self.temperature
            logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
        else:
            return -pos_sim.mean()
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=anchor_embeds.device)
        return F.cross_entropy(logits, labels)


class TrueTSCGCWeightedTCL(torch.nn.Module):
    def __init__(self, temperature: float = 0.1):
        super(TrueTSCGCWeightedTCL, self).__init__()
        self.temperature = temperature

    def forward(self, anchor_embeds_view1: torch.Tensor,
                list_of_pos_embeds_view2_for_each_anchor: list[torch.Tensor],
                list_of_pos_weights_for_each_anchor: list[torch.Tensor],
                negative_pool_view2: torch.Tensor):
        total_loss = torch.tensor(0.0, device=anchor_embeds_view1.device)
        num_valid_anchors_processed = 0
        anchor_embeds_v1_norm = F.normalize(anchor_embeds_view1, p=2, dim=1)
        negative_pool_v2_norm = F.normalize(negative_pool_view2, p=2, dim=1)
        for i in range(anchor_embeds_view1.shape[0]):
            anchor_i_v1 = anchor_embeds_v1_norm[i].unsqueeze(0)
            if i >= len(list_of_pos_embeds_view2_for_each_anchor) or list_of_pos_embeds_view2_for_each_anchor[i].shape[
                0] == 0: continue
            pos_embeds_v2_for_anchor_i = F.normalize(list_of_pos_embeds_view2_for_each_anchor[i], p=2, dim=1)
            pos_weights_for_anchor_i = list_of_pos_weights_for_each_anchor[i]
            sim_anchor_v1_to_pos_v2 = torch.matmul(anchor_i_v1, pos_embeds_v2_for_anchor_i.t()).squeeze(0)
            numerator_term = torch.sum(pos_weights_for_anchor_i * torch.exp(sim_anchor_v1_to_pos_v2 / self.temperature))
            sim_anchor_v1_to_neg_pool_v2 = torch.matmul(anchor_i_v1, negative_pool_v2_norm.t()).squeeze(0)
            denominator_neg_term = torch.sum(torch.exp(sim_anchor_v1_to_neg_pool_v2 / self.temperature))
            current_loss_for_anchor_i = -torch.log(
                numerator_term / (numerator_term + denominator_neg_term).clamp(min=1e-9))
            if not (torch.isnan(current_loss_for_anchor_i) or torch.isinf(current_loss_for_anchor_i)):
                total_loss += current_loss_for_anchor_i;
                num_valid_anchors_processed += 1
        if num_valid_anchors_processed == 0: return torch.tensor(0.0, device=anchor_embeds_view1.device)
        return total_loss / num_valid_anchors_processed


BINARY_OPERATORS = {
    'hadamard': lambda u, v: u * v, 'l1': lambda u, v: np.abs(u - v),
    'l2': lambda u, v: (u - v) ** 2, 'avg': lambda u, v: (u + v) / 2.0
}


def split_graph_edges(graph: nx.Graph, test_ratio: float = 0.2, val_ratio: float = 0.1, seed: int = None):
    if seed is not None: np.random.seed(seed); random.seed(seed)
    edges = list(graph.edges());
    nodes = list(graph.nodes())
    num_edges = len(edges);
    num_nodes = len(nodes)
    if num_edges == 0: empty_arr = np.array([]).reshape(0, 2); empty_lbl = np.array([],
                                                                                    dtype=np.int64); return graph.copy(), (
        empty_arr, empty_lbl), (empty_arr, empty_lbl), (empty_arr, empty_lbl)
    non_edges = [];
    adj = set(graph.edges()) | set((v, u) for u, v in graph.edges());
    num_total_negative_samples = num_edges
    node_array = np.array(nodes, dtype=object)
    if num_nodes < 2:
        X_neg = np.array([]).reshape(0, 2)
    else:
        neg_attempts = 0;
        max_neg_attempts = num_total_negative_samples * 20;
        possible_neg_set = set()
        while len(possible_neg_set) < num_total_negative_samples and neg_attempts < max_neg_attempts:
            neg_attempts += 1;
            if len(node_array) < 2: break
            u, v = np.random.choice(node_array, 2, replace=False);
            edge_tuple = tuple(sorted((u, v)))
            if u != v and edge_tuple not in adj: possible_neg_set.add(edge_tuple)
        non_edges = list(possible_neg_set)
    X_pos = np.array(edges, dtype=object);
    Y_pos = np.ones(len(X_pos), dtype=np.int64)
    X_neg = np.array(non_edges, dtype=object);
    Y_neg = np.zeros(len(X_neg), dtype=np.int64)
    X_all = np.concatenate((X_pos, X_neg)) if len(X_neg) > 0 else X_pos
    Y_all = np.concatenate((Y_pos, Y_neg)) if len(Y_neg) > 0 else Y_pos
    if len(X_all) < 2 or (len(Y_all) > 0 and len(np.unique(Y_all)) < 2): print(
        "警告：总样本数过少或只有单一类别..."); g_train = graph.copy(); g_train.remove_edges_from(
        list(g_train.edges())); g_train.add_edges_from(X_pos if len(X_pos) > 0 else []); return g_train, (X_all,
                                                                                                          Y_all), (
        np.array([]).reshape(0, 2), np.array([])), (np.array([]).reshape(0, 2), np.array([]))

    stratify_Y_all = Y_all if len(np.unique(Y_all)) > 1 else None
    X_train_val, X_test, Y_train_val, Y_test = train_test_split(X_all, Y_all, test_size=test_ratio, random_state=seed,
                                                                stratify=stratify_Y_all)

    relative_val_size = 0.0
    if (1.0 - test_ratio) > 1e-6 and len(Y_train_val) >= 2 and (
            len(Y_train_val) == 0 or len(np.unique(Y_train_val)) > 1):  # Check Y_train_val as well
        relative_val_size = val_ratio / (1.0 - test_ratio);
        if relative_val_size >= 1.0 or relative_val_size <= 0.0: relative_val_size = 0.5 if (
                                                                                                        1.0 - test_ratio) > val_ratio and len(
            Y_train_val) >= 4 else 0.0  # ensure valid split, and enough samples for stratification

        if relative_val_size > 0 and len(Y_train_val) >= 2 and len(
                np.unique(Y_train_val)) > 1:  # Check again before split
            X_train, X_val, Y_train, Y_val = train_test_split(X_train_val, Y_train_val, test_size=relative_val_size,
                                                              random_state=seed, stratify=Y_train_val)
        else:
            X_train, Y_train = X_train_val, Y_train_val; X_val, Y_val = np.array([]).reshape(0,
                                                                                             X_all.shape[1]), np.array(
                [])
    else:
        X_train, Y_train = X_train_val, Y_train_val; X_val, Y_val = np.array([]).reshape(0, X_all.shape[
            1] if X_all.ndim > 1 and X_all.shape[0] > 0 else 2), np.array([])

    g_train = nx.Graph();
    g_train.add_nodes_from(nodes);
    train_pos_edges = [tuple(edge) for i, edge in enumerate(X_train) if Y_train[i] == 1];
    g_train.add_edges_from(train_pos_edges)
    return g_train, (X_train, Y_train), (X_val, Y_val), (X_test, Y_test)


class LinkPredictor:
    def __init__(self, embeddings: dict, binary_operator_name: str = 'hadamard', seed: int = 42):
        if not isinstance(embeddings, dict): raise ValueError("Embeddings must be a dict.")
        self.embeddings = embeddings;
        self.operator = BINARY_OPERATORS.get(binary_operator_name)
        if self.operator is None: raise ValueError(f"Unknown operator: {binary_operator_name}")
        # Ensure cv is at most number of samples in smallest class, and at least 2 if possible
        # This logic is complex to put directly in init, better handled in train or by user
        self.cv_folds = 5
        lr_clf = LogisticRegressionCV(Cs=10, cv=self.cv_folds, scoring="roc_auc", max_iter=10000, random_state=seed,
                                      solver='liblinear', class_weight='balanced', n_jobs=-1)
        self.model = Pipeline(steps=[("scaler", StandardScaler()), ("classifier", lr_clf)])

    def _create_edge_features(self, edges: np.ndarray) -> tuple[np.ndarray | None, np.ndarray]:
        edge_features, valid_indices = [], []
        if edges.shape[0] == 0: return None, np.array([], dtype=int)
        for i, edge_pair in enumerate(edges):
            u, v = str(edge_pair[0]), str(edge_pair[1])
            if u in self.embeddings and v in self.embeddings:
                emb_u, emb_v = self.embeddings[u], self.embeddings[v]
                edge_features.append(self.operator(emb_u, emb_v));
                valid_indices.append(i)
        if not edge_features: return None, np.array([], dtype=int)
        return np.array(edge_features, dtype=np.float32), np.array(valid_indices, dtype=int)

    def train(self, X_train: np.ndarray, Y_train: np.ndarray) -> bool:
        train_edge_features, valid_train_indices = self._create_edge_features(X_train)
        if train_edge_features is None or train_edge_features.shape[0] == 0: print(
            "错误：无法创建训练边特征..."); return False
        Y_train_filtered = Y_train[valid_train_indices]

        unique_classes_train, counts_train = np.unique(Y_train_filtered, return_counts=True)
        if len(unique_classes_train) < 2: print(
            f"错误：训练数据中有效样本仅包含单一类别 ({unique_classes_train})。"); return False

        # Adjust CV folds for LogisticRegressionCV if necessary
        min_class_count_train = counts_train.min()
        actual_cv_folds = min(self.cv_folds, min_class_count_train)
        if actual_cv_folds < 2:
            print(f"警告: 训练集中最小类别样本数 ({min_class_count_train}) 过少，无法进行交叉验证。使用简单逻辑回归。")
            lr_simple = LogisticRegression(solver='liblinear', class_weight='balanced',
                                           random_state=self.model.named_steps['classifier'].random_state,
                                           max_iter=10000)
            self.model.set_params(classifier=lr_simple)  # Replace classifier
        else:
            # Ensure the original classifier is LogisticRegressionCV and update its cv param
            if isinstance(self.model.named_steps['classifier'], LogisticRegressionCV):
                self.model.named_steps['classifier'].cv = actual_cv_folds
            else:  # If it was replaced by LogisticRegression, re-init with CV
                lr_cv = LogisticRegressionCV(Cs=10, cv=actual_cv_folds, scoring="roc_auc", max_iter=10000,
                                             random_state=self.model.named_steps['classifier'].random_state,
                                             solver='liblinear', class_weight='balanced', n_jobs=-1)
                self.model.set_params(classifier=lr_cv)

        self.model.fit(train_edge_features, Y_train_filtered);
        return True

    def evaluate(self, X_eval: np.ndarray, Y_eval: np.ndarray, set_name: str = "评估集") -> dict:
        # 更新默认指标列表以包含 recall 和 precision
        metric_keys = ["auc_roc", "auc_pr", "f1", "accuracy", "balanced_accuracy", "recall", "precision"]
        default_scores = {m: 0.0 for m in metric_keys}

        eval_edge_features, valid_eval_indices = self._create_edge_features(X_eval)
        if eval_edge_features is None or eval_edge_features.shape[0] == 0: print(
            f"警告：无法为 {set_name} 创建边特征..."); return default_scores

        Y_eval_filtered = Y_eval[valid_eval_indices]
        if len(Y_eval_filtered) == 0: print(f"警告：{set_name} 中沒有有效标签。"); return default_scores

        try:
            getattr(self.model, "classes_")  # Check if model is fitted
        except AttributeError:
            print(f"错误: 模型未在 {set_name} 评估前训练。"); return default_scores

        Y_pred = self.model.predict(eval_edge_features)

        unique_classes_eval = np.unique(Y_eval_filtered)
        if len(unique_classes_eval) < 2:
            print(
                f"警告：{set_name} 中有效样本仅包含单一类别 ({unique_classes_eval})。AUC/AUPR/Recall/Precision 可能不准确或意义有限。")
            acc = accuracy_score(Y_eval_filtered, Y_pred)
            # adjusted=False for single class if sklearn version requires
            bal_acc = balanced_accuracy_score(Y_eval_filtered, Y_pred, adjusted=(len(unique_classes_eval) > 1))
            f1_val = f1_score(Y_eval_filtered, Y_pred, zero_division=0)
            recall_val = recall_score(Y_eval_filtered, Y_pred, zero_division=0)
            precision_val = precision_score(Y_eval_filtered, Y_pred, zero_division=0)
            return {"auc_roc": 0.5, "auc_pr": np.mean(Y_eval_filtered) if len(Y_eval_filtered) > 0 else 0.0,
                    "f1": f1_val, "accuracy": acc, "balanced_accuracy": bal_acc,
                    "recall": recall_val, "precision": precision_val}

        Y_prob = self.model.predict_proba(eval_edge_features)[:, 1]
        try:
            return {
                "auc_roc": roc_auc_score(Y_eval_filtered, Y_prob),
                "auc_pr": average_precision_score(Y_eval_filtered, Y_prob),
                "f1": f1_score(Y_eval_filtered, Y_pred, zero_division=0),
                "accuracy": accuracy_score(Y_eval_filtered, Y_pred),
                "balanced_accuracy": balanced_accuracy_score(Y_eval_filtered, Y_pred, adjusted=True),
                "recall": recall_score(Y_eval_filtered, Y_pred, zero_division=0),  # 新增
                "precision": precision_score(Y_eval_filtered, Y_pred, zero_division=0)  # 新增
            }
        except ValueError as e:
            print(f"计算指标时出错 for {set_name}: {e}."); return default_scores