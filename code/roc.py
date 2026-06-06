import torch
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import os
import time
import pandas as pd
import networkx as nx
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                             accuracy_score, balanced_accuracy_score, recall_score,
                             precision_score, roc_curve, precision_recall_curve)

from parser import parse_args
from dataset import HypergraphDataset
from model import PPI2Complex
from utils import (set_global_seed, split_graph_edges,
                   networkx_to_torch_sparse_adj, normalize_adjacency_matrix, get_pseudo_labels_and_hcn_lcn)


# ===============================================================================================
#  为了方便和完整性，我们将修改后的 LinkPredictor 直接放入 main.py 中
#  注意: 这会覆盖您从 utils.py 导入的原始 LinkPredictor
# ===============================================================================================
class LinkPredictor:
    """一个用于链接预测任务的逻辑回归分类器封装。"""

    def __init__(self, embeddings: dict, binary_operator_name: str = 'hadamard', seed: int = 42):
        """
        初始化链接预测器。
        Args:
            embeddings (dict): 节点ID到其嵌入向量的映射。
            binary_operator_name (str): 用于组合节点对嵌入的二元操作符名称。
            seed (int): 用于逻辑回归模型的随机种子。
        """
        self.embeddings = embeddings
        self.binary_operator = self._get_binary_operator(binary_operator_name)
        if not self.binary_operator:
            raise ValueError(f"错误：未知的二元操作符 '{binary_operator_name}'")
        self.model = LogisticRegression(solver='liblinear', random_state=seed, max_iter=1000)
        print(f"链接预测器已使用 '{binary_operator_name}' 操作符初始化。")

    def _get_binary_operator(self, op_name: str):
        """根据名称获取二元操作函数。"""
        if op_name == 'hadamard':
            return lambda u, v: u * v
        elif op_name == 'l1':
            return lambda u, v: np.abs(u - v)
        elif op_name == 'l2':
            return lambda u, v: (u - v) ** 2
        elif op_name == 'average':
            return lambda u, v: (u + v) / 2.0
        else:
            return None

    def _create_edge_embeddings(self, edge_list: list[tuple]) -> np.ndarray:
        """为边列表中的每条边创建嵌入。"""
        embs = []
        for u_id, v_id in edge_list:
            u_emb = self.embeddings.get(str(u_id))
            v_emb = self.embeddings.get(str(v_id))
            if u_emb is not None and v_emb is not None:
                embs.append(self.binary_operator(u_emb, v_emb))
            else:
                # 如果找不到任一节点的嵌入，则填充一个零向量
                # 假设嵌入维度可以通过第一个嵌入向量推断
                any_emb = next(iter(self.embeddings.values()), None)
                if any_emb is not None:
                    embs.append(np.zeros_like(any_emb))
                else:  # 如果没有任何嵌入
                    raise ValueError("错误：嵌入字典为空，无法推断维度。")
        return np.array(embs)

    def train(self, X_train: list[tuple], Y_train: np.ndarray) -> bool:
        """
        训练逻辑回归模型。
        Args:
            X_train (list[tuple]): 训练集的边列表。
            Y_train (np.ndarray): 对应的标签。
        Returns:
            bool: 如果训练成功则返回True，否则返回False。
        """
        if len(X_train) == 0 or len(Y_train) == 0:
            print("警告：训练数据为空，跳过训练。")
            return False
        try:
            edge_embeddings = self._create_edge_embeddings(X_train)
            if edge_embeddings.shape[0] != len(Y_train):
                print(f"警告: 嵌入数量 ({edge_embeddings.shape[0]}) 与标签数量 ({len(Y_train)}) 不匹配。")
                return False
            self.model.fit(edge_embeddings, Y_train)
            return True
        except Exception as e:
            print(f"链接预测模型训练期间发生严重错误: {e}")
            import traceback;
            traceback.print_exc()
            return False

    def evaluate(self, X_test: list[tuple], Y_test: np.ndarray, set_name: str):
        """
        在测试集上评估模型，并返回详细的性能指标和曲线数据。
        """
        edge_embeddings = self._create_edge_embeddings(X_test)
        Y_pred_probs = self.model.predict_proba(edge_embeddings)[:, 1]
        Y_pred_binary = self.model.predict(edge_embeddings)

        # 计算性能指标
        scores = {
            'auc_roc': roc_auc_score(Y_test, Y_pred_probs),
            'auc_pr': average_precision_score(Y_test, Y_pred_probs),
            'f1': f1_score(Y_test, Y_pred_binary),
            'accuracy': accuracy_score(Y_test, Y_pred_binary),
            'balanced_accuracy': balanced_accuracy_score(Y_test, Y_pred_binary),
            'recall': recall_score(Y_test, Y_pred_binary),
            'precision': precision_score(Y_test, Y_pred_binary, zero_division=0)
        }
        print(f"在 {set_name} 上的性能:")
        for metric, value in scores.items():
            print(f"  - {metric.upper()}: {value:.4f}")

        # 计算ROC和PR曲线数据
        fpr, tpr, _ = roc_curve(Y_test, Y_pred_probs)
        precision, recall, _ = precision_recall_curve(Y_test, Y_pred_probs)

        # 将曲线数据打包成DataFrame
        roc_data = pd.DataFrame({'x_coordinate': fpr, 'y_coordinate': tpr})
        pr_data = pd.DataFrame({'x_coordinate': recall, 'y_coordinate': precision})

        return scores, roc_data, pr_data


def main(args):
    start_run_time = time.time()
    set_global_seed(args.seed)
    device = torch.device(args.device)
    print(f"使用设备: {device}")

    # 定义结果输出目录 (相对于 code/ 目录)
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'result')
    os.makedirs(results_dir, exist_ok=True)
    print(f"结果将保存到: {results_dir}")

    print("\n===== 步骤 1: 加载和预处理数据集 =====")
    try:
        dataset_obj = HypergraphDataset(args)
        adj_orig_torch_sparse = networkx_to_torch_sparse_adj(dataset_obj.graph_original, dataset_obj.node_to_idx,
                                                             device)
        adj_orig_norm_for_encoder = normalize_adjacency_matrix(adj_orig_torch_sparse, add_self_loops=True)

        adj_orig_dense_target_for_recon = nx.to_numpy_array(
            dataset_obj.graph_original,
            nodelist=dataset_obj.node_list,
            dtype=np.float32
        )
        adj_orig_dense_target_for_recon = torch.from_numpy(adj_orig_dense_target_for_recon).to(device)

        adj_rc_norm_for_encoder = None
        if args.use_ricci_augmentation and dataset_obj.graph_augmented_rc:
            adj_rc_torch_sparse = networkx_to_torch_sparse_adj(dataset_obj.graph_augmented_rc, dataset_obj.node_to_idx,
                                                               device)
            adj_rc_norm_for_encoder = normalize_adjacency_matrix(adj_rc_torch_sparse, add_self_loops=True)

        H_clique_torch = dataset_obj.H_clique.to(device) if dataset_obj.H_clique is not None else None
        node_features_esm = dataset_obj.node_features_esm.to(device).float()
    except Exception as e:
        print(f"数据集初始化/图转换严重错误: {e}");
        import traceback;
        traceback.print_exc();
        return
    print(f"最终节点特征维度: {dataset_obj.feature_dim}")

    print("\n===== 步骤 2: 初始化模型和优化器 =====")
    model = PPI2Complex(args, initial_feature_dim=dataset_obj.feature_dim).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    print("\n===== 步骤 3: 开始模型训练 =====")
    if args.lambda_hscl > 0 and args.scl_update_pseudo_labels_every > 0:
        print(f"SCL: Epoch 1前，尝试初始化伪标签...")
        with torch.no_grad():
            temp_embeds_dict = model.encode(node_features_esm, adj_orig_norm_for_encoder, adj_rc_norm_for_encoder,
                                            H_clique_torch)
            initial_embeds_for_scl = model.get_final_embeddings(temp_embeds_dict)
            if initial_embeds_for_scl is None and node_features_esm.shape[0] > 0:
                initial_embeds_for_scl = model.initial_projection(node_features_esm)
            if initial_embeds_for_scl is not None and initial_embeds_for_scl.numel() > 0:
                num_clu = max(2, int(dataset_obj.num_nodes * args.scl_cluster_k_ratio))
                if dataset_obj.num_nodes < num_clu: num_clu = max(1,
                                                                  dataset_obj.num_nodes // 2) if dataset_obj.num_nodes >= 2 else 1
                if num_clu == 1 and dataset_obj.num_nodes > 1: num_clu = 2
                if dataset_obj.num_nodes >= 1 and num_clu > 0:
                    model.scl_pseudo_labels, model.scl_hcn_indices, model.scl_lcn_indices, model.scl_centroids = \
                        get_pseudo_labels_and_hcn_lcn(initial_embeds_for_scl.detach(), num_clu, args.scl_hcn_ratio,
                                                      args.device)
                    if model.scl_pseudo_labels is not None: print(f"SCL初始伪标签已生成。")
                else:
                    print("SCL: 节点数或簇数不足，无法初始化伪标签。")
            else:
                print("SCL: 用于初始化伪标签的嵌入无效，跳过。")

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        current_embeddings_dict = model.encode(node_features_esm, adj_orig_norm=adj_orig_norm_for_encoder,
                                               adj_rc_norm=adj_rc_norm_for_encoder, H_clique=H_clique_torch)

        if args.lambda_hscl > 0 and args.scl_update_pseudo_labels_every > 0 and \
                (epoch % args.scl_update_pseudo_labels_every == 0) and epoch < args.epochs and epoch > 1:
            print(f"Epoch {epoch}: 准备更新SCL伪标签...")
            with torch.no_grad():
                Z_hyper_curr = current_embeddings_dict.get('hyper_clique')
                Z_orig_curr = current_embeddings_dict.get('orig')
                Z_rc_curr = current_embeddings_dict.get('rc_aug')

                if Z_orig_curr is not None and Z_hyper_curr is not None:
                    Zs_scl_curr = 0.5 * (Z_orig_curr + Z_rc_curr) if Z_rc_curr is not None else Z_orig_curr
                    Zc_scl_curr = (1 - args.fusion_gamma) * Zs_scl_curr + args.fusion_gamma * Z_hyper_curr

                    num_clu = max(2, int(dataset_obj.num_nodes * args.scl_cluster_k_ratio))
                    if dataset_obj.num_nodes < num_clu: num_clu = max(1,
                                                                      dataset_obj.num_nodes // 2) if dataset_obj.num_nodes >= 2 else 1
                    if num_clu == 1 and dataset_obj.num_nodes > 1: num_clu = 2
                    if dataset_obj.num_nodes >= 1 and num_clu > 0:
                        model.scl_pseudo_labels, model.scl_hcn_indices, model.scl_lcn_indices, model.scl_centroids = \
                            get_pseudo_labels_and_hcn_lcn(Zc_scl_curr.detach(), num_clu, args.scl_hcn_ratio,
                                                          args.device)
                        if model.scl_pseudo_labels is not None: print(f"SCL伪标签已在Epoch {epoch} 更新。")
                    else:
                        print("SCL:节点数或簇数不足，跳过伪标签更新。")
                else:
                    print("警告：SCL伪标签更新跳过，当前基础嵌入不完整。")

        total_loss, loss_breakdown = model.compute_all_losses(current_embeddings_dict, node_features_esm,
                                                              adj_orig_dense_target=adj_orig_dense_target_for_recon,
                                                              H_clique=H_clique_torch,
                                                              adj_orig_norm_for_encoder=adj_orig_norm_for_encoder,
                                                              current_epoch=epoch)
        if torch.isnan(total_loss) or torch.isinf(total_loss) or (
                abs(total_loss.item()) < 1e-9 and epoch > 1 and total_loss.requires_grad):
            print(f"警告: Epoch {epoch} 损失无效或为零 ({total_loss.item()}).")
            if torch.isnan(total_loss) or torch.isinf(total_loss): print("NaN/Inf损失，训练终止。"); break
        else:
            if total_loss.requires_grad:
                total_loss.backward()
                if args.gradient_clip_value is not None and args.gradient_clip_value > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.gradient_clip_value)
                optimizer.step()
            elif abs(total_loss.item()) < 1e-9 and not total_loss.requires_grad:
                print(f"Epoch {epoch} 总损失为0且不要求梯度，跳过优化步骤。")

        if epoch % args.eval_every_epochs == 0 or epoch == args.epochs:
            loss_val = total_loss.item() if isinstance(total_loss, torch.Tensor) and total_loss.requires_grad else (
                total_loss if isinstance(total_loss, float) else 0.0)
            loss_str = f"总损失: {loss_val:.4f}"
            for name, val_item in loss_breakdown.items():
                val = val_item
                loss_str += f", {name}: {val:.4f}"
            print(f"Epoch [{epoch}/{args.epochs}], {loss_str}")
    print("模型训练完成。")

    print("\n===== 步骤 4: 生成最终嵌入并评估 =====")
    model.eval()
    with torch.no_grad():
        final_embed_dict_tensor = model.encode(node_features_esm, adj_orig_norm_for_encoder, adj_rc_norm_for_encoder,
                                               H_clique_torch)
        final_embed_tensor = model.get_final_embeddings(final_embed_dict_tensor)
        if final_embed_tensor is None or final_embed_tensor.numel() == 0: raise RuntimeError("未能生成有效最终嵌入。")
        final_embed_np = final_embed_tensor.cpu().numpy()
    final_embed_map = {node_id: final_embed_np[idx] for node_id, idx in dataset_obj.node_to_idx.items()}
    if final_embed_map:
        print(f"已生成最终嵌入，形状: ({len(final_embed_map)}, {final_embed_np.shape[1]})")
        if dataset_obj.graph_original.number_of_edges() >= 10:
            print("\n--- 开始链接预测评估 ---")

            # 设定进行5次试验
            num_trials = 5
            all_trial_results = []

            for i in range(num_trials):
                trial_seed = args.seed + i
                print(f"\n链接预测试验 {i + 1}/{num_trials} (种子: {trial_seed})...")
                g_train, (X_train, Y_train), (X_val, Y_val), (X_test, Y_test) = split_graph_edges(
                    dataset_obj.graph_original, test_ratio=args.link_pred_test_ratio,
                    val_ratio=args.link_pred_val_ratio, seed=trial_seed)
                if len(X_train) < 2 or len(X_test) < 2 or (len(Y_train) > 0 and len(np.unique(Y_train)) < 2):
                    print(f"警告:试验 {i + 1}样本不足或类别单一，跳过。")
                    continue

                # 使用我们修改后的 LinkPredictor
                link_predictor = LinkPredictor(embeddings=final_embed_map, binary_operator_name='hadamard',
                                               seed=trial_seed)
                if not link_predictor.train(X_train, Y_train):
                    print(f"警告:试验 {i + 1}链接预测模型训练失败。")
                    continue

                # 评估并获取曲线数据
                test_scores, roc_data, pr_data = link_predictor.evaluate(X_test, Y_test, "测试集")

                # --- 新增代码：构建并保存曲线数据文件 ---
                roc_data['model_name'] = 'PPI2Complex'
                roc_data['trial'] = i + 1
                roc_data['curve_type'] = 'ROC'
                roc_data['auc_metric_name'] = 'ROC-AUC'
                roc_data['auc_value'] = test_scores['auc_roc']

                pr_data['model_name'] = 'PPI2Complex'
                pr_data['trial'] = i + 1
                pr_data['curve_type'] = 'PR'
                pr_data['auc_metric_name'] = 'PR-AUC'
                pr_data['auc_value'] = test_scores['auc_pr']

                # 合并ROC和PR数据
                curves_df = pd.concat([roc_data, pr_data], ignore_index=True)

                # 重新排列列以匹配示例文件
                curves_df = curves_df[['model_name', 'trial', 'curve_type', 'x_coordinate', 'y_coordinate',
                                       'auc_metric_name', 'auc_value']]

                # 保存文件
                output_filename = os.path.join(results_dir, f'trial_{i + 1}_roc_pr_curves.csv')
                try:
                    curves_df.to_csv(output_filename, index=False)
                    print(f"试验 {i + 1} 的ROC/PR曲线数据已成功保存到: {output_filename}")
                except Exception as e:
                    print(f"错误：无法保存文件 {output_filename}。错误信息: {e}")

                # --- 代码修改结束 ---

                test_scores['trial'] = i + 1
                all_trial_results.append(test_scores)

            if all_trial_results:
                results_df = pd.DataFrame(all_trial_results)
                print("\n--- 链接预测平均性能 (测试集) ---")
                metrics_to_print = ["auc_roc", "auc_pr", "f1", "accuracy", "balanced_accuracy", "recall", "precision"]
                for metric in metrics_to_print:
                    if metric in results_df:
                        mean_val = results_df[metric].mean()
                        std_val = results_df[metric].std()
                        print(f" 平均 {metric.upper()}: {mean_val:.4f} ± {std_val:.4f}")
        else:
            print("警告：图中边数过少，跳过链接预测。")
    else:
        raise RuntimeError("错误：未能生成最终嵌入。")

    end_run_time = time.time()
    print(f"\n总运行时间: {end_run_time - start_run_time:.2f} 秒")
    print("===== PPI2Complex 运行结束 =====")


if __name__ == "__main__":
    args = parse_args()
    import json

    print("===== PPI2Complex 配置参数 =====")
    print(json.dumps(vars(args), indent=2, ensure_ascii=False))
    print("================================")

    # 手动设置试验次数为5，以确保生成5个文件
    args.link_pred_n_trials = 5

    main(args)