# dataset_dpfunc.py
"""DPFunc 蛋白质功能预测数据集加载器。

使用 InterPro 域特征（而非 ESM 序列特征）构建蛋白质相似图，
复用 PPI2Complex 模型进行嵌入学习。

数据来源：data_dpfunc/ 目录，包含：
  - id_map.pkl, all_protein_interpros.pkl, inter_idx.pkl
  - train/valid/test_id_map.pkl
  - {bp,cc,mf}_{train,valid,test}_go.txt + _pid_list.txt
"""

import os
import pickle
import numpy as np
import networkx as nx
import torch
from tqdm import tqdm
from scipy.sparse import lil_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import TruncatedSVD

from utils import ricci_curvature_graph_augmentation


class DPFuncDataset:
    """蛋白质功能预测数据集。

    与 HypergraphDataset 保持相同接口，可复用 PPI2Complex 模型。
    使用 InterPro 域 SVD 降维特征替代 ESM 特征，
    使用 k-NN 相似图替代 PPI 图。
    """

    def __init__(self, args):
        self.args = args
        self.dataset_dir = os.path.join(args.data_path, args.dataset_name)

        # 与 HypergraphDataset 保持一致的接口
        self.graph_original: nx.Graph = None
        self.graph_augmented_rc: nx.Graph = None
        self.node_list: list[str] = []
        self.node_to_idx: dict[str, int] = {}
        self.num_nodes: int = 0
        self.hyperedges: list[frozenset[str]] = []
        self.H_clique: torch.sparse.FloatTensor = None
        self.num_hyperedges: int = 0
        self.node_features_esm: torch.Tensor = None  # 保持命名兼容，实际为 InterPro 特征
        self.feature_dim: int = 0

        self._load_and_process()

        if args.use_ricci_augmentation:
            self._apply_ricci_augmentation()

    # ------------------------------------------------------------------
    # 主流程
    # ------------------------------------------------------------------

    def _load_and_process(self):
        print("=" * 50)
        print(f"加载 DPFunc 数据集: {self.args.dataset_name}")
        print("=" * 50)

        self._load_raw_data()
        self._build_or_load_features()
        self._build_or_load_graph()
        self._build_hypergraph()

    def _load_raw_data(self):
        """加载原始数据文件。"""
        print("\n[1/4] 加载原始数据...")

        required = ['id_map.pkl', 'all_protein_interpros.pkl', 'inter_idx.pkl']
        for fn in required:
            fp = os.path.join(self.dataset_dir, fn)
            if not os.path.exists(fp):
                raise FileNotFoundError(f"缺少必需文件: {fp}")

        with open(os.path.join(self.dataset_dir, 'id_map.pkl'), 'rb') as f:
            self.id_map = pickle.load(f)
        with open(os.path.join(self.dataset_dir, 'all_protein_interpros.pkl'), 'rb') as f:
            self.interpro_dict = pickle.load(f)
        with open(os.path.join(self.dataset_dir, 'inter_idx.pkl'), 'rb') as f:
            self.inter_idx = pickle.load(f)

        self.node_list = sorted(self.id_map.keys())
        self.node_to_idx = {node: i for i, node in enumerate(self.node_list)}
        self.num_nodes = len(self.node_list)
        self.num_interpro = len(self.inter_idx)
        print(f"  蛋白质: {self.num_nodes}, InterPro 域: {self.num_interpro}")

    # ------------------------------------------------------------------
    # 特征构建
    # ------------------------------------------------------------------

    def _build_or_load_features(self):
        """构建 InterPro 稀疏特征 → SVD 降维 → 缓存。"""
        print("\n[2/4] 构建节点特征...")

        cache_path = os.path.join(
            self.dataset_dir,
            f'features_svd{self.args.dpfunc_svd_dim}.pt'
        )

        if os.path.exists(cache_path) and not self.args.force_regenerate_esm:
            print(f"  从缓存加载: {cache_path}")
            data = torch.load(cache_path, map_location='cpu', weights_only=True)
            self.node_features_esm = data['features']
            self.feature_dim = self.node_features_esm.shape[1]
            print(f"  特征形状: {self.node_features_esm.shape}")
            return

        # 构建稀疏二值特征矩阵
        print("  构建 InterPro 稀疏特征矩阵...")
        X = lil_matrix((self.num_nodes, self.num_interpro), dtype=np.float32)
        missing = 0
        for i, node in enumerate(tqdm(self.node_list, desc="  编码 InterPro")):
            ip_set = self.interpro_dict.get(node, set())
            if not ip_set:
                missing += 1
                continue
            idxs = [self.inter_idx[ip] for ip in ip_set if ip in self.inter_idx]
            if idxs:
                X[i, idxs] = 1.0
        X = X.tocsr()
        if missing > 0:
            print(f"  警告: {missing} 个蛋白质无 InterPro 域标注")
        print(f"  稀疏矩阵: {X.shape}, 非零元={X.nnz}, 稠密度={X.nnz/(X.shape[0]*X.shape[1])*100:.2f}%")

        # SVD 降维
        svd_dim = min(self.args.dpfunc_svd_dim, self.num_interpro, self.num_nodes - 1)
        print(f"  TruncatedSVD 降维 → {svd_dim}...")
        svd = TruncatedSVD(n_components=svd_dim, random_state=42)
        X_reduced = svd.fit_transform(X)
        print(f"  解释方差比合计: {svd.explained_variance_ratio_.sum():.4f}")

        self.node_features_esm = torch.from_numpy(X_reduced.astype(np.float32))
        self.feature_dim = svd_dim

        torch.save({'features': self.node_features_esm, 'node_list': self.node_list}, cache_path)
        print(f"  特征已缓存: {cache_path}")

    # ------------------------------------------------------------------
    # 图构建
    # ------------------------------------------------------------------

    def _build_or_load_graph(self):
        """从 SVD 特征构建 k-NN 相似图。"""
        print("\n[3/4] 构建相似图...")

        graph_path = os.path.join(
            self.dataset_dir,
            f'graph_k{self.args.dpfunc_knn_k}.pkl'
        )

        if os.path.exists(graph_path):
            print(f"  从缓存加载: {graph_path}")
            with open(graph_path, 'rb') as f:
                self.graph_original = pickle.load(f)
            print(f"  节点={self.graph_original.number_of_nodes()}, "
                  f"边={self.graph_original.number_of_edges()}")
            return

        k = min(self.args.dpfunc_knn_k + 1, self.num_nodes)
        print(f"  k-NN (k={self.args.dpfunc_knn_k}, cosine)...")
        features_np = self.node_features_esm.numpy()

        nbrs = NearestNeighbors(n_neighbors=k, metric='cosine', algorithm='brute', n_jobs=-1)
        nbrs.fit(features_np)
        distances, indices = nbrs.kneighbors(features_np)

        self.graph_original = nx.Graph()
        self.graph_original.add_nodes_from(self.node_list)

        for i, (nbr_idx, nbr_dist) in enumerate(tqdm(zip(indices, distances),
                                                       total=self.num_nodes, desc="  构建边")):
            for j, d in zip(nbr_idx, nbr_dist):
                if i == j:
                    continue
                sim = 1.0 - d / 2.0  # cosine_dist → cosine_sim
                if sim > 1e-6:
                    self.graph_original.add_edge(self.node_list[i], self.node_list[j], weight=float(sim))

        print(f"  节点={self.graph_original.number_of_nodes()}, "
              f"边={self.graph_original.number_of_edges()}")

        with open(graph_path, 'wb') as f:
            pickle.dump(self.graph_original, f)
        print(f"  图已缓存: {graph_path}")

    # ------------------------------------------------------------------
    # 超图构建
    # ------------------------------------------------------------------

    def _build_hypergraph(self):
        """从相似图的团簇构建超图。"""
        print("\n[4/4] 构建超图...")

        min_size = self.args.min_clique_size
        print(f"  查找团簇 (最小规模={min_size})...")
        try:
            self.hyperedges = [frozenset(map(str, c))
                               for c in nx.find_cliques(self.graph_original)
                               if len(c) >= min_size]
        except Exception as e:
            print(f"  警告: 团簇查找出错 ({e})，回退到使用边作为超边")
            self.hyperedges = [frozenset(map(str, (u, v)))
                               for u, v in self.graph_original.edges()]

        self.num_hyperedges = len(self.hyperedges)
        print(f"  超边数: {self.num_hyperedges}")

        if self.num_hyperedges > 0 and self.num_nodes > 0:
            rows, cols = [], []
            he_idx_map = {h: i for i, h in enumerate(self.hyperedges)}
            for node_id, nidx in self.node_to_idx.items():
                for he_obj, hidx in he_idx_map.items():
                    if node_id in he_obj:
                        rows.append(nidx)
                        cols.append(hidx)

            if rows:
                indices = torch.tensor([rows, cols], dtype=torch.long)
                values = torch.ones(len(rows), dtype=torch.float32)
                self.H_clique = torch.sparse_coo_tensor(
                    indices, values, (self.num_nodes, self.num_hyperedges)).coalesce()
            else:
                self.H_clique = torch.sparse_coo_tensor(
                    torch.empty((2, 0), dtype=torch.long), [],
                    (self.num_nodes, self.num_hyperedges))
        else:
            self.H_clique = torch.sparse_coo_tensor(
                size=(self.num_nodes, max(0, self.num_hyperedges)))

        nnz = self.H_clique._nnz() if self.H_clique is not None else 0
        print(f"  H 矩阵: {self.H_clique.shape}, 非零元={nnz}")

    # ------------------------------------------------------------------
    # Ricci 曲率增强
    # ------------------------------------------------------------------

    def _apply_ricci_augmentation(self):
        print("\n[可选] Ricci 曲率图增强...")
        if self.graph_original is None:
            raise ValueError("原始图未加载，无法进行 Ricci 增强。")
        self.graph_augmented_rc = ricci_curvature_graph_augmentation(
            self.graph_original,
            alpha=self.args.ricci_alpha,
            cutoff_min=self.args.ricci_cutoff_min,
            method=self.args.ricci_process_method
        )
        print(f"  Ricci 图: 节点={self.graph_augmented_rc.number_of_nodes()}, "
              f"边={self.graph_augmented_rc.number_of_edges()}")
