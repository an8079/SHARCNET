# dataset.py
import os
import pandas as pd
import networkx as nx
import torch
import numpy as np
from tqdm import tqdm
import re

try:
    import transformers
    from transformers import AutoTokenizer, AutoModel

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("关键错误: 未找到 'transformers' 库。ESM特征生成将失败。程序将无法运行。")
    TRANSFORMERS_AVAILABLE = False

try:
    from modelscope import snapshot_download
    MODELSCOPE_AVAILABLE = True
except ImportError:
    MODELSCOPE_AVAILABLE = False

from utils import ricci_curvature_graph_augmentation  # 从utils导入


class HypergraphDataset:
    def __init__(self, args):
        self.args = args
        self.dataset_dir = os.path.join(args.data_path, args.dataset_name)

        self.graph_original: nx.Graph = None
        self.adj_orig_norm: torch.sparse.FloatTensor = None  # 归一化后的原始图邻接矩阵 (GCN用)

        self.graph_augmented_rc: nx.Graph = None
        self.adj_rc_norm: torch.sparse.FloatTensor = None  # 归一化后的RC增强图邻接矩阵 (GCN用)

        self.node_list: list[str] = []
        self.node_to_idx: dict[str, int] = {}
        self.num_nodes: int = 0

        self.hyperedges: list[frozenset[str]] = []
        self.H_clique: torch.sparse.FloatTensor = None  # 基于团簇的超图关联矩阵 (HGNN用)
        self.num_hyperedges: int = 0

        self.node_features_esm: torch.Tensor = None  # ESM特征
        self.feature_dim: int = 0

        if not TRANSFORMERS_AVAILABLE:  # 在初始化早期就检查
            raise ImportError("关键依赖 'transformers' 未安装。无法继续。")

        self._load_graph_and_build_hypergraph()
        self._load_or_generate_esm_features()  # 强制成功或报错

        if args.use_ricci_augmentation:
            self._apply_ricci_augmentation()

    def _load_graph_and_build_hypergraph(self):
        edge_file = os.path.join(self.dataset_dir, self.args.edge_list_file)
        print(f"步骤1.1: 从 {edge_file} 加载原始PPI图...")
        if not os.path.exists(edge_file):
            raise FileNotFoundError(f"错误: 边列表文件未找到于 {edge_file}")

        try:
            df = pd.read_csv(edge_file)
            required_cols = ['source', 'target']
            if not all(col in df.columns for col in required_cols):
                print(f"警告: '{edge_file}' 缺少 'source' 或 'target' 列名。假设前两列为source/target。")
                df = pd.read_csv(edge_file, header=None)
                df = df.iloc[:, :2];
                df.columns = required_cols

            df[required_cols[0]] = df[required_cols[0]].astype(str)
            df[required_cols[1]] = df[required_cols[1]].astype(str)
            self.graph_original = nx.from_pandas_edgelist(df, 'source', 'target')
        except Exception as e:
            print(f"错误: 加载图文件失败: {e}");
            raise

        self.graph_original = nx.Graph(self.graph_original)
        self.graph_original.remove_edges_from(list(nx.selfloop_edges(self.graph_original)))

        if self.graph_original.number_of_nodes() == 0:
            raise ValueError("错误: 加载的图为空或无节点。")

        self.node_list = sorted([str(node) for node in self.graph_original.nodes()])
        self.node_to_idx = {node: i for i, node in enumerate(self.node_list)}
        self.num_nodes = len(self.node_list)
        print(f"原始图加载完成: {self.num_nodes} 个节点, {self.graph_original.number_of_edges()} 条边。")

        # 构建基于团簇的超图
        print(f"步骤1.2: 构建基于团簇的超图 (最小团簇规模: {self.args.min_clique_size})...")
        try:
            clique_iterator = nx.find_cliques(self.graph_original)
            self.hyperedges = [frozenset(map(str, c)) for c in clique_iterator if len(c) >= self.args.min_clique_size]
        except Exception as e:
            print(f"错误: 查找团簇失败: {e}.");
            self.hyperedges = []

        self.num_hyperedges = len(self.hyperedges)
        print(f"发现 {self.num_hyperedges} 个超边。")

        if self.num_hyperedges > 0 and self.num_nodes > 0:
            rows, cols = [], []
            hyperedge_idx_map = {h: i for i, h in enumerate(self.hyperedges)}
            for node_original_id, node_idx_mapped in self.node_to_idx.items():
                for hyperedge_obj, h_mapped_idx in hyperedge_idx_map.items():
                    if node_original_id in hyperedge_obj:
                        rows.append(node_idx_mapped);
                        cols.append(h_mapped_idx)

            if not rows:
                self.H_clique = torch.sparse_coo_tensor(torch.empty((2, 0), dtype=torch.long), [],
                                                        (self.num_nodes, self.num_hyperedges))
            else:
                indices = torch.tensor([rows, cols], dtype=torch.long)
                values = torch.ones(len(rows), dtype=torch.float32)
                self.H_clique = torch.sparse_coo_tensor(indices, values,
                                                        (self.num_nodes, self.num_hyperedges)).coalesce()
        else:
            self.H_clique = torch.sparse_coo_tensor(size=(self.num_nodes, max(0, self.num_hyperedges)))
        print(
            f"超图关联矩阵 H_clique 构建完成: 形状 {self.H_clique.shape}, 非零元数量 {self.H_clique._nnz() if self.H_clique is not None else 0}")

    def _get_sequences_from_tsv(self, sequence_tsv_path: str) -> dict[str, str] | None:
        """从TSV文件加载蛋白质序列。 (与上一版类似，保持健壮性)"""
        print(f"--- 尝试从TSV加载序列: {sequence_tsv_path} ---")
        if not os.path.exists(sequence_tsv_path):
            print(f"错误: 蛋白质序列TSV文件 '{sequence_tsv_path}' 未找到。")
            return None  # 将导致上层函数报错

        sequences_from_file = {}
        # 尝试更通用的ID列名，VEuPathDB比较特殊
        # 常见的ID列名可能是 'Entry', 'Protein ID', 'Gene ID', or the first column if no header
        # 假设TSV第一列是ID，第二列是序列，或者有明确的'Entry'/'Sequence'/'VEuPathDB'列

        try:
            df_check_header = pd.read_csv(sequence_tsv_path, sep=None, nrows=0, engine='python')  # sep=None 自动检测
            header = df_check_header.columns.tolist()

            id_col, seq_col = None, None
            if 'Entry' in header and 'Sequence' in header:
                id_col, seq_col = 'Entry', 'Sequence'
            elif 'VEuPathDB' in header and 'Sequence' in header:  # 保留对VEuPathDB的支持
                id_col, seq_col = 'VEuPathDB', 'Sequence'
            elif len(header) >= 2:  # 尝试前两列
                id_col, seq_col = header[0], header[1]
                print(f"警告: 未找到明确的ID/序列列名，尝试使用列 '{id_col}' 作为ID, '{seq_col}' 作为序列。")
            else:
                raise ValueError("TSV文件列数不足或列名不符合预期。")

            seq_df = pd.read_csv(sequence_tsv_path, sep=None, engine='python', dtype=str, usecols=[id_col, seq_col])

            # VEuPathDB列可能包含多个分号分隔的ID，或者需要从中提取Ensembl/UniProt ID
            # UniProt ID通常是P... Q... O... A-N R-Z...
            ensembl_uniprot_regex = re.compile(
                r"(ENSG\d{11}|FBgn\d{7}|[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2})")

            for _, row in tqdm(seq_df.iterrows(), total=len(seq_df), desc="解析TSV并映射序列"):
                raw_id_field = row.get(id_col)
                sequence = row.get(seq_col)

                if pd.isna(sequence) or pd.isna(raw_id_field):
                    continue

                sequence_cleaned = str(sequence).upper().replace("-", "").replace(" ", "").replace("*", "")

                # 尝试匹配node_list中的ID
                # 逻辑：如果raw_id_field直接在node_list中，用它
                # 否则，如果raw_id_field包含多个潜在ID（如VEuPathDB），则拆分并匹配
                # 否则，尝试对raw_id_field本身进行正则匹配

                found_match = False
                if raw_id_field in self.node_to_idx:
                    sequences_from_file[raw_id_field] = sequence_cleaned
                    found_match = True

                if not found_match:
                    potential_ids_in_field = [pid.strip() for pid in str(raw_id_field).split(';') if pid.strip()]
                    for pid in potential_ids_in_field:
                        if pid in self.node_to_idx:
                            sequences_from_file[pid] = sequence_cleaned
                            found_match = True
                            break

                if not found_match:  # 如果直接ID和分号分隔ID都未匹配，尝试正则
                    matches = ensembl_uniprot_regex.findall(str(raw_id_field))
                    for match_tuple in matches:
                        for found_id_in_regex in match_tuple:
                            if found_id_in_regex and found_id_in_regex in self.node_to_idx:
                                sequences_from_file[found_id_in_regex] = sequence_cleaned
                                found_match = True
                                break
                        if found_match: break

        except Exception as e:
            print(f"错误: 处理蛋白质序列TSV文件失败: {e}")
            import traceback;
            traceback.print_exc()
            return None  # 将导致上层函数报错

        final_sequences = {node_id: seq for node_id, seq in sequences_from_file.items() if node_id in self.node_to_idx}
        loaded_count = len(final_sequences)

        print(f"成功为 {loaded_count}/{self.num_nodes} 个图节点映射了序列。")
        if loaded_count == 0 and self.num_nodes > 0:
            print("错误: 加载的序列与图节点匹配数为0。")  # 改为错误
            return None

        missing_seq = self.num_nodes - loaded_count
        if missing_seq > 0:  # 这是一个严重问题，因为我们不再回退
            print(f"错误: {missing_seq} 个图节点缺少序列/未映射序列。ESM特征将不完整。")
            # 根据新要求，这里应该导致失败
            return None  # 表示部分失败，上层会处理

        return final_sequences

    def _load_esm_model(self, model_path_or_id: str):
        """
        加载ESM模型。优先尝试本地路径和HuggingFace，失败后自动尝试ModelScope下载。

        Args:
            model_path_or_id: 本地路径、HuggingFace模型ID或ModelScope模型ID

        Returns:
            (tokenizer, model) 元组
        """
        # 1. 如果是本地已存在的目录，直接加载
        if os.path.isdir(model_path_or_id):
            print(f"从本地路径加载ESM模型: {model_path_or_id}")
            tokenizer = AutoTokenizer.from_pretrained(model_path_or_id)
            model = AutoModel.from_pretrained(model_path_or_id).to(self.args.device)
            model.eval()
            print("ESM模型加载成功。")
            return tokenizer, model

        # 2. 尝试从HuggingFace加载
        print(f"尝试从HuggingFace Hub加载ESM模型: {model_path_or_id}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path_or_id)
            model = AutoModel.from_pretrained(model_path_or_id).to(self.args.device)
            model.eval()
            print("ESM模型从HuggingFace加载成功。")
            return tokenizer, model
        except Exception as hf_error:
            print(f"HuggingFace加载失败: {hf_error}")

        # 3. HuggingFace失败，尝试从ModelScope下载
        # 将HuggingFace模型名映射到ModelScope模型名
        modelscope_model_id = model_path_or_id
        if model_path_or_id == 'facebook/esm2_t33_650M_UR50D':
            modelscope_model_id = 'AI4Science/esm2_t33_650M_UR50D'

        print(f"\n尝试从ModelScope下载ESM模型: {modelscope_model_id}")
        if not MODELSCOPE_AVAILABLE:
            raise RuntimeError(
                f"错误: HuggingFace和ModelScope均无法加载ESM模型。\n"
                f"  HuggingFace失败原因: 见上方错误\n"
                f"  ModelScope未安装。请安装: pip install modelscope\n"
                f"  或者手动下载模型到本地后通过 --esm_model_name 指定本地路径。"
            )

        try:
            # ModelScope snapshot_download 会下载模型到本地缓存并返回本地路径
            cache_dir = os.path.join(self.args.data_path, '.model_cache')
            os.makedirs(cache_dir, exist_ok=True)
            print(f"ModelScope模型将缓存到: {cache_dir}")
            local_model_path = snapshot_download(
                modelscope_model_id,
                cache_dir=cache_dir
            )
            print(f"ModelScope下载完成，本地路径: {local_model_path}")
            tokenizer = AutoTokenizer.from_pretrained(local_model_path)
            model = AutoModel.from_pretrained(local_model_path).to(self.args.device)
            model.eval()
            print("ESM模型从ModelScope加载成功。")
            return tokenizer, model
        except Exception as ms_error:
            raise RuntimeError(
                f"错误: HuggingFace和ModelScope均无法加载ESM模型。\n"
                f"  HuggingFace错误: {hf_error}\n"
                f"  ModelScope错误: {ms_error}\n"
                f"  请手动下载模型 '{model_path_or_id}' 到本地，然后通过 --esm_model_name 指定本地路径。"
            )

    def _load_or_generate_esm_features(self):
        """加载或生成节点的ESM特征。如果失败，则抛出错误。"""
        print(f"步骤1.3: 加载/生成ESM节点特征...")

        sanitized_model_name = self.args.esm_model_name.replace('/', '_').replace('-', '_')
        default_feature_filename = f"{self.args.dataset_name}_{sanitized_model_name}_dim{self.args.esm_embedding_dim}_embeddings.pt"

        # 优先使用用户指定的预计算路径
        if self.args.precomputed_esm_path:
            feature_file = self.args.precomputed_esm_path
        else:  # 否则构造默认路径
            feature_file = os.path.join(self.dataset_dir, default_feature_filename)

        os.makedirs(os.path.dirname(feature_file), exist_ok=True)

        if os.path.exists(feature_file) and not self.args.force_regenerate_esm:
            print(f"尝试从 {feature_file} 加载预计算的ESM特征...")
            try:
                loaded_data = torch.load(feature_file, map_location=torch.device('cpu'))
                if isinstance(loaded_data,
                              dict) and 'node_features' in loaded_data and 'node_list_stored' in loaded_data:
                    if loaded_data['node_list_stored'] == self.node_list:
                        self.node_features_esm = loaded_data['node_features']
                        if self.node_features_esm.shape[0] != self.num_nodes:
                            raise ValueError(
                                f"预计算特征节点数 ({self.node_features_esm.shape[0]}) 与图 ({self.num_nodes}) 不匹配。")
                        if self.node_features_esm.shape[1] != self.args.esm_embedding_dim:
                            raise ValueError(
                                f"预计算特征维度 ({self.node_features_esm.shape[1]}) 与期望 ({self.args.esm_embedding_dim}) 不匹配。请删除或指定正确的预计算文件，或调整参数。")
                        print("预计算ESM特征加载成功。")
                        self.feature_dim = self.node_features_esm.shape[1]
                        self.node_features_esm = self.node_features_esm.to(self.args.device)
                        return  # 成功加载
                    else:
                        print("警告: 预计算特征的节点列表与当前图不匹配。将重新生成。")
                        self.node_features_esm = None
                else:
                    print(f"警告: 预计算文件 {feature_file} 格式不正确。将重新生成。")
                    self.node_features_esm = None
            except Exception as e:
                print(f"错误: 加载预计算ESM特征文件失败: {e}。将尝试重新生成。")
                self.node_features_esm = None

        # 如果没有加载预计算文件，或强制重新生成
        print(f"使用蛋白质序列文件 '{self.args.protein_sequence_file}' 生成ESM特征...")
        sequence_file_full_path = os.path.join(self.dataset_dir, self.args.protein_sequence_file)

        sequences = self._get_sequences_from_tsv(sequence_file_full_path)
        if not sequences or len(sequences) < self.num_nodes:  # 确保所有节点的序列都已加载
            # _get_sequences_from_tsv 内部会打印更详细的缺失节点警告
            raise RuntimeError(f"未能为图中的所有 {self.num_nodes} 个节点获取有效序列。ESM特征生成中止。")

        print(f"使用 '{self.args.esm_model_name}' 生成ESM特征 (目标维度: {self.args.esm_embedding_dim})...")
        model_path_or_id = self.args.esm_model_name
        tokenizer, model = self._load_esm_model(model_path_or_id)

        all_embeddings = torch.zeros((self.num_nodes, self.args.esm_embedding_dim),
                                     device=self.args.device, dtype=torch.float32)
        processed_nodes_count = 0

        with torch.no_grad():
            for i in tqdm(range(0, self.num_nodes, self.args.esm_batch_size), desc="生成ESM嵌入"):
                batch_node_original_ids = self.node_list[i: i + self.args.esm_batch_size]
                batch_sequences_strings = []
                valid_global_indices_for_embedding = []

                for node_id_str in batch_node_original_ids:
                    seq_str = sequences.get(node_id_str)  # sequences现在应该包含所有图节点的序列
                    if seq_str:  # 理论上这里所有节点都应该有序列
                        batch_sequences_strings.append(seq_str)
                        valid_global_indices_for_embedding.append(self.node_to_idx[node_id_str])
                    # else: # 这不应该发生，因为上面检查了 sequences 是否包含所有节点
                    # print(f"严重内部错误：节点 {node_id_str} 在序列映射中丢失。")

                if not batch_sequences_strings: continue

                try:
                    tokens = tokenizer(batch_sequences_strings, return_tensors='pt',
                                       padding=True, truncation=True, max_length=1022).to(
                        self.args.device)  # ESM通常最大长度1024，减去特殊token
                    outputs = model(**tokens)
                    token_embeds = outputs.last_hidden_state

                    attn_mask = tokens['attention_mask'].unsqueeze(-1).expand(token_embeds.size()).float()
                    sum_embeds = torch.sum(token_embeds * attn_mask, 1)
                    sum_mask = attn_mask.sum(1).clamp(min=1e-9)
                    mean_embeds = sum_embeds / sum_mask

                    if mean_embeds.shape[1] != self.args.esm_embedding_dim:
                        raise ValueError(
                            f"ESM输出维度 {mean_embeds.shape[1]} 与期望维度 {self.args.esm_embedding_dim} 不符。请检查ESM模型和参数。")

                    all_embeddings[valid_global_indices_for_embedding] = mean_embeds
                    processed_nodes_count += len(valid_global_indices_for_embedding)
                except Exception as e:
                    raise RuntimeError(f"\nESM批处理 {i // self.args.esm_batch_size} 失败: {e}")

        if processed_nodes_count < self.num_nodes:
            raise RuntimeError(f"ESM特征生成不完整: 仅为 {processed_nodes_count}/{self.num_nodes} 个节点生成了特征。")

        self.node_features_esm = all_embeddings
        self.feature_dim = self.node_features_esm.shape[1]
        print(f"ESM特征已成功生成并加载: 形状 {self.node_features_esm.shape}")

        try:
            torch.save({'node_features': self.node_features_esm.cpu(), 'node_list_stored': self.node_list},
                       feature_file)
            print(f"新生成的ESM特征已保存到: {feature_file}")
        except Exception as e:
            print(f"警告: 保存新生成的ESM特征失败: {e}")

    def _apply_ricci_augmentation(self):
        """应用Ricci曲率图增强 (如果启用)。"""
        print(f"步骤1.4: 应用Ricci曲率图增强 (方法: {self.args.ricci_process_method})...")
        if self.graph_original is None:
            raise ValueError("错误: 原始图未加载，无法进行Ricci曲率增强。")

        self.graph_augmented_rc = ricci_curvature_graph_augmentation(
            self.graph_original,
            alpha=self.args.ricci_alpha,
            cutoff_min=self.args.ricci_cutoff_min,
            method=self.args.ricci_process_method
        )
        print(
            f"Ricci曲率增强图构建完成: {self.graph_augmented_rc.number_of_nodes()} 个节点, {self.graph_augmented_rc.number_of_edges()} 条边。")