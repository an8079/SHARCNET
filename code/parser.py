# parser.py

import argparse
import torch


# 将 def parse_args(): 修改为 def parse_args(argv=None):
def parse_args(argv=None):
    """
    定义并解析命令行参数。
    返回:
        argparse.Namespace: 包含所有参数的对象。
    """
    parser = argparse.ArgumentParser(description="PPI2Complex: 深度融合超图语义对比与Ricci曲率增强的蛋白质嵌入框架")

    # --- 数据集与特征参数 ---
    # ... (这里的所有 parser.add_argument(...) 内容保持不变)
    parser.add_argument('--dataset_name', default='c_elegans', type=str,
                        help='数据集名称 (例如: c_elegans, HuRI, yeast, hy)')
    parser.add_argument('--data_path', default='../data', type=str,
                        help='数据集根目录路径 (相对于 code/ 目录)')
    parser.add_argument('--protein_sequence_file', default='protein_seq.tsv', type=str,
                        help='蛋白质序列文件名 (在数据集特定子目录中)')
    parser.add_argument('--edge_list_file', default='edge_list.csv', type=str,
                        help='PPI网络的边列表文件名 (在数据集特定子目录中)')
    parser.add_argument('--esm_model_name', default='facebook/esm2_t33_650M_UR50D', type=str,
                        help='HuggingFace/ModelScope上的ESM模型ID或本地ESM模型路径')
    parser.add_argument('--esm_embedding_dim', default=1280, type=int,
                        help='ESM特征维度 (与所选ESM模型一致)')
    parser.add_argument('--esm_batch_size', default=16, type=int,
                        help='生成ESM特征时的批处理大小')
    parser.add_argument('--precomputed_esm_path', default=None, type=str,
                        help='预计算的ESM特征文件路径 (.pt)')
    parser.add_argument('--force_regenerate_esm', action='store_true',
                        help='即使存在预计算文件也强制重新生成ESM特征')
    parser.add_argument('--feature_mask_rate', default=0.2, type=float,
                        help='为RC增强视图的输入特征进行随机掩码的比例 (0表示不掩码)')

    # --- 超图与Ricci曲率增强参数 ---
    parser.add_argument('--min_clique_size', default=2, type=int,
                        help='定义超图的超边时，所用团簇的最小规模')
    parser.add_argument('--use_ricci_augmentation', action='store_true', default=True,
                        help='是否使用Ricci曲率进行图增强')
    parser.add_argument('--ricci_alpha', default=0.5, type=float,
                        help='Ollivier-Ricci曲率计算中的alpha参数')
    parser.add_argument('--ricci_cutoff_min', default=-0.7, type=float,
                        help='Ricci曲率阈值，低于此值的边将被移除或权重降低 (参考TSCGC)')
    parser.add_argument('--ricci_process_method', default='remove', type=str, choices=['remove', 'weight'],
                        help="Ricci曲率处理边的方法：'remove'移除低曲率边, 'weight'根据曲率调整边权重")

    # --- PPI2Complex 模型参数 ---
    parser.add_argument('--embedding_dim', default=128, type=int,
                        help='GCN/HGNN最终输出的蛋白质嵌入维度')
    parser.add_argument('--gcn_hidden_dims', type=int, nargs='*', default=[256],
                        help='GCN编码器的隐藏层维度列表。最后一层输出embedding_dim。')
    parser.add_argument('--hgnn_hidden_dims', type=int, nargs='*', default=[256],
                        help='HGNN编码器的隐藏层维度列表。最后一层输出embedding_dim。')
    parser.add_argument('--num_gcn_layers', type=int, default=2,
                        help='GCN编码器的层数')
    parser.add_argument('--num_hgnn_layers', type=int, default=2,
                        help='HGNN编码器的层数')
    parser.add_argument('--dropout_rate', default=0.4, type=float,
                        help='模型中的dropout比率')
    parser.add_argument('--contrastive_temperature', default=0.07, type=float,
                        help='对比损失中的温度参数')
    parser.add_argument('--fusion_gamma', default=0.5, type=float,
                        help='融合 Zs 和 Zh 得到 Zc 时的权重参数 (用于特征重构)')

    # --- 损失函数权重 ---
    parser.add_argument('--lambda_hscl', default=1.0, type=float,
                        help='超图语义对比损失 (L_HSCL - SCL like) 的权重')
    parser.add_argument('--lambda_tcl', default=1.0, type=float,
                        help='拓扑对比损失 (L_TCL - G_orig vs G_rc_aug, hypergraph-neighbor sampled) 的权重')
    parser.add_argument('--lambda_align', default=0.5, type=float,
                        help='视图对齐损失 (L_Align - G_hyper vs G_rc_aug) 的权重')
    parser.add_argument('--lambda_recon_struct', default=0.1, type=float,
                        help='结构重构损失 (L_str) 的权重')
    parser.add_argument('--lambda_recon_feat', default=0.1, type=float,
                        help='特征重构损失 (L_feat) 的权重')

    # --- SCL (L_HSCL) 相关参数 (来自TSCGC) ---
    parser.add_argument('--scl_cluster_k_ratio', default=0.1, type=float,
                        help='SCL中K-Means的簇数量相对于总节点数的比例 (近似)')
    parser.add_argument('--scl_hcn_ratio', default=0.1, type=float,
                        help='SCL中每簇选为HCN的节点比例 (基于隶属度排序)')
    parser.add_argument('--scl_lcn_knn_k', default=10, type=int,
                        help='SCL中LCN选择正样本的KNN数量')
    parser.add_argument('--scl_update_pseudo_labels_every', default=10, type=int,
                        help='SCL中每隔多少epoch更新一次伪标签 (0表示不更新，即预计算一次)')

    # --- 训练参数 ---
    parser.add_argument('--epochs', default=10, type=int,
                        help='训练的总轮数')
    parser.add_argument('--learning_rate', default=1e-3, type=float,
                        help='优化器的学习率')
    parser.add_argument('--weight_decay', default=1e-5, type=float,
                        help='AdamW优化器的权重衰减')
    parser.add_argument('--contrastive_batch_size', default=1024, type=int,
                        help='对比学习中锚点采样批大小(用于TCL, SCL)')
    parser.add_argument('--gradient_clip_value', default=1.0, type=float,
                        help='梯度裁剪的阈值 (设为0或None则禁用)')
    parser.add_argument('--eval_every_epochs', default=20, type=int,
                        help='每隔多少轮评估一次模型性能 (并可能更新SCL伪标签)')

    # --- 评估参数 (链接预测) ---
    parser.add_argument('--link_pred_n_trials', default=5, type=int,
                        help='链接预测评估的独立试验次数')
    parser.add_argument('--link_pred_test_ratio', default=0.2, type=float,
                        help='链接预测中用于测试集的边比例')
    parser.add_argument('--link_pred_val_ratio', default=0.1, type=float,
                        help='链接预测中用于验证集的边比例 (从训练集中划分)')

    # --- 运行与环境参数 ---
    parser.add_argument('--seed', default=42, type=int,
                        help='全局随机种子，用于可复现性')
    parser.add_argument('--device', default=None, type=str,
                        help="指定设备 ('cuda' or 'cpu')。若不指定，则自动检测CUDA。")

    # 将 parser.parse_args() 修改为 parser.parse_args(argv)
    args = parser.parse_args(argv)

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if not args.use_ricci_augmentation:
        args.lambda_tcl = 0.0;
        args.lambda_align = 0.0;
        args.feature_mask_rate = 0.0
        print("警告：Ricci曲率增强未启用，L_TCL, L_Align 及特征掩码已自动禁用。")
    if args.link_pred_test_ratio + args.link_pred_val_ratio >= 1.0:
        raise ValueError("错误：链接预测的测试集和验证集比例之和必须小于1.0。")
    # 动态调整gcn和hgnn的隐藏层，如果只指定了层数但未指定维度
    if args.num_gcn_layers > 0 and not args.gcn_hidden_dims:
        args.gcn_hidden_dims = [args.embedding_dim] * (args.num_gcn_layers - 1)
    if args.num_hgnn_layers > 0 and not args.hgnn_hidden_dims:
        args.hgnn_hidden_dims = [args.embedding_dim] * (args.num_hgnn_layers - 1)

    print(f"最终将使用设备: {args.device}")
    return args
