# main.py

import torch
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import os
import time
import pandas as pd
import networkx as nx
import json
import traceback

from parser import parse_args
from dataset import HypergraphDataset
from model import HyperDNERC2
from utils import (set_global_seed, split_graph_edges, LinkPredictor,
                   networkx_to_torch_sparse_adj, normalize_adjacency_matrix, get_pseudo_labels_and_hcn_lcn)


def main(args):
    """
    执行一次完整的 SHARCNet 训练和评估流程。

    Args:
        args (argparse.Namespace): 包含所有配置参数的对象。

    Returns:
        dict: 一个包含链接预测任务平均性能指标及其标准差的字典。
              如果评估未执行或失败，则返回一个空字典。
    """
    start_run_time = time.time()
    set_global_seed(args.seed)
    device = torch.device(args.device)
    print(f"使用设备: {device}")
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
        print(f"数据集初始化/图转换严重错误: {e}")
        traceback.print_exc()
        return {}  # 返回空字典表示失败

    print(f"最终节点特征维度: {dataset_obj.feature_dim}")

    print("\n===== 步骤 2: 初始化模型和优化器 =====")
    model = HyperDNERC2(args, initial_feature_dim=dataset_obj.feature_dim).to(device)
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
                        if model.scl_pseudo_labels is not None and epoch % args.eval_every_epochs == 0:
                            print(f"SCL伪标签已在Epoch {epoch} 更新。")

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
                if epoch % args.eval_every_epochs == 0:
                    print(f"Epoch {epoch} 总损失为0且不要求梯度，跳过优化步骤。")

        if epoch % args.eval_every_epochs == 0 or epoch == args.epochs:
            loss_val = total_loss.item() if isinstance(total_loss,
                                                       torch.Tensor) and total_loss.numel() > 0 and total_loss.requires_grad else (
                total_loss if isinstance(total_loss, float) else 0.0)
            loss_str = f"总损失: {loss_val:.4f}"
            for name, val_item in loss_breakdown.items():
                loss_str += f", {name}: {val_item:.4f}"
            print(f"Epoch [{epoch}/{args.epochs}], {loss_str}")

    print("模型训练完成。")

    print("\n===== 步骤 4: 生成最终嵌入并评估 =====")
    model.eval()
    with torch.no_grad():
        final_embed_dict_tensor = model.encode(node_features_esm, adj_orig_norm_for_encoder, adj_rc_norm_for_encoder,
                                               H_clique_torch)
        final_embed_tensor = model.get_final_embeddings(final_embed_dict_tensor)
        if final_embed_tensor is None or final_embed_tensor.numel() == 0:
            print("错误：未能生成有效最终嵌入。")
            return {}  # 返回空字典
        final_embed_np = final_embed_tensor.cpu().numpy()

    final_embed_map = {node_id: final_embed_np[idx] for node_id, idx in dataset_obj.node_to_idx.items()}

    if final_embed_map:
        print(f"已生成最终嵌入，形状: ({len(final_embed_map)}, {final_embed_np.shape[1]})")
        if dataset_obj.graph_original.number_of_edges() >= 10:
            print("\n--- 开始链接预测评估 ---")
            all_trial_results = []
            for i in range(args.link_pred_n_trials):
                trial_seed = args.seed + i
                print(f"\n链接预测试验 {i + 1}/{args.link_pred_n_trials} (种子: {trial_seed})...")
                g_train, (X_train, Y_train), (X_val, Y_val), (X_test, Y_test) = split_graph_edges(
                    dataset_obj.graph_original, test_ratio=args.link_pred_test_ratio,
                    val_ratio=args.link_pred_val_ratio, seed=trial_seed)

                if len(X_train) < 2 or len(X_test) < 2 or (len(Y_train) > 0 and len(np.unique(Y_train)) < 2):
                    print(f"警告:试验 {i + 1}样本不足或类别单一，跳过。")
                    continue

                link_predictor = LinkPredictor(embeddings=final_embed_map, binary_operator_name='hadamard',
                                               seed=trial_seed)
                if not link_predictor.train(X_train, Y_train):
                    print(f"警告:试验 {i + 1}链接预测模型训练失败。")
                    continue

                test_scores = link_predictor.evaluate(X_test, Y_test, "测试集")
                print(f"试验 {i + 1} 测试集: {test_scores}")
                test_scores['trial'] = i + 1
                all_trial_results.append(test_scores)

            # **核心修改：计算并返回指定的平均指标和标准差**
            if all_trial_results:
                results_df = pd.DataFrame(all_trial_results)
                print("\n--- 链接预测平均性能 (测试集) ---")

                # 定义需要报告的指标及其在CSV文件中的列名
                # 键: DataFrame中的列名, 值: 输出到CSV的列名
                metrics_map = {
                    'auc_roc': 'AUC',
                    'auc_pr': 'AUPR',
                    'f1': 'F1',
                    'accuracy': 'ACC'
                }

                output_scores = {}

                for key_in_df, col_name_in_csv in metrics_map.items():
                    # 确保DataFrame中存在该列
                    if key_in_df in results_df.columns:
                        mean_val = results_df[key_in_df].mean()
                        std_val = results_df[key_in_df].std()

                        # 1. 在控制台打印详细信息
                        print(f" 平均 {col_name_in_csv}: {mean_val:.4f} ± {std_val:.4f}")

                        # 2. 准备要返回并保存到CSV的数据
                        #    格式为 mean_AUC, std_AUC, mean_AUPR, std_AUPR ...
                        output_scores[f'mean_{col_name_in_csv}'] = mean_val
                        output_scores[f'std_{col_name_in_csv}'] = std_val

                end_run_time = time.time()
                print(f"\n单次运行时间: {end_run_time - start_run_time:.2f} 秒")
                print("=" * 40)

                # 返回包含指定指标的均值和标准差的字典
                return output_scores
            else:
                print("所有链接预测评估试验均失败。")
                return {}  # 如果没有结果，返回空字典
        else:
            print("警告：图中边数过少，跳过链接预测。")
            return {}
    else:
        print("错误：未能生成最终嵌入。")
        return {}


# 当此文件被直接运行时，执行默认流程
if __name__ == "__main__":
    args = parse_args()
    print("===== HyperDNE-RC² (完整版) 单次运行模式 =====")
    print(json.dumps(vars(args), indent=2, ensure_ascii=False))
    print("========================================")
    main(args)