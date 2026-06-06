# main_dpfunc.py
"""DPFunc 数据集训练入口。

使用 InterPro 域特征 + k-NN 相似图 + SHARCNet 模型进行蛋白质嵌入学习。
"""

import os
import sys
import time
import json
import traceback
import argparse

import torch
import torch.optim as optim
import numpy as np
import pandas as pd
import networkx as nx

from parser import parse_args as parse_base_args
from dataset_dpfunc import DPFuncDataset
from model import HyperDNERC2
from utils import (
    set_global_seed, split_graph_edges, LinkPredictor,
    networkx_to_torch_sparse_adj, normalize_adjacency_matrix,
    get_pseudo_labels_and_hcn_lcn
)


def parse_args(argv=None):
    """扩展基础参数，添加 DPFunc 特有参数。"""
    parser = argparse.ArgumentParser(
        description="SHARCNet + DPFunc 蛋白质功能预测数据集训练",
        parents=[argparse.ArgumentParser(add_help=False)],  # 先创建空壳
        conflict_handler='resolve'
    )

    # ---- DPFunc 专用参数 ----
    parser.add_argument('--dpfunc_knn_k', default=10, type=int,
                        help='k-NN 相似图的邻居数')
    parser.add_argument('--dpfunc_svd_dim', default=1280, type=int,
                        help='InterPro 特征 SVD 降维目标维度')
    parser.add_argument('--dpfunc_namespace', default='bp', type=str,
                        choices=['bp', 'cc', 'mf'],
                        help='GO 功能注释命名空间 (bp/cc/mf)')

    # ---- 基础参数 (继承自 parser.py 的默认值) ----
    parser.add_argument('--dataset_name', default='data_dpfunc', type=str)
    parser.add_argument('--data_path', default='../data', type=str)
    parser.add_argument('--esm_model_name', default='', type=str,
                        help='DPFunc 不使用 ESM，忽略此参数')
    parser.add_argument('--esm_embedding_dim', default=1280, type=int)
    parser.add_argument('--esm_batch_size', default=16, type=int)
    parser.add_argument('--precomputed_esm_path', default=None, type=str)
    parser.add_argument('--force_regenerate_esm', action='store_true')
    parser.add_argument('--feature_mask_rate', default=0.2, type=float)
    parser.add_argument('--min_clique_size', default=2, type=int,
                        help='超边最小团簇规模 (DPFunc 默认 2)')
    parser.add_argument('--use_ricci_augmentation', action='store_true', default=True)
    parser.add_argument('--ricci_alpha', default=0.5, type=float)
    parser.add_argument('--ricci_cutoff_min', default=-0.7, type=float)
    parser.add_argument('--ricci_process_method', default='remove', type=str,
                        choices=['remove', 'weight'])
    parser.add_argument('--embedding_dim', default=128, type=int)
    parser.add_argument('--gcn_hidden_dims', type=int, nargs='*', default=[256])
    parser.add_argument('--hgnn_hidden_dims', type=int, nargs='*', default=[256])
    parser.add_argument('--num_gcn_layers', type=int, default=2)
    parser.add_argument('--num_hgnn_layers', type=int, default=2)
    parser.add_argument('--dropout_rate', default=0.4, type=float)
    parser.add_argument('--contrastive_temperature', default=0.07, type=float)
    parser.add_argument('--fusion_gamma', default=0.5, type=float)
    parser.add_argument('--lambda_hscl', default=1.0, type=float)
    parser.add_argument('--lambda_tcl', default=1.0, type=float)
    parser.add_argument('--lambda_align', default=0.5, type=float)
    parser.add_argument('--lambda_recon_struct', default=0.1, type=float)
    parser.add_argument('--lambda_recon_feat', default=0.1, type=float)
    parser.add_argument('--scl_cluster_k_ratio', default=0.05, type=float,
                        help='SCL 簇数占节点比例 (DPFunc默认0.05)')
    parser.add_argument('--scl_hcn_ratio', default=0.1, type=float)
    parser.add_argument('--scl_lcn_knn_k', default=10, type=int)
    parser.add_argument('--scl_update_pseudo_labels_every', default=10, type=int)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--learning_rate', default=1e-3, type=float)
    parser.add_argument('--weight_decay', default=1e-5, type=float)
    parser.add_argument('--contrastive_batch_size', default=1024, type=int)
    parser.add_argument('--gradient_clip_value', default=1.0, type=float)
    parser.add_argument('--eval_every_epochs', default=20, type=int)
    parser.add_argument('--link_pred_n_trials', default=5, type=int)
    parser.add_argument('--link_pred_test_ratio', default=0.2, type=float)
    parser.add_argument('--link_pred_val_ratio', default=0.1, type=float)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--device', default=None, type=str)
    parser.add_argument('--protein_sequence_file', default='', type=str)
    parser.add_argument('--edge_list_file', default='', type=str)

    args = parser.parse_args(argv)

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if not args.use_ricci_augmentation:
        args.lambda_tcl = 0.0
        args.lambda_align = 0.0
        args.feature_mask_rate = 0.0
        print("Ricci 曲率增强已禁用，TCL/Align/特征掩码自动置零。")
    if args.link_pred_test_ratio + args.link_pred_val_ratio >= 1.0:
        raise ValueError("link_pred_test_ratio + link_pred_val_ratio 必须 < 1.0")
    if args.num_gcn_layers > 0 and not args.gcn_hidden_dims:
        args.gcn_hidden_dims = [args.embedding_dim] * (args.num_gcn_layers - 1)
    if args.num_hgnn_layers > 0 and not args.hgnn_hidden_dims:
        args.hgnn_hidden_dims = [args.embedding_dim] * (args.num_hgnn_layers - 1)

    print(f"设备: {args.device}")
    return args


def main(args):
    start_run_time = time.time()
    set_global_seed(args.seed)
    device = torch.device(args.device)
    print(f"使用设备: {device}")

    # ----------------------------------------------------------------
    # 步骤 1: 加载数据集
    # ----------------------------------------------------------------
    print("\n===== 步骤 1: 加载 DPFunc 数据集 =====")
    try:
        dataset_obj = DPFuncDataset(args)
        adj_orig_torch_sparse = networkx_to_torch_sparse_adj(
            dataset_obj.graph_original, dataset_obj.node_to_idx, device)
        adj_orig_norm_for_encoder = normalize_adjacency_matrix(
            adj_orig_torch_sparse, add_self_loops=True)

        adj_orig_dense_target = nx.to_numpy_array(
            dataset_obj.graph_original, nodelist=dataset_obj.node_list, dtype=np.float32)
        adj_orig_dense_target = torch.from_numpy(adj_orig_dense_target).to(device)

        adj_rc_norm_for_encoder = None
        if args.use_ricci_augmentation and dataset_obj.graph_augmented_rc:
            adj_rc_sparse = networkx_to_torch_sparse_adj(
                dataset_obj.graph_augmented_rc, dataset_obj.node_to_idx, device)
            adj_rc_norm_for_encoder = normalize_adjacency_matrix(adj_rc_sparse, add_self_loops=True)

        H_clique_torch = dataset_obj.H_clique.to(device) if dataset_obj.H_clique is not None else None
        node_features = dataset_obj.node_features_esm.to(device).float()
    except Exception as e:
        print(f"数据集加载失败: {e}")
        traceback.print_exc()
        return {}
    print(f"节点特征维度: {dataset_obj.feature_dim}")

    # ----------------------------------------------------------------
    # 步骤 2: 初始化模型
    # ----------------------------------------------------------------
    print("\n===== 步骤 2: 初始化模型和优化器 =====")
    model = HyperDNERC2(args, initial_feature_dim=dataset_obj.feature_dim).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # ----------------------------------------------------------------
    # 步骤 3: 训练
    # ----------------------------------------------------------------
    print("\n===== 步骤 3: 训练 =====")

    # SCL 初始化
    if args.lambda_hscl > 0 and args.scl_update_pseudo_labels_every > 0:
        print("SCL: 初始化伪标签...")
        with torch.no_grad():
            temp_emb = model.encode(node_features, adj_orig_norm_for_encoder,
                                    adj_rc_norm_for_encoder, H_clique_torch)
            init_emb = model.get_final_embeddings(temp_emb)
            if init_emb is None and node_features.shape[0] > 0:
                init_emb = model.initial_projection(node_features)
            if init_emb is not None and init_emb.numel() > 0:
                num_clu = max(2, int(dataset_obj.num_nodes * args.scl_cluster_k_ratio))
                if dataset_obj.num_nodes < num_clu:
                    num_clu = max(1, dataset_obj.num_nodes // 2)
                if num_clu == 1 and dataset_obj.num_nodes > 1:
                    num_clu = 2
                if dataset_obj.num_nodes >= 1 and num_clu > 0:
                    model.scl_pseudo_labels, model.scl_hcn_indices, \
                        model.scl_lcn_indices, model.scl_centroids = \
                        get_pseudo_labels_and_hcn_lcn(
                            init_emb.detach(), num_clu, args.scl_hcn_ratio, args.device)

    # 训练循环
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()

        emb_dict = model.encode(node_features,
                                adj_orig_norm=adj_orig_norm_for_encoder,
                                adj_rc_norm=adj_rc_norm_for_encoder,
                                H_clique=H_clique_torch)

        # SCL 伪标签更新
        if args.lambda_hscl > 0 and args.scl_update_pseudo_labels_every > 0 and \
                (epoch % args.scl_update_pseudo_labels_every == 0) and 1 < epoch < args.epochs:
            with torch.no_grad():
                Z_hyper = emb_dict.get('hyper_clique')
                Z_orig = emb_dict.get('orig')
                Z_rc = emb_dict.get('rc_aug')
                if Z_orig is not None and Z_hyper is not None:
                    Zs = 0.5 * (Z_orig + Z_rc) if Z_rc is not None else Z_orig
                    Zc = (1 - args.fusion_gamma) * Zs + args.fusion_gamma * Z_hyper
                    num_clu = max(2, int(dataset_obj.num_nodes * args.scl_cluster_k_ratio))
                    if dataset_obj.num_nodes < num_clu:
                        num_clu = max(1, dataset_obj.num_nodes // 2)
                    if num_clu == 1 and dataset_obj.num_nodes > 1:
                        num_clu = 2
                    if dataset_obj.num_nodes >= 1 and num_clu > 0:
                        model.scl_pseudo_labels, model.scl_hcn_indices, \
                            model.scl_lcn_indices, model.scl_centroids = \
                            get_pseudo_labels_and_hcn_lcn(
                                Zc.detach(), num_clu, args.scl_hcn_ratio, args.device)

        total_loss, loss_detail = model.compute_all_losses(
            emb_dict, node_features,
            adj_orig_dense_target=adj_orig_dense_target,
            H_clique=H_clique_torch,
            adj_orig_norm_for_encoder=adj_orig_norm_for_encoder,
            current_epoch=epoch)

        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print(f"Epoch {epoch}: NaN/Inf 损失，终止训练。")
            break

        if total_loss.requires_grad:
            total_loss.backward()
            if args.gradient_clip_value and args.gradient_clip_value > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.gradient_clip_value)
            optimizer.step()

        if epoch % args.eval_every_epochs == 0 or epoch == args.epochs:
            loss_str = f"loss={total_loss.item():.4f}"
            for k, v in loss_detail.items():
                loss_str += f", {k}={v:.4f}"
            print(f"Epoch [{epoch}/{args.epochs}] {loss_str}")

    print("训练完成。")

    # ----------------------------------------------------------------
    # 步骤 4: 链接预测评估
    # ----------------------------------------------------------------
    print("\n===== 步骤 4: 生成嵌入并评估链接预测 =====")
    model.eval()
    with torch.no_grad():
        final_emb_dict = model.encode(node_features,
                                      adj_orig_norm=adj_orig_norm_for_encoder,
                                      adj_rc_norm=adj_rc_norm_for_encoder,
                                      H_clique=H_clique_torch)
        final_emb_tensor = model.get_final_embeddings(final_emb_dict)
        if final_emb_tensor is None:
            print("错误: 无法生成最终嵌入。")
            return {}
        final_emb_np = final_emb_tensor.cpu().numpy()
    final_emb_map = {nid: final_emb_np[idx] for nid, idx in dataset_obj.node_to_idx.items()}
    print(f"嵌入: {len(final_emb_map)} x {final_emb_np.shape[1]}")

    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'result')
    os.makedirs(results_dir, exist_ok=True)

    if dataset_obj.graph_original.number_of_edges() >= 10:
        print("\n--- 链接预测评估 ---")
        all_trial_results = []

        for i in range(args.link_pred_n_trials):
            trial_seed = args.seed + i
            print(f"\n试验 {i+1}/{args.link_pred_n_trials} (seed={trial_seed})...")
            _, (X_train, Y_train), _, (X_test, Y_test) = split_graph_edges(
                dataset_obj.graph_original,
                test_ratio=args.link_pred_test_ratio,
                val_ratio=args.link_pred_val_ratio,
                seed=trial_seed)

            if len(X_train) < 2 or len(X_test) < 2:
                print("  样本不足，跳过。")
                continue

            lp = LinkPredictor(embeddings=final_emb_map, binary_operator_name='hadamard', seed=trial_seed)
            if not lp.train(X_train, Y_train):
                print("  训练失败，跳过。")
                continue

            scores = lp.evaluate(X_test, Y_test, "测试集")
            scores['trial'] = i + 1
            all_trial_results.append(scores)
            print(f"  AUROC={scores['auc_roc']:.4f}, AUPR={scores['auc_pr']:.4f}, F1={scores['f1']:.4f}")

        if all_trial_results:
            df = pd.DataFrame(all_trial_results)
            print("\n--- 平均性能 ---")
            metrics = ['auc_roc', 'auc_pr', 'f1', 'accuracy']
            for m in metrics:
                if m in df.columns:
                    print(f"  {m}: {df[m].mean():.4f} +/- {df[m].std():.4f}")

            output = {}
            for m in metrics:
                if m in df.columns:
                    output[f'mean_{m}'] = df[m].mean()
                    output[f'std_{m}'] = df[m].std()

            csv_path = os.path.join(results_dir, f'dpfunc_{args.dpfunc_namespace}_results.csv')
            df.to_csv(csv_path, index=False)
            print(f"结果已保存: {csv_path}")

            elapsed = time.time() - start_run_time
            print(f"\n总耗时: {elapsed:.1f}s")
            return output

    print("边数不足，跳过链接预测。")
    return {}


if __name__ == "__main__":
    args = parse_args()
    print("===== SHARCNet + DPFunc 训练 =====")
    print(f"命名空间: {args.dpfunc_namespace}")
    print(f"k-NN: k={args.dpfunc_knn_k}, SVD: dim={args.dpfunc_svd_dim}")
    print(f"数据集: {args.dataset_name}")
    print("=" * 40)
    main(args)
