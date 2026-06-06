# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from utils import (
    InfoNCELoss, TrueTSCGCWeightedTCL,
    get_pseudo_labels_and_hcn_lcn, mask_features
)
from sklearn.neighbors import NearestNeighbors


class GCNLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, use_bias: bool = True):
        super(GCNLayer, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if use_bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            stdv = 1. / math.sqrt(self.weight.size(1))  # size(1) is out_features
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x: torch.Tensor, adj_norm: torch.sparse.FloatTensor) -> torch.Tensor:
        support = torch.mm(x, self.weight)
        output = torch.spmm(adj_norm, support)
        if self.bias is not None:
            return output + self.bias
        return output


class HGNNLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, use_bias: bool = True):
        super(HGNNLayer, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_channels, out_channels))
        if use_bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, X: torch.Tensor, H: torch.sparse.FloatTensor) -> torch.Tensor:
        N, M = H.shape[0], H.shape[1]
        if X.shape[0] != N: raise ValueError(f"特征形状 ({X.shape[0]}) 与关联矩阵的节点数 ({N}) 不匹配。")
        if H._nnz() == 0:  # 如果没有超边或没有节点属于超边
            output = X @ self.weight
            if self.bias is not None: output = output + self.bias
            # No activation here, handled by the _build_encoder_stack / FeatureDecoder
            return output

        D_v_inv_sqrt = torch.sparse.sum(H, dim=1).to_dense().clamp(min=1e-8).pow(-0.5).unsqueeze(1)
        D_e_inv = torch.sparse.sum(H, dim=0).to_dense().clamp(min=1e-8).pow(-1.0).unsqueeze(1)  # Now (M,1)

        X_Theta = X @ self.weight
        Y_node_to_hyperedge = D_v_inv_sqrt * X_Theta
        Z_hyperedge_embeds = torch.sparse.mm(H.t(), Y_node_to_hyperedge)
        Z_prime_hyperedge_aggr = D_e_inv * Z_hyperedge_embeds  # Element-wise multiplication due to D_e_inv being (M,1)
        Y_prime_hyperedge_to_node = torch.sparse.mm(H, Z_prime_hyperedge_aggr)
        output = D_v_inv_sqrt * Y_prime_hyperedge_to_node
        if self.bias is not None: output = output + self.bias
        return output


class StructureDecoder(nn.Module):
    def forward(self, Z1: torch.Tensor, Z2: torch.Tensor = None) -> torch.Tensor:
        target_Z = Z2 if Z2 is not None else Z1
        return torch.sigmoid(torch.matmul(Z1, target_Z.t()))


class FeatureDecoder(nn.Module):
    def __init__(self, layers_modulelist: nn.ModuleList):  # Expects a ModuleList
        super(FeatureDecoder, self).__init__()
        self.decoder_layers = layers_modulelist

    def forward(self, Zc: torch.Tensor, adj_or_H=None) -> torch.Tensor:
        X_re = Zc
        # Iterate through ModuleList that contains [Layer, Act, Dropout, Layer, Act, Dropout, ..., LastLayer]
        layer_idx = 0
        while layer_idx < len(self.decoder_layers):
            current_main_layer = self.decoder_layers[layer_idx]

            if isinstance(current_main_layer, (GCNLayer, HGNNLayer)):
                # GCN/HGNN layers require adj_or_H
                if adj_or_H is None:
                    # Allow first layer to be effectively dense if adj_or_H is None and it's a GCN layer
                    # This is a bit of a special case, might need rethinking if a GCN layer must always have adj
                    if not (isinstance(current_main_layer, GCNLayer) and current_main_layer.weight.shape[0] ==
                            X_re.shape[1]):
                        raise ValueError(f"Decoder {type(current_main_layer).__name__} 层需要结构矩阵 (adj_or_H)。")
                X_re = current_main_layer(X_re, adj_or_H)  # Pass adj_or_H
            else:  # If not GCN/HGNN, it must be an activation or dropout already applied
                raise TypeError(f"FeatureDecoder的堆栈中期望GCN/HGNN层，但得到 {type(current_main_layer)}")

            layer_idx += 1
            # Check for Activation
            if layer_idx < len(self.decoder_layers) and isinstance(self.decoder_layers[layer_idx], (nn.ReLU, nn.GELU)):
                X_re = self.decoder_layers[layer_idx](X_re)
                layer_idx += 1
            # Check for Dropout
            if layer_idx < len(self.decoder_layers) and isinstance(self.decoder_layers[layer_idx], nn.Dropout):
                X_re = self.decoder_layers[layer_idx](X_re)
                layer_idx += 1
        return X_re


class PPI2Complex(nn.Module):
    def __init__(self, args, initial_feature_dim: int):
        super(PPI2Complex, self).__init__()
        self.args = args
        self.initial_feature_dim = initial_feature_dim

        gcn_input_dim = args.gcn_hidden_dims[0] if args.gcn_hidden_dims else args.embedding_dim
        hgnn_input_dim = args.hgnn_hidden_dims[0] if args.hgnn_hidden_dims else args.embedding_dim

        self.initial_projection = nn.Linear(initial_feature_dim, gcn_input_dim)
        nn.init.xavier_uniform_(self.initial_projection.weight);
        nn.init.zeros_(self.initial_projection.bias)

        self.initial_projection_hgnn = nn.Linear(initial_feature_dim, hgnn_input_dim) \
            if initial_feature_dim != hgnn_input_dim or gcn_input_dim != hgnn_input_dim \
            else self.initial_projection
        if self.initial_projection_hgnn is not self.initial_projection:
            nn.init.xavier_uniform_(self.initial_projection_hgnn.weight);
            nn.init.zeros_(self.initial_projection_hgnn.bias)

        self.gcn_encoder_orig_layers = self._build_encoder_decoder_stack(GCNLayer, gcn_input_dim, args.gcn_hidden_dims,
                                                                         args.num_gcn_layers, args.embedding_dim,
                                                                         is_decoder=False)
        if args.use_ricci_augmentation:
            self.gcn_encoder_rc_layers = self._build_encoder_decoder_stack(GCNLayer, gcn_input_dim,
                                                                           args.gcn_hidden_dims, args.num_gcn_layers,
                                                                           args.embedding_dim, is_decoder=False)
        else:
            self.gcn_encoder_rc_layers = None  # Explicitly None

        self.hgnn_encoder_clique_layers = self._build_encoder_decoder_stack(HGNNLayer, hgnn_input_dim,
                                                                            args.hgnn_hidden_dims, args.num_hgnn_layers,
                                                                            args.embedding_dim, is_decoder=False)

        self.struct_decoder = StructureDecoder()

        feat_dec_gcn_in_dim = args.embedding_dim  # Zc is input to feature decoders
        feat_dec_gcn_hid_dims = list(
            reversed(args.gcn_hidden_dims[:-1])) if args.num_gcn_layers > 1 and args.gcn_hidden_dims else []
        self.feat_decoder_gcn_stack = self._build_encoder_decoder_stack(GCNLayer, feat_dec_gcn_in_dim,
                                                                        feat_dec_gcn_hid_dims, args.num_gcn_layers,
                                                                        initial_feature_dim, is_decoder=True)

        feat_dec_hgnn_in_dim = args.embedding_dim
        feat_dec_hgnn_hid_dims = list(
            reversed(args.hgnn_hidden_dims[:-1])) if args.num_hgnn_layers > 1 and args.hgnn_hidden_dims else []
        self.feat_decoder_hgnn_stack = self._build_encoder_decoder_stack(HGNNLayer, feat_dec_hgnn_in_dim,
                                                                         feat_dec_hgnn_hid_dims, args.num_hgnn_layers,
                                                                         initial_feature_dim, is_decoder=True)

        self.feat_decoder_gcn = FeatureDecoder(self.feat_decoder_gcn_stack)
        self.feat_decoder_hgnn = FeatureDecoder(self.feat_decoder_hgnn_stack)

        self.info_nce_loss = InfoNCELoss(temperature=args.contrastive_temperature)
        self.true_tscgc_tcl_loss = TrueTSCGCWeightedTCL(temperature=args.contrastive_temperature)

        self.scl_pseudo_labels: torch.Tensor = None
        self.scl_hcn_indices: torch.Tensor = None
        self.scl_lcn_indices: torch.Tensor = None
        self.scl_centroids: torch.Tensor = None

    def _build_encoder_decoder_stack(self, LayerClass, current_input_dim, hidden_dims_config_list, num_total_layers,
                                     final_layer_out_dim, is_decoder=False):
        layers = nn.ModuleList()
        if num_total_layers == 0: return layers

        current_dim = current_input_dim

        for i in range(num_total_layers):
            is_last_layer = (i == num_total_layers - 1)

            if is_last_layer:
                out_dim = final_layer_out_dim
            elif i < len(hidden_dims_config_list):  # Use specified hidden dim
                out_dim = hidden_dims_config_list[i]
            elif hidden_dims_config_list:  # Not enough specified, use last one
                out_dim = hidden_dims_config_list[-1]
            else:  # No hidden dims specified, but not last layer yet (e.g. num_layers=2, hidden_dims=[]) -> intermediate is final_out_dim
                out_dim = final_layer_out_dim

            layers.append(LayerClass(current_dim, out_dim))
            if not is_last_layer:
                layers.append(nn.GELU() if is_decoder else nn.ReLU())  # Decoder intermediate use GELU, Encoder use ReLU
                layers.append(nn.Dropout(self.args.dropout_rate))
            # For decoders, the very last layer (outputting features) should not have activation/dropout here
            # This is handled by `is_last_layer` and FeatureDecoder not adding them after the stack.
            current_dim = out_dim
        return layers

    def _apply_encoder_decoder_stack(self, x_input, adj_or_H, layer_modulelist):
        x = x_input
        layer_idx = 0
        while layer_idx < len(layer_modulelist):  # Iterate through ModuleList: Layer, [Act], [Dropout]
            current_main_layer = layer_modulelist[layer_idx]
            x = current_main_layer(x, adj_or_H)  # GCNLayer/HGNNLayer takes structure
            layer_idx += 1
            if layer_idx < len(layer_modulelist) and isinstance(layer_modulelist[layer_idx], (nn.ReLU, nn.GELU)):
                x = layer_modulelist[layer_idx](x);
                layer_idx += 1
            if layer_idx < len(layer_modulelist) and isinstance(layer_modulelist[layer_idx], nn.Dropout):
                x = layer_modulelist[layer_idx](x);
                layer_idx += 1
        return x

    def encode(self, node_features_esm: torch.Tensor,
               adj_orig_norm: torch.sparse.FloatTensor,
               adj_rc_norm: torch.sparse.FloatTensor = None,
               H_clique: torch.sparse.FloatTensor = None) -> dict:
        embeddings = {}
        proj_feat_gcn = self.initial_projection(node_features_esm)

        embeddings['orig'] = F.normalize(
            self._apply_encoder_decoder_stack(proj_feat_gcn, adj_orig_norm, self.gcn_encoder_orig_layers), p=2, dim=1)

        if self.args.use_ricci_augmentation and adj_rc_norm is not None and self.gcn_encoder_rc_layers is not None:
            masked_features = mask_features(node_features_esm, self.args.feature_mask_rate)
            proj_masked_feat = self.initial_projection(masked_features)  # Use same projection
            embeddings['rc_aug'] = F.normalize(
                self._apply_encoder_decoder_stack(proj_masked_feat, adj_rc_norm, self.gcn_encoder_rc_layers), p=2,
                dim=1)
        else:
            embeddings['rc_aug'] = None

        if H_clique is not None and H_clique._nnz() > 0 and self.hgnn_encoder_clique_layers:
            proj_feat_hgnn = self.initial_projection_hgnn(node_features_esm)
            embeddings['hyper_clique'] = F.normalize(
                self._apply_encoder_decoder_stack(proj_feat_hgnn, H_clique, self.hgnn_encoder_clique_layers), p=2,
                dim=1)
        else:
            embeddings['hyper_clique'] = None
            if H_clique is not None and H_clique.size(0) > 0:  # Placeholder if no hyperedges but nodes exist
                embeddings['hyper_clique'] = torch.zeros(H_clique.size(0), self.args.embedding_dim,
                                                         device=node_features_esm.device)
        return embeddings

    def compute_reconstruction_losses(self, embeddings: dict, node_features_esm: torch.Tensor,
                                      adj_orig_dense_target: torch.Tensor, H_clique: torch.sparse.FloatTensor,
                                      adj_orig_norm_for_dec: torch.sparse.FloatTensor) -> tuple[torch.Tensor, dict]:
        loss_recon_struct = torch.tensor(0.0, device=self.args.device);
        loss_recon_feat = torch.tensor(0.0, device=self.args.device)
        loss_dict = {'recon_struct': 0.0, 'recon_feat': 0.0}

        Z_orig, Z_rc, Z_hyper = embeddings.get('orig'), embeddings.get('rc_aug'), embeddings.get('hyper_clique')

        # Ensure all necessary embeddings are present
        if Z_orig is None: Z_orig = torch.zeros_like(Z_rc if Z_rc is not None else Z_hyper,
                                                     device=self.args.device) if (
                    Z_rc is not None or Z_hyper is not None) else None
        if Z_hyper is None and Z_orig is not None: Z_hyper = torch.zeros_like(Z_orig,
                                                                              device=self.args.device)  # Use Z_orig shape as fallback
        if Z_orig is None or Z_hyper is None: print("警告：计算重构损失所需Z_orig或Z_hyper缺失。"); return torch.tensor(
            0.0, device=self.args.device), loss_dict

        Zs = 0.5 * (Z_orig + Z_rc) if Z_rc is not None else Z_orig

        if self.args.lambda_recon_struct > 0:
            A_hat_orig = self.struct_decoder(Zs)
            loss_str_orig = F.binary_cross_entropy(A_hat_orig, adj_orig_dense_target)

            A_hyper_target_dense = torch.zeros_like(adj_orig_dense_target)  # Default if no H_clique
            if H_clique is not None and H_clique._nnz() > 0:
                try:
                    A_hyper_target_sparse = torch.sparse.mm(H_clique, H_clique.t()).coalesce()
                    A_hyper_target_dense = (A_hyper_target_sparse.to_dense() > 0).float()
                    A_hyper_target_dense.fill_diagonal_(0)
                except Exception as e:
                    print(f"警告: 计算A_hyper_target失败: {e}")

            A_hat_hyper = self.struct_decoder(Z_hyper)
            loss_str_hyper = F.binary_cross_entropy(A_hat_hyper, A_hyper_target_dense)
            loss_recon_struct = loss_str_orig + loss_str_hyper
            loss_dict['recon_struct'] = loss_recon_struct.item()

        if self.args.lambda_recon_feat > 0:
            gamma = self.args.fusion_gamma
            Zc = (1 - gamma) * Zs + gamma * Z_hyper

            X_re_gcn = self.feat_decoder_gcn(Zc, adj_orig_norm_for_dec)  # Pass appropriate adj for GCN decoder
            loss_feat_gcn = F.mse_loss(X_re_gcn, node_features_esm)

            loss_feat_hgnn = torch.tensor(0.0, device=self.args.device)
            if H_clique is not None and H_clique._nnz() > 0 and self.feat_decoder_hgnn.decoder_layers:
                X_re_hgnn = self.feat_decoder_hgnn(Zc, H_clique)
                loss_feat_hgnn = F.mse_loss(X_re_hgnn, node_features_esm)

            loss_recon_feat = loss_feat_gcn + loss_feat_hgnn
            loss_dict['recon_feat'] = loss_recon_feat.item()

        total_recon_loss = self.args.lambda_recon_struct * loss_recon_struct + \
                           self.args.lambda_recon_feat * loss_recon_feat
        return total_recon_loss, loss_dict

    def compute_tcl_loss_full_tscgc(self, Z_orig: torch.Tensor, Z_rc: torch.Tensor,
                                    H_clique: torch.sparse.FloatTensor,
                                    node_features_esm: torch.Tensor) -> torch.Tensor:
        if not (self.args.lambda_tcl > 0 and Z_orig is not None and Z_rc is not None and \
                H_clique is not None and H_clique._nnz() > 0):
            return torch.tensor(0.0, device=self.args.device)

        num_nodes = Z_orig.shape[0]
        adj_hyper_coo = torch.sparse.mm(H_clique, H_clique.t()).coalesce()

        all_anchor_embeds_orig_list, all_positive_sets_rc_list, all_positive_weights_list = [], [], []

        num_anchors_to_sample = min(num_nodes, self.args.contrastive_batch_size)
        if num_anchors_to_sample == 0: return torch.tensor(0.0, device=Z_orig.device)
        sampled_anchor_indices = torch.randperm(num_nodes, device=Z_orig.device)[:num_anchors_to_sample]

        for i_idx_tensor in sampled_anchor_indices:
            i = i_idx_tensor.item()
            anchor_embed_orig = Z_orig[i]

            row_indices, col_indices = adj_hyper_coo.indices()
            mask_i = (row_indices == i)
            neighbor_indices_in_hyper = col_indices[mask_i]
            neighbor_indices_in_hyper = neighbor_indices_in_hyper[neighbor_indices_in_hyper != i]

            if neighbor_indices_in_hyper.numel() == 0: continue

            current_pos_embeds_rc = Z_rc[neighbor_indices_in_hyper]

            anchor_esm_norm = F.normalize(node_features_esm[i].unsqueeze(0), p=2, dim=1)
            neighbor_esm_norm = F.normalize(node_features_esm[neighbor_indices_in_hyper], p=2, dim=1)

            esm_similarities = torch.matmul(anchor_esm_norm, neighbor_esm_norm.t()).squeeze(0)
            weights = (esm_similarities + 1.0) / 2.0  # Map cosine sim to [0,1] as weights

            all_anchor_embeds_orig_list.append(anchor_embed_orig)
            all_positive_sets_rc_list.append(current_pos_embeds_rc)
            all_positive_weights_list.append(weights)

        if not all_anchor_embeds_orig_list: return torch.tensor(0.0, device=Z_orig.device)

        return self.true_tscgc_tcl_loss(  # Changed parameter names in call
            anchor_embeds_view1=torch.stack(all_anchor_embeds_orig_list),
            list_of_pos_embeds_view2_for_each_anchor=all_positive_sets_rc_list,
            list_of_pos_weights_for_each_anchor=all_positive_weights_list,
            negative_pool_view2=Z_rc
        )

    def compute_hscl_loss_full_tscgc(self, Z_to_cluster: torch.Tensor, Z_contrast_view: torch.Tensor,
                                     node_features_esm: torch.Tensor, current_epoch: int) -> torch.Tensor:
        if not (self.args.lambda_hscl > 0 and Z_to_cluster is not None and Z_contrast_view is not None and \
                self.scl_pseudo_labels is not None and \
                self.scl_hcn_indices is not None and self.scl_lcn_indices is not None):  # Check all SCL states
            return torch.tensor(0.0, device=self.args.device)

        scl_anchors_list, scl_positives_list = [], []
        num_nodes = Z_to_cluster.shape[0]

        all_scl_candidate_indices = torch.cat([self.scl_hcn_indices, self.scl_lcn_indices]).unique()
        if all_scl_candidate_indices.numel() == 0: return torch.tensor(0.0, device=self.args.device)

        num_scl_anchors = min(all_scl_candidate_indices.numel(), self.args.contrastive_batch_size)
        sampled_scl_anchor_indices = all_scl_candidate_indices[
            torch.randperm(all_scl_candidate_indices.numel(), device=self.args.device)[:num_scl_anchors]]

        is_hcn_mask = torch.zeros(num_nodes, dtype=torch.bool, device=self.args.device)
        if self.scl_hcn_indices.numel() > 0: is_hcn_mask[self.scl_hcn_indices] = True

        lcn_knn_map = {}
        actual_lcn_anchors_for_knn = self.scl_lcn_indices[torch.isin(self.scl_lcn_indices, sampled_scl_anchor_indices)]
        if actual_lcn_anchors_for_knn.numel() > 0 and self.args.scl_lcn_knn_k > 0:
            knn_source_embeds_cpu = node_features_esm.cpu().numpy()  # KNN on ESM features
            if knn_source_embeds_cpu.shape[0] > self.args.scl_lcn_knn_k:
                knn_finder = NearestNeighbors(
                    n_neighbors=min(self.args.scl_lcn_knn_k + 1, knn_source_embeds_cpu.shape[0]))
                knn_finder.fit(knn_source_embeds_cpu)
                query_lcn_embeds_for_knn_cpu = node_features_esm[actual_lcn_anchors_for_knn].cpu().numpy()
                if query_lcn_embeds_for_knn_cpu.shape[0] > 0:
                    _, knn_indices_for_lcn_anchors = knn_finder.kneighbors(query_lcn_embeds_for_knn_cpu)
                    for i, anchor_lcn_idx_tensor in enumerate(actual_lcn_anchors_for_knn):
                        anchor_lcn_idx = anchor_lcn_idx_tensor.item()
                        lcn_knn_map[anchor_lcn_idx] = [idx for idx in knn_indices_for_lcn_anchors[i, 1:] if
                                                       idx != anchor_lcn_idx]

        for anchor_idx_tensor in sampled_scl_anchor_indices:
            anchor_idx = anchor_idx_tensor.item();
            anchor_embed_curr = Z_to_cluster[anchor_idx]
            if is_hcn_mask[anchor_idx]:
                scl_anchors_list.append(anchor_embed_curr);
                scl_positives_list.append(Z_contrast_view[anchor_idx])
                anchor_pseudo_label = self.scl_pseudo_labels[anchor_idx]
                same_cluster_hcn_indices = self.scl_hcn_indices[
                    self.scl_pseudo_labels[self.scl_hcn_indices] == anchor_pseudo_label]
                same_cluster_hcn_indices = same_cluster_hcn_indices[same_cluster_hcn_indices != anchor_idx]
                for other_hcn_idx_tensor in same_cluster_hcn_indices:  # Limit number of positives per anchor if too many
                    other_hcn_idx = other_hcn_idx_tensor.item()
                    scl_anchors_list.append(anchor_embed_curr);
                    scl_positives_list.append(Z_contrast_view[other_hcn_idx])
                    scl_anchors_list.append(anchor_embed_curr);
                    scl_positives_list.append(Z_to_cluster[other_hcn_idx])
            elif anchor_idx in lcn_knn_map:
                for neighbor_idx in lcn_knn_map[anchor_idx][:self.args.scl_lcn_knn_k]:  # Limit to K
                    scl_anchors_list.append(anchor_embed_curr);
                    scl_positives_list.append(Z_contrast_view[neighbor_idx])

        if not scl_anchors_list: return torch.tensor(0.0, device=self.args.device)
        scl_anchors_tensor = torch.stack(scl_anchors_list);
        scl_positives_tensor = torch.stack(scl_positives_list)
        return self.info_nce_loss(scl_anchors_tensor, scl_positives_tensor, all_candidates_for_neg=Z_contrast_view)

    def compute_all_losses(self, embeddings: dict, node_features_esm: torch.Tensor,
                           adj_orig_dense_target: torch.Tensor, H_clique: torch.sparse.FloatTensor,
                           adj_orig_norm_for_encoder: torch.sparse.FloatTensor, current_epoch: int) -> tuple:
        # (与上一轮代码一致)
        total_loss = torch.tensor(0.0, device=self.args.device);
        all_loss_details = {}
        if self.args.lambda_recon_struct > 0 or self.args.lambda_recon_feat > 0:
            recon_loss, recon_loss_detail = self.compute_reconstruction_losses(embeddings, node_features_esm,
                                                                               adj_orig_dense_target, H_clique,
                                                                               adj_orig_norm_for_encoder)
            total_loss += recon_loss;
            all_loss_details.update(recon_loss_detail)
        Z_hyper, Z_orig, Z_rc = embeddings.get('hyper_clique'), embeddings.get('orig'), embeddings.get('rc_aug')

        # TCL Loss
        loss_tcl = self.compute_tcl_loss_full_tscgc(Z_orig, Z_rc, H_clique, node_features_esm)
        if Z_orig is not None and Z_rc is not None:  # Only add if views are available
            total_loss += self.args.lambda_tcl * loss_tcl;
            all_loss_details['tcl_full'] = loss_tcl.item()
        else:
            all_loss_details['tcl_full'] = 0.0

        # HSCL Loss
        if Z_hyper is not None and Z_orig is not None:
            Zs_scl = 0.5 * (Z_orig + Z_rc) if Z_rc is not None else Z_orig
            Zc_scl = (1 - self.args.fusion_gamma) * Zs_scl + self.args.fusion_gamma * Z_hyper
            loss_hscl = self.compute_hscl_loss_full_tscgc(Zc_scl, Z_orig, node_features_esm, current_epoch)
            total_loss += self.args.lambda_hscl * loss_hscl;
            all_loss_details['hscl_full'] = loss_hscl.item()
        else:
            all_loss_details['hscl_full'] = 0.0

        # Align Loss
        if self.args.lambda_align > 0 and Z_hyper is not None and Z_rc is not None:
            loss_align1 = self.info_nce_loss(Z_hyper, Z_rc, all_candidates_for_neg=Z_rc);
            loss_align2 = self.info_nce_loss(Z_rc, Z_hyper, all_candidates_for_neg=Z_hyper)
            loss_align = (loss_align1 + loss_align2) / 2.0;
            total_loss += self.args.lambda_align * loss_align;
            all_loss_details['align'] = loss_align.item()
        else:
            all_loss_details['align'] = 0.0

        all_loss_details['total_combined_loss'] = total_loss.item();
        return total_loss, all_loss_details

    def get_final_embeddings(self, embeddings: dict) -> torch.Tensor | None:
        # (与上一轮代码一致)
        Z_orig, Z_rc, Z_hyper = embeddings.get('orig'), embeddings.get('rc_aug'), embeddings.get('hyper_clique')
        if Z_orig is None and Z_rc is None and Z_hyper is None: return None
        # Provide fallbacks if some embeddings are None to allow Zs and Zc calculation
        # Fallback to zeros of the same shape and device if a component is None
        # This requires knowing the number of nodes and embedding dimension
        num_nodes = self.args.contrastive_batch_size  # This is not right, need actual num_nodes
        # A better way is to check if the key exists and is not None before using

        # Simplest non-None check:
        available_embeds = [em for em_name, em in embeddings.items() if em is not None]
        if not available_embeds: return None  # All are None

        # Use the first available embedding to determine shape/device for fallbacks if needed
        ref_embed_for_fallback = available_embeds[0]

        if Z_orig is None: Z_orig = torch.zeros_like(ref_embed_for_fallback)
        if Z_hyper is None: Z_hyper = torch.zeros_like(ref_embed_for_fallback)
        # Z_rc can be None if not use_ricci_augmentation

        Zs = 0.5 * (Z_orig + Z_rc) if Z_rc is not None else Z_orig
        if Z_hyper is not None: Zc = (1 - self.args.fusion_gamma) * Zs + self.args.fusion_gamma * Z_hyper; return Zc
        return Zs