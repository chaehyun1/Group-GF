import torch
from utils import *
import torch

class GroupGF:
    def __init__(self, alpha, beta, power, 
                 user_filter, group_filter, uni_filter, 
                 top_k, device):
        self.alpha = alpha
        self.beta = beta
        self.power = power
        self.user_filter = user_filter
        self.group_filter = group_filter
        self.uni_filter = uni_filter
        self.top_k = top_k
        self.device = device
    
    def run_model(self, R_tr_g, R_ts_g, R_tr_u, 
                  g_neg, gu_mat, 
                  train_group_n_items, train_user_n_items):
        
        # Compute unified matrix
        R_uni = torch.cat((R_tr_g, R_tr_u), dim=0) 
        R_uni_norm = normalize_sparse_adjacency_matrix(R_uni, 0.5)
        P_uni = R_uni_norm.T @ R_uni_norm

        # Convert to dense matrices
        new_R_tr_g = R_tr_g.to_dense()  # (group x item)
        new_R_tr_u =  R_tr_u.to_dense()  # (user x item)

        # Create augmented matrices
        augmented_user_matrix = torch.cat((new_R_tr_u, gu_mat.T), dim=1)  # (user x (item + group))
        augmented_group_matrix = torch.cat((new_R_tr_g, gu_mat), dim=1)  # (group x (item + user))

        # Normalize augmented matrices
        augmented_user_matrix_norm =  normalize_sparse_adjacency_matrix(augmented_user_matrix, 0.5)
        augmented_group_matrix_norm = normalize_sparse_adjacency_matrix(augmented_group_matrix, 0.5)

        # Compute P_tilde_dagger
        augmented_user_P = augmented_user_matrix_norm.T @ augmented_user_matrix_norm
        augmented_user_P = augmented_user_P[:train_user_n_items, :train_user_n_items]
        augmented_group_P = augmented_group_matrix_norm.T @ augmented_group_matrix_norm
        augmented_group_P = augmented_group_P[:train_group_n_items, :train_group_n_items]
        
        # Cleanup to save memory
        del augmented_user_matrix, augmented_group_matrix 

        # Compute P_bar
        augmented_user_P.data **= self.power
        augmented_group_P.data **= self.power
        P_uni.data **= self.power

        # Apply filters
        filter_P_user = filter(augmented_user_P, self.user_filter)
        filter_P_group = filter(augmented_group_P, self.group_filter)
        filter_P_uni = filter(P_uni, self.uni_filter)
        
        # Combine matrices
        new_P = (1 - self.alpha - self.beta) * filter_P_user + self.alpha * filter_P_group + self.beta * filter_P_uni 

        # Move tensors to device
        augmented_user_P = augmented_user_P.to(device=self.device).float()
        augmented_group_P = augmented_group_P.to(device=self.device).float()
        new_R_tr_g = new_R_tr_g.to(device=self.device).float()
        new_R_tr_u = new_R_tr_u.to(device=self.device).float()
        new_P = new_P.to(device=self.device).float()
        
        # Compute results
        new_results = new_R_tr_g @ new_P

        # Cleanup to save memory
        del new_R_tr_g, new_R_tr_u 
        
        # Compute evaluation metrics
        inf_m = -99999
        new_group_gt_mat = R_ts_g.to_dense()
        new_results = new_results.cpu() + (inf_m) * R_tr_g.to_dense()
        new_group_gt_mat = new_group_gt_mat.cpu().detach().numpy()
        new_results = new_results.cpu().detach().numpy()
        
        hit_result = hit_at_k(new_group_gt_mat, new_results, g_neg, k=self.top_k)
        ndcn_result = ndcg_at_k(new_group_gt_mat, new_results, g_neg, k=self.top_k)
        
        return hit_result, ndcn_result