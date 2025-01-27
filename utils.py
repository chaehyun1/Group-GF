import numpy as np
import torch

def filter(P, filter_dimension):
    if filter_dimension == 1:
        return P
    elif filter_dimension == 2:
        return 2 * P - P @ P
    elif filter_dimension == 3:
        return P + 0.01 * ((-1) * P @ P @ P + 10 * P @ P + (-29) * P)
    else:
        raise ValueError("Filter dimension is not valid.")


def hit_at_k(gt_mat, results, negative_list, k=10):
    each_group_score = {}
    hit_sum = 0 
    for i in range(gt_mat.shape[0]):
        relevant_items = set(np.where(gt_mat[i, :] > 0)[0]) 
        candidates = list(set(relevant_items).union(set(negative_list[i])))
        candidates = [item for item in candidates if item < results.shape[1]] 
        top_predicted_items = np.argsort(-results[i, candidates])[:k]
        top_predicted_items = [candidates[item] for item in top_predicted_items] 
        
        if len(relevant_items.intersection(top_predicted_items)) > 0:
            hit_sum += 1
            each_group_score[i] = 1 
        else:
            each_group_score[i] = 0
        
    hit_rate = hit_sum / gt_mat.shape[0] 
    return hit_rate

def ndcg_at_k(gt_mat, results, negative_list, k=10):
    each_group_score = {}
    ndcg_sum = 0
    for i in range(gt_mat.shape[0]):
        relevant_items = set(np.where(gt_mat[i, :] > 0)[0])
        candidates = list(set(relevant_items).union(set(negative_list[i])))
        candidates = [item for item in candidates if item < results.shape[1]]  
        top_predicted_items = np.argsort(-results[i, candidates])[:k]
        top_predicted_items = [candidates[item] for item in top_predicted_items]
        dcg = 0
        idcg = 0
        for j in range(min(k, len(top_predicted_items))):
            if top_predicted_items[j] in relevant_items:
                dcg += 1 / np.log2(j + 2)
            if j < len(relevant_items):
                idcg += 1 / np.log2(j + 2)
        
        if idcg > 0:
            ndcg_sum += dcg / idcg
            each_group_score[i] = dcg / idcg 
        else:
            ndcg_sum += 0
            each_group_score[i] = 0
 
    ndcg = ndcg_sum / gt_mat.shape[0]
    return ndcg


def normalize_sparse_adjacency_matrix(adj_matrix, alpha):
    rowsum = torch.sparse.mm(
        adj_matrix, torch.ones((adj_matrix.shape[1], 1), device=adj_matrix.device)
    ).squeeze() 
    rowsum = torch.pow(rowsum, -alpha)  
    colsum = torch.sparse.mm(
        adj_matrix.t(), torch.ones((adj_matrix.shape[0], 1), device=adj_matrix.device)
    ).squeeze() 
    colsum = torch.pow(colsum, alpha - 1)

    rowsum[rowsum == float('inf')] = 0.
    colsum[colsum == float('inf')] = 0.

    indices = (
        torch.arange(0, rowsum.size(0)).unsqueeze(1).repeat(1, 2).to(adj_matrix.device)
    )
    d_mat_rows = torch.sparse_coo_tensor(
        indices.t(), rowsum, torch.Size([rowsum.size(0), rowsum.size(0)])
    ).to(device=adj_matrix.device)
    indices = (
        torch.arange(0, colsum.size(0)).unsqueeze(1).repeat(1, 2).to(adj_matrix.device)
    )
    d_mat_cols = torch.sparse_coo_tensor(
        indices.t(), colsum, torch.Size([colsum.size(0), colsum.size(0)])
    ).to(device=adj_matrix.device)

    # Normalize adjacency matrix
    norm_adj = d_mat_rows.mm(adj_matrix).mm(d_mat_cols) # R_tilde = D_U^-1 * R * D_I^-1
    return norm_adj

