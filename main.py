import torch
import scipy.sparse as sp
import os
import argparse
from utils import *
from dataset import Dataset
import torch
import time 

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
current_directory = os.getcwd()

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset",
    type=str,    default="douban", # "CAMRa2011" or "Mafengwo"
    help="Either CAMRa2011 or Mafengwo",
)

parser.add_argument(
    "--verbose",
    type=int,
    default=1,
    help="Whether to print the results or not. 1 prints the results, 0 does not.",
)

parser.add_argument("--alpha", type=float, default=0.1) # 2, 0.9 mafengwo
parser.add_argument("--beta", type=float, default=0.1)
parser.add_argument("--power", type=float, default=1, help="For normalization of P")
parser.add_argument("--user_filter", type=int, default=1, help="filter of user")
parser.add_argument("--group_filter", type=int, default=2, help="filter of group")
parser.add_argument("--uni_filter", type=int, default=2, help="filter of unified")
# parser.add_argument("--filter_pair", type=str, default="filter_2D_3D", help="pair filter of user and group")

args = parser.parse_args()
if args.verbose:
    print(f"Device: {device}")

# load dataset
dataset = args.dataset
path = current_directory + f'/data/{dataset}/'
data = Dataset(path)
R_tr_g, R_ts_g, R_tr_u, R_ts_u, C, g_neg, u_neg, gu_mat = data.getDataset()

# shape
train_n_groups = R_tr_g.shape[0]
train_group_n_items = R_tr_g.shape[1]
train_n_users = R_tr_u.shape[0]
train_user_n_items = R_tr_u.shape[1]

if args.verbose:
    print(f"number of tr_groups: {train_n_groups}")
    print(f"number of tr_groups_items: {train_group_n_items}")
    print(f"number of tr_users: {train_n_users}")
    print(f"number of tr_users_items: {train_user_n_items}")
    # print(R_tr_g.shape, R_ts_g.shape, R_tr_u.shape, R_ts_u.shape, C.shape, gu_mat.shape)

# runtime 시작
start = time.time()

# R unified 
R_uni = torch.cat((R_tr_g, R_tr_u), dim=0) 
R_uni_norm = normalize_sparse_adjacency_matrix(R_uni, 0.5)  # P_Uni, to further consider relation between group and user. (augmentation may not enough)
P_uni = R_uni_norm.T @ R_uni_norm

new_R_tr_g = R_tr_g.to_dense()  # (group x item)
new_R_tr_u =  R_tr_u.to_dense()  # (user x item)

# Augmented matrices
# print(new_R_tr_g.shape, gu_mat.shape)
augmented_user_matrix = torch.cat((new_R_tr_u, gu_mat.T), dim=1)  # (user x (item + group))
augmented_group_matrix = torch.cat((new_R_tr_g, gu_mat), dim=1)  # (group x (item + user))


# Normalize the augmented matrices
augmented_user_matrix_norm =  normalize_sparse_adjacency_matrix(augmented_user_matrix, 0.5)
augmented_group_matrix_norm = normalize_sparse_adjacency_matrix(augmented_group_matrix, 0.5)

# P_tilde = R^T @ R
augmented_user_P = augmented_user_matrix_norm.T @ augmented_user_matrix_norm
augmented_user_P = augmented_user_P[:train_user_n_items, :train_user_n_items]
augmented_group_P = augmented_group_matrix_norm.T @ augmented_group_matrix_norm
augmented_group_P = augmented_group_P[:train_user_n_items, :train_user_n_items]
del augmented_user_matrix, augmented_group_matrix  # 불필요해진 변수 해제

augmented_user_P.data **= args.power
augmented_group_P.data **= args.power

P_uni.data **= args.power

# 여기서 안됨 
filter_P_user = filter(augmented_user_P, args.user_filter)
filter_P_group = filter(augmented_group_P, args.group_filter)
filter_P_uni = filter(P_uni, args.uni_filter)
new_P = (1-args.alpha-args.beta) * filter_P_user + args.beta * filter_P_uni + args.alpha * filter_P_group  

# new_P =  filter(augmented_user_P, augmented_group_P, args.alpha, args.filter_pair) # 원래 코드 
# new_P = filter_ablation_no_user(augmented_user_P, augmented_group_P, args.alpha, args.filter_pair) # no user info
# new_P = filter_ablation_no_group(augmented_user_P, augmented_group_P, args.alpha, args.filter_pair) # no group info

#206sec
#7291
#-g 7272
#-uni 0.6665
#-m 0.6104
#-a 0.6728

augmented_user_P = augmented_user_P.to(device=device).float()
augmented_group_P = augmented_group_P.to(device=device).float()
new_R_tr_g = new_R_tr_g.to(device=device).float()
new_R_tr_u = new_R_tr_u.to(device=device).float()
new_P = new_P.to(device=device).float()
# P_uni = P_uni.to(device=device).float()

# train_user_results = new_R_tr_u @ augmented_user_P
# train_group_results = new_R_tr_g @ augmented_group_P


# P_uni.data**=: 0.88 , 3rd filter (0.1 0.3) -> Camra 5027 -> 5037 NDCG
# new_results = new_R_tr_g @ (new_P + 0.1*(P_uni + 0.3*((-1)*P_uni@P_uni@P_uni + 10*P_uni@P_uni+ (-29)*P_uni))) #-> Camra
# new_results = new_R_tr_g @ (new_P + 0.35*(P_uni))  # mafengwu 8373 -> 8412 NDCG
new_results = new_R_tr_g @ new_P


print(f"성능 측정 전 Time: {time.time() - start:.4f}s")

del new_R_tr_g, new_R_tr_u  # 더 이상 사용하지 않는 변수는 해제
# Now get the results
inf_m = -99999
new_group_gt_mat = R_ts_g.to_dense()
new_results = new_results.cpu() + (inf_m) * R_tr_g.to_dense()
new_group_gt_mat = new_group_gt_mat.cpu().detach().numpy()
new_results = new_results.cpu().detach().numpy()

print(f"NEW MODEL Hit@K: {hit_at_k(new_group_gt_mat, new_results, g_neg, k=10, dataset=dataset, groupSize=True):.4f}")
print(f"NEW MODEL NDCG@K: {ndcg_at_k(new_group_gt_mat, new_results, g_neg, k=10, dataset=dataset, groupSize=True):.4f}")

print(f"성능 측정 후 Time: {time.time() - start:.4f}s")