"""Helper functions for loading dataset"""
import scipy.sparse as sp
import numpy as np
from collections import defaultdict
from scipy.sparse import csr_matrix
import torch
import pandas as pd
from math import comb
import pickle
from itertools import combinations

# code reference: https://github.com/FDUDSDE/WWW2023ConsRec.git

def load_rating_file_to_list(filename):
    """Return **List** format user/group-item interactions"""
    rating_list = []
    lines = open(filename, 'r').readlines()

    for line in lines:
        contents = line.split()
        # Each line: user item
        rating_list.append([int(contents[0]), int(contents[1])])
    return rating_list


# def load_rating_file_to_matrix(filename, dataset, num_users=None, num_items=None):
#     """Return **Matrix** format user/group-item interactions"""
#     if num_users is None:
#         num_users, num_items = 0, 0
#         lines = open(filename, 'r').readlines()
#         for line in lines:
#             contents = line.split()
#             u, i = int(contents[0]), int(contents[1])
#             num_users = max(num_users, u)
#             num_items = max(num_items, i)

#     mat = sp.dok_matrix((num_users + 1, num_items + 1), dtype=np.float32)
#     for line in lines:
#         contents = line.split()
#         if len(contents) > 2:
#             u, i, rating = int(contents[0]), int(contents[1]), int(contents[2])
#             if rating > 0:
#                 mat[u, i] = 1.0
#         else:
#             u, i = int(contents[0]), int(contents[1])
#             mat[u, i] = 1.0
#     return mat

def load_rating_file_to_matrix(filename, dataset, num_items):
    """Return **Matrix** format user/group-item interactions with given num_items"""
    num_users = 0
    lines = open(filename, 'r').readlines()
    for line in lines:
        contents = line.split()
        u, i = int(contents[0]), int(contents[1])
        num_users = max(num_users, u)

    mat = sp.dok_matrix((num_users + 1, num_items), dtype=np.float32)
    for line in lines:
        contents = line.split()
        if len(contents) > 2:
            u, i, rating = int(contents[0]), int(contents[1]), int(contents[2])
            if rating > 0:
                mat[u, i] = 1.0
        else:
            u, i = int(contents[0]), int(contents[1])
            mat[u, i] = 1.0
    return mat


def load_negative_file(filename):
    """Return **List** format negative files"""
    negative_list = []

    lines = open(filename, 'r').readlines()

    for line in lines:
        negatives = line.split()[1:]
        negatives = [int(neg_item) for neg_item in negatives]
        negative_list.append(negatives)
    return negative_list


def load_group_member_to_dict(user_in_group_path):
    """Return **Dict** format group-to-member-list mapping"""
    group_member_dict = defaultdict(list)
    lines = open(user_in_group_path, 'r').readlines()

    for line in lines:
        contents = line.split()
        group = int(contents[0])
        for member in contents[1].split(','):
            group_member_dict[group].append(int(member))
    return group_member_dict


def build_group_graph(group_data, num_groups):
    """Return group-level graph (**a weighted graph** with weights defined as ratio of common members and items)"""
    matrix = np.zeros((num_groups, num_groups))

    for i in range(num_groups):
        group_a = set(group_data[i])
        for j in range(i + 1, num_groups):
            group_b = set(group_data[j])
            overlap = group_a & group_b
            union = group_a | group_b
            # weight computation
            matrix[i][j] = float(len(overlap) / len(union))
            matrix[j][i] = matrix[i][j]

    matrix = matrix + np.diag([1.0] * num_groups)
    degree = np.sum(np.array(matrix), 1)
    # \mathbf{D}^{-1} \dot \mathbf{A}
    return np.dot(np.diag(1.0 / degree), matrix)


def build_hyper_graph(group_member_dict, group_train_path, num_users, num_items, num_groups, group_item_dict=None):
    """Return member-level hyper-graph"""
    # Construct group-to-item-list mapping
    if group_item_dict is None:
        group_item_dict = defaultdict(list)

        for line in open(group_train_path, 'r').readlines():
            contents = line.split()
            if len(contents) > 2:
                group, item, rating = int(contents[0]), int(contents[1]), int(contents[2])
                if rating > 0:
                    group_item_dict[group].append(item)
            else:
                group, item = int(contents[0]), int(contents[1])
                group_item_dict[group].append(item)

    def _prepare(group_dict, rows, axis=0):
        nodes, groups = [], []

        for group_id in range(num_groups):
            groups.extend([group_id] * len(group_dict[group_id]))
            nodes.extend(group_dict[group_id])

        hyper_graph = csr_matrix((np.ones(len(nodes)), (nodes, groups)), shape=(rows, num_groups))
        hyper_deg = np.array(hyper_graph.sum(axis=axis)).squeeze()
        hyper_deg[hyper_deg == 0.] = 1
        hyper_deg = sp.diags(1.0 / hyper_deg)
        return hyper_graph, hyper_deg

    # Two separate hypergraphs (user_hypergraph, item_hypergraph for hypergraph convolution computation)
    user_hg, user_hg_deg = _prepare(group_member_dict, num_users)
    item_hg, item_hg_deg = _prepare(group_item_dict, num_items)

    for group_id, items in group_item_dict.items():
        group_item_dict[group_id] = [item + num_users for item in items]
    group_data = [group_member_dict[group_id] + group_item_dict[group_id] for group_id in range(num_groups)]
    full_hg, hg_dg = _prepare(group_data, num_users + num_items, axis=1)

    user_hyper_graph = torch.sparse.mm(convert_sp_mat_to_sp_tensor(user_hg_deg),
                                       convert_sp_mat_to_sp_tensor(user_hg).t())
    item_hyper_graph = torch.sparse.mm(convert_sp_mat_to_sp_tensor(item_hg_deg),
                                       convert_sp_mat_to_sp_tensor(item_hg).t())
    full_hyper_graph = torch.sparse.mm(convert_sp_mat_to_sp_tensor(hg_dg), convert_sp_mat_to_sp_tensor(full_hg))
    print(
        f"User hyper-graph {user_hyper_graph.shape}, Item hyper-graph {item_hyper_graph.shape}, Full hyper-graph {full_hyper_graph.shape}")

    return user_hyper_graph, item_hyper_graph, full_hyper_graph, group_data


def convert_sp_mat_to_sp_tensor(x):
    """Convert `csr_matrix` into `torch.SparseTensor` format"""
    coo = x.tocoo().astype(np.float32)
    row = torch.Tensor(coo.row).long()
    col = torch.Tensor(coo.col).long()
    index = torch.stack([row, col])
    data = torch.FloatTensor(coo.data)
    return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))


def build_light_gcn_graph(group_item_net, num_groups, num_items):
    """Return item-level graph (**a group-item bipartite graph**)"""
    adj_mat = sp.dok_matrix((num_groups + num_items, num_groups + num_items), dtype=np.float32)
    adj_mat = adj_mat.tolil()

    R = group_item_net.tolil()
    adj_mat[:num_groups, num_groups:] = R
    adj_mat[num_groups:, :num_groups] = R.T
    adj_mat = adj_mat.todok()
    # print(adj_mat, adj_mat.shape)

    row_sum = np.array(adj_mat.sum(axis=1))
    d_inv = np.power(row_sum, -0.5).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat = sp.diags(d_inv)
    # print(d_mat)

    norm_adj = d_mat.dot(adj_mat)
    norm_adj = norm_adj.dot(d_mat)
    norm_adj = norm_adj.tocsr()
    graph = convert_sp_mat_to_sp_tensor(norm_adj)
    return graph.coalesce()


def groupMember_relationship(dataset):
    gm = pd.read_csv(f'./data/{dataset}/groupMember.txt', sep=' ', header=None, names=['group', 'user'])
    gm = gm.set_index('group').user.str.split(',', expand=True).stack().reset_index(level=1, drop=True).reset_index(name='user')
    gm = gm.astype(int)
    
    grouped_users = {group: gm[gm['group'] == group]['user'].tolist() for group in gm['group'].unique()}
    
    return grouped_users 

def cosine_similarity(v1, v2):
    v1 = np.array(v1)
    v2 = np.array(v2)
    
    dot_product = np.dot(v1, v2)
    
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    
    if norm_v1 == 0 or norm_v2 == 0:
        similarity = 0
        
    else:
        similarity = dot_product / (norm_v1 * norm_v2)
    
    return similarity

def average_cosine_similarity(user_vector_list):
    similarities = []
    
    for v1, v2 in combinations(user_vector_list, 2):
        similarity = cosine_similarity(v1, v2)
        similarities.append(similarity)
    
    average_similarity = np.mean(similarities)
    
    return average_similarity

def iou_similarity(v1, v2):
    # Convert lists to arrays
    v1 = np.array(v1)
    v2 = np.array(v2)
    
    # Calculate Intersection and Union
    intersection = np.sum(np.minimum(v1, v2))
    union = np.sum(np.maximum(v1, v2))
    
    # Compute IoU similarity
    if union == 0:
        similarity = 0
    else:
        similarity = intersection / union
    
    return similarity

def average_iou_similarity(user_vector_list):
    similarities = []
    
    for v1, v2 in combinations(user_vector_list, 2):
        similarity = iou_similarity(v1, v2)
        similarities.append(similarity)
    
    # Compute average IoU similarity
    average_similarity = np.mean(similarities)
    
    return average_similarity


# Test code
# if __name__ == "__main__":
#     g_m_d = {0: [0, 1, 2], 1: [2, 3], 2: [4, 5, 6]}
#     g_i_d = {0: [0, 1], 1: [1, 2], 2: [3]}
#     user_g, item_g, hg, g_data = build_hyper_graph(g_m_d, "", 7, 4, 3, g_i_d)
#
#     print(user_g)
#     print(item_g)
#     print(hg)
#     print()
#     g = build_group_graph(g_data, 3)
#     print(g)

def gm_similarity(data):
    train_group_result, test_group_result, train_user_result, test_user_result = data.train_g, data.test_g, data.train_u, data.test_u # adj_matrix
    train_user_to_index = {user: index for index, user in enumerate(data.unique_user_info)} # 여기서 user: 인덱스로 표현하기 (unique user)
    train_group_to_index = {group: index for index, group in enumerate(data.unique_group_info)} # 여기서 group: 인덱스로 표현하기 (unique group)
    
    group_members = data.loadGroupMembers() # 그룹 당 그 그룹에 속한 여러 멤버 아이디 모으기 / 딕셔너리 
    group_member_matrix = torch.zeros((data.num_groups + 1, data.num_users + 1)) # 그룹 수 x 유저 수 + 1
    
    for group_id, members in group_members.items(): # 그룹 아이디와 그룹에 속한 멤버 아이디
        if group_id in train_group_to_index: # 그룹 아이디가 트레인 그룹에 있으면
            group_index = train_group_to_index[group_id] # 그룹 아이디에 해당하는 인덱스
            
            # matching values에 해당하는 유저에 대한 인덱스 찾기 
            user_vector_list = {}
            for m in members:
                if m in train_user_to_index:
                    member_idx = train_user_to_index[m]
                    user_vector_list[m] = train_user_result[member_idx]
            
            similarity_sum = {i: 0 for i in members}
            if len(members) > 1:
                for i, j in combinations(members, 2):
                    if i in user_vector_list.keys() and j in user_vector_list.keys():
                        sim = cosine_similarity(user_vector_list[i], user_vector_list[j])
                        similarity_sum[i] += sim
                        similarity_sum[j] += sim
            else:
                similarity_sum[members[0]] = 1 # 그룹에 속한 멤버가 1명이면 1
            
            for member_id in members: # 멤버 아이디
                if member_id <= data.num_users: # 멤버 아이디가 유저 수보다 작거나 같으면
                    group_member_matrix[group_index, member_id] = similarity_sum[member_id] 
                    
    # 0~1 사이로 정규화: min-max 스케일링
    min_val = torch.min(group_member_matrix)
    max_val = torch.max(group_member_matrix)
    scaled_matrix = (group_member_matrix - min_val) / (max_val - min_val)
    
    return scaled_matrix


def member_relationship(data):
    train_group_result, test_group_result, train_user_result, test_user_result = data.train_g, data.test_g, data.train_u, data.test_u # adj_matrix
    train_user_to_index = {user: index for index, user in enumerate(data.unique_user_info)} # 여기서 user: 인덱스로 표현하기 (unique user)
    train_group_to_index = {group: index for index, group in enumerate(data.unique_group_info)} # 여기서 group: 인덱스로 표현하기 (unique group)
    
    group_members = data.loadGroupMembers() # 그룹 당 그 그룹에 속한 여러 멤버 아이디 모으기 / 딕셔너리 
    group_member_matrix = torch.zeros((data.num_groups + 1, data.num_users + 1)) # 그룹 수 x 유저 수 + 1
    
    for group_id, members in group_members.items(): # 그룹 아이디와 그룹에 속한 멤버 아이디
        if group_id in train_group_to_index: # 그룹 아이디가 트레인 그룹에 있으면
            group_index = train_group_to_index[group_id] # 그룹 아이디에 해당하는 인덱스
            
            # matching values에 해당하는 유저에 대한 인덱스 찾기 
            user_vector_list = {}
            for m in members:
                if m in train_user_to_index:
                    member_idx = train_user_to_index[m]
                    user_vector_list[m] = train_user_result[member_idx]
                    
            relationship_sum = {i: 0 for i in members}
            if len(members) > 1:
                for i, j in combinations(members, 2):
                    if i in user_vector_list.keys() and j in user_vector_list.keys():
                        intersection = sum([1 for x, y in zip(user_vector_list[i], user_vector_list[j]) if x == 1 and y == 1])
                        if torch.sum(user_vector_list[i]) == 0 or torch.sum(user_vector_list[j]) == 0:
                            relationship_sum[i] += 0
                            relationship_sum[j] += 0
                        else:
                            relationship_sum[i] += intersection / torch.sum(user_vector_list[i])
                            relationship_sum[j] += intersection / torch.sum(user_vector_list[j])
            else:
                relationship_sum[members[0]] = 0
                
            for member_id in members: # 멤버 아이디
                if member_id <= data.num_users: # 멤버 아이디가 유저 수보다 작거나 같으면
                    group_member_matrix[group_index, member_id] = relationship_sum[member_id] 
    
    # 0~1 사이로 정규화: min-max 스케일링
    min_val = torch.min(group_member_matrix)
    max_val = torch.max(group_member_matrix)
    scaled_matrix = (group_member_matrix - min_val) / (max_val - min_val)
    
    return scaled_matrix

def gm_relationship(data, dataset):
    with open(f'./{dataset}/train_g.pkl', 'rb') as f:
        train_group_result = pickle.load(f)

    with open(f'./{dataset}/train_u.pkl', 'rb') as f:
        train_user_result = pickle.load(f)

    with open(f'./{dataset}/test_g.pkl', 'rb') as f:
        test_group_result = pickle.load(f)

    with open(f'./{dataset}/test_u.pkl', 'rb') as f:
        test_user_result = pickle.load(f)

    with open(f'./{dataset}/unique_user_info.pkl', 'rb') as f:
        train_user_to_index = pickle.load(f)

    with open(f'./{dataset}/unique_group_info.pkl', 'rb') as f:
        train_group_to_index = pickle.load(f)
        
    with open(f'./{dataset}/train_user_item_to_index.pkl', 'rb') as f:
        train_user_item_to_index = pickle.load(f)
        
    with open(f'./{dataset}/train_group_item_to_index.pkl', 'rb') as f:
        train_group_item_to_index = pickle.load(f)
    
    group_members = data.loadGroupMembers() # 그룹 당 그 그룹에 속한 여러 멤버 아이디 모으기 / 딕셔너리 
    group_member_matrix = torch.zeros((data.num_groups + 1, data.num_users + 1)) # 그룹 수 x 유저 수 + 1
    
    for group_id, members in group_members.items(): # 그룹 아이디와 그룹에 속한 멤버 아이디
        if group_id in train_group_to_index: # 그룹 아이디가 트레인 그룹에 있으면
            group_idx = train_group_to_index[group_id] # 그룹 아이디에 해당하는 인덱스
            group_item_vector = train_group_result[group_idx] # 그룹 아이디에 해당하는 rating 벡터
            group_sort_dict = sorting_dict(group_item_vector)
            
            # matching values에 해당하는 유저에 대한 인덱스 찾기 
            # user_vector_list = {} # 그룹에 속한 멤버들의 아이디와 그에 해당하는 인덱스
            for m in members:
                if m in train_user_to_index:
                    member_idx = train_user_to_index[m]
                    # user_vector_list[m] = train_user_result[member_idx]
                    user_item_vector = train_user_result[member_idx]
                    user_sort_dict = sorting_dict(user_item_vector)
                    
                    # 그룹과 유저의 관계성 계산: 스피어만 상관계수
                    if len(user_sort_dict) == 1 or len(group_sort_dict) == 1:
                        relationship = 0
                    else: 
                        relationship = kendall_tau_correlation(user_sort_dict, group_sort_dict)
                    group_member_matrix[group_idx, member_idx] = relationship
    
    return group_member_matrix


def spearman_correlation(user_sort_dict, group_sort_dict):  
    # 공통된 키 찾기
    common_keys = set(user_sort_dict.keys()).intersection(group_sort_dict.keys())
    
    # 공통된 키가 없으면 예외 처리
    if not common_keys:
        raise ValueError("No common keys found between the dictionaries.")
    
    # 공통된 키에 해당하는 값을 추출하여 리스트로 변환
    user_ranks = [user_sort_dict[key] for key in common_keys]
    group_ranks = [group_sort_dict[key] for key in common_keys]
    
    # 공통된 키의 개수
    n = len(common_keys)
    
    if n <= 1: # 논의 필요
        spearman_corr = 0
    else:
        # 순위 차이의 제곱을 계산
        sum_d_squared = sum((user_ranks[i] - group_ranks[i]) ** 2 for i in range(n))
        
        # 스피어만 상관 계수 계산
        spearman_corr = 1 - (6 * sum_d_squared) / (n * (n**2 - 1))
        spearman_corr = max(min(spearman_corr, 1), -1) # 값 왜곡 정규화 
    
    return spearman_corr
    

def sorting_dict(item_vector):
    # 값이 0이 아닌 항목들의 인덱스와 값을 필터링
    filtered_items = [(idx, val) for idx, val in enumerate(item_vector) if val != 0]
    
    # 요소가 하나라면 순위는 무조건 1로 설정
    if len(filtered_items) == 1:
        ranking = {filtered_items[0][0]: 1} 
    else:
        # 내림차순으로 정렬된 인덱스를 구함
        sorted_indices = sorted(filtered_items, key=lambda x: x[1], reverse=True) # [(idx, val), ...]

        ranking = {}
        current_rank = 1
        i = 0
        
        while i < len(sorted_indices):
            # 동일한 값의 항목들을 찾음
            same_val_items = [sorted_indices[i]]
            while i + 1 < len(sorted_indices) and sorted_indices[i][1] == sorted_indices[i + 1][1]:
                same_val_items.append(sorted_indices[i + 1])
                i += 1

            # 평균 순위를 계산
            avg_rank = current_rank + (len(same_val_items) - 1) / 2.0

            # 해당 항목들의 인덱스에 평균 순위를 할당
            for idx, val in same_val_items:
                ranking[idx] = avg_rank

            # 다음 순위로 이동
            current_rank += len(same_val_items)
            i += 1
    
    return ranking # raking = {item_id: rank}

def kendall_tau_correlation(user_sort_dict, group_sort_dict):
    # 공통된 키 찾기
    common_keys = set(user_sort_dict.keys()).intersection(group_sort_dict.keys())
    
    # 공통된 키가 없으면 예외 처리
    if not common_keys:
        return 0
        # raise ValueError("No common keys found between the dictionaries.")
    
    # 공통된 키에 해당하는 값을 추출하여 리스트로 변환
    user_ranks = [user_sort_dict[key] for key in common_keys]
    group_ranks = [group_sort_dict[key] for key in common_keys]
    
    # 공통된 키의 개수
    n = len(common_keys)
    
    if n <= 1:
        return 0
    
    # 순위 간 쌍을 비교하여 일치 여부를 확인
    concordant_pairs = 0
    discordant_pairs = 0
    
    for i in range(n):
        for j in range(i + 1, n):
            sign_user = user_ranks[i] - user_ranks[j]
            sign_group = group_ranks[i] - group_ranks[j]
            
            if sign_user * sign_group > 0:
                concordant_pairs += 1
            elif sign_user * sign_group < 0:
                discordant_pairs += 1
    
    tau = (concordant_pairs - discordant_pairs) / comb(n, 2)
    
    return tau

def multi_hop(augmented_matrix_norm):
    P_tilde = augmented_matrix_norm.T @ augmented_matrix_norm
    alpha_1 = 0.5
    alpha_2 = 0.3
    one_hop = alpha_1*P_tilde
    two_hop = alpha_2*(P_tilde @ P_tilde)
    three_hop = (1-alpha_1-alpha_2)*(P_tilde @ P_tilde @ P_tilde)
    
    return one_hop + two_hop + three_hop