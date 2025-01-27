import scipy.sparse as sp
import numpy as np
from collections import defaultdict
import torch
import pandas as pd

# code reference: https://github.com/FDUDSDE/WWW2023ConsRec.git

def get_max_item_id(train_file, test_file, group_train_file, group_test_file):
    """Get the maximum item ID from both user and group rating files"""
    max_item_id = 0
    for file in [train_file, test_file, group_train_file, group_test_file]:
        with open(file, 'r') as f:
            for line in f:
                contents = line.split()
                item_id = int(contents[1])
                max_item_id = max(max_item_id, item_id)
    return max_item_id + 1

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

def load_rating_file_to_list(filename):
    """Return **List** format user/group-item interactions"""
    rating_list = []
    lines = open(filename, 'r').readlines()

    for line in lines:
        contents = line.split()
        rating_list.append([int(contents[0]), int(contents[1])])
    return rating_list

def load_negative(filename):
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

def convert_sp_mat_to_sp_tensor(x):
    """Convert `csr_matrix` into `torch.SparseTensor` format"""
    coo = x.tocoo().astype(np.float32)
    row = torch.Tensor(coo.row).long()
    col = torch.Tensor(coo.col).long()
    index = torch.stack([row, col])
    data = torch.FloatTensor(coo.data)
    return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))

def datasetToDataFrame(file_train, user=True):
    data = []
    with open(file_train, 'r') as file:
        for line in file:
            line = line.strip().split()
            if len(line) == 2:
                line.append(1) 
            data.append([int(x) for x in line])

    if user:
        df = pd.DataFrame(data, columns=['user_id', 'item_id', 'rating'])
    else:
        df = pd.DataFrame(data, columns=['group_id', 'item_id', 'rating'])
    return df

def loadGroupMembers(group_member_file):
    group_members = {}
    with open(group_member_file, 'r') as file:
        for line in file:
            line = line.strip().split()
            group_id = int(line[0])
            member_ids = [int(x) for x in line[1].split(',')]
            group_members[group_id] = member_ids
    return group_members

def load_negative_file(filename, item_to_index):
    """Return **List** format negative files with correct indices, sorted by user ID"""
    negative_dict = {}

    with open(filename, 'r') as file:
        lines = file.readlines()

        for line in lines:
            line = line.strip().split()
            user_item = line[0].strip('()').split(',')
            user_id = int(user_item[0])

            # Get the list of negative items and convert them to correct indices
            negatives = line[1:]
            negatives = [item_to_index[int(neg_item)] for neg_item in negatives if int(neg_item) in item_to_index]
            negatives.sort()  # Ensure the negatives are sorted by index

            negative_dict[user_id] = negatives

    # Sort the dictionary by user_id and create a list
    negative_list = [negatives for user_id, negatives in sorted(negative_dict.items())]

    return negative_list

def calculate_similarity(group_tensor, user_tensor):
    group_norm = torch.norm(group_tensor, p=2, dim=1, keepdim=True)
    user_norm = torch.norm(user_tensor, p=2, dim=1, keepdim=True)
    similarity = torch.mm(group_tensor, user_tensor.t()) / (group_norm * user_norm.t())
    similarity[torch.isnan(similarity)] = 0
    return similarity