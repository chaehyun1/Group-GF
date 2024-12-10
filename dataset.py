import pandas as pd
import scipy.sparse as sp
import numpy as np
import torch
import os
from datautil import load_rating_file_to_matrix, load_rating_file_to_list, load_negative_file, \
    load_group_member_to_dict, build_hyper_graph, build_group_graph, build_light_gcn_graph
import pickle

class Dataset:
    def __init__(self, path):
        self.path = path
        self.group_train_file = path + '/groupRatingTrain.txt'
        self.group_test_file = path + '/groupRatingTest.txt'
        self.train_file = path + '/userRatingTrain.txt'
        self.test_file = path + '/userRatingTest.txt'
        self.group_member_file = path + '/groupMember.txt'
        self.group_negative_file = path + '/groupRatingNegative.txt'
        self.user_negative_file = path + '/userRatingNegative.txt'
        
        # Load matrices with consistent number of items
        self.num_items = self.get_max_item_id()
        self.user_train_matrix = load_rating_file_to_matrix(self.train_file, path, self.num_items)
        self.user_test_matrix = load_rating_file_to_matrix(self.test_file, path, self.num_items)
        self.user_test_ratings = load_rating_file_to_list(self.test_file)
        self.user_test_negatives = load_negative_file(self.user_negative_file)
        self.num_users = self.user_train_matrix.shape[0] - 1
        
        # print(f"UserItem: {self.user_train_matrix.shape} with {len(self.user_train_matrix.keys())} "
        #       f"interactions, sparsity: {(1-len(self.user_train_matrix.keys()) / self.num_users / self.num_items):.5f}")
    
        # Group data
        self.group_train_matrix = load_rating_file_to_matrix(self.group_train_file, path, self.num_items)
        self.group_test_matrix = load_rating_file_to_matrix(self.group_test_file, path, self.num_items)
        self.group_test_ratings = load_rating_file_to_list(self.group_test_file)
        self.group_test_negatives = load_negative_file(self.group_negative_file)
        self.num_groups = self.group_train_matrix.shape[0] - 1
        self.group_member_dict = load_group_member_to_dict(self.group_member_file)

        # print(f"GroupItem: {self.group_train_matrix.shape} with {len(self.group_train_matrix.keys())} interactions, "
        #       f"sparsity: {(1-len(self.group_train_matrix.keys()) / self.num_groups / self.num_items):.5f}")
        
        # 추가한 부분
        self.train_g = None
        self.train_u = None
        self.test_g = None
        self.test_u = None
        self.unique_user_info = None
        self.unique_group_info = None 

    def get_max_item_id(self):
        """Get the maximum item ID from both user and group rating files"""
        max_item_id = 0
        for file in [self.train_file, self.test_file, self.group_train_file, self.group_test_file]:
            with open(file, 'r') as f:
                for line in f:
                    contents = line.split()
                    item_id = int(contents[1])
                    max_item_id = max(max_item_id, item_id)
        return max_item_id + 1

    def datasetToDataFrame(self, file_train, user=True):
        data = []
        with open(file_train, 'r') as file:
            for line in file:
                line = line.strip().split()
                if len(line) == 2:
                    line.append(1)  # rating 값이 없을 때 1로 채움
                data.append([int(x) for x in line])

        if user:
            df = pd.DataFrame(data, columns=['user_id', 'item_id', 'rating'])
        else:
            df = pd.DataFrame(data, columns=['group_id', 'item_id', 'rating'])
        return df

    def loadGroupMembers(self):
        group_members = {}
        with open(self.group_member_file, 'r') as file:
            for line in file:
                line = line.strip().split()
                group_id = int(line[0])
                member_ids = [int(x) for x in line[1].split(',')]
                group_members[group_id] = member_ids
        return group_members
    
    def load_negative_file(self, filename, item_to_index):
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

    def scale_sparse_tensor(self, tensor):
        max_value = tensor._values().max().item()
        min_value = tensor._values().min().item()
        scaled_values = (tensor._values() - min_value) / (max_value - min_value)
        scaled_tensor = torch.sparse_coo_tensor(tensor._indices(), scaled_values, tensor.size())
        return scaled_tensor

    def calculate_similarity(self, group_tensor, user_tensor):
        group_norm = torch.norm(group_tensor, p=2, dim=1, keepdim=True)
        user_norm = torch.norm(user_tensor, p=2, dim=1, keepdim=True)
        similarity = torch.mm(group_tensor, user_tensor.t()) / (group_norm * user_norm.t())
        similarity[torch.isnan(similarity)] = 0
        return similarity

    def getDataset(self):
        df_group_train = self.datasetToDataFrame(self.group_train_file, False) # 46948 355913
        df_user_train = self.datasetToDataFrame(self.train_file)

        train_groups = df_group_train['group_id'].unique() # 15913
        train_users = df_user_train['user_id'].unique() # 6050
        train_group_item = df_group_train['item_id'].unique() 
        train_user_item = df_user_train['item_id'].unique()
        
        max_item_id = max(df_group_train['item_id'].max(), df_user_train['item_id'].max())

        item_to_index = {item: idx for idx, item in enumerate(range(max_item_id + 1))}

        train_group_to_index = {group: index for index, group in enumerate(train_groups)}  
        train_user_to_index = {user: index for index, user in enumerate(train_users)}
        train_group_item_to_index = {item: idx for idx, item in enumerate(train_group_item)}
        train_user_item_to_index = {item: idx for idx, item in enumerate(train_user_item)}
        
        # Update matrix with the same item index
        self.group_train_matrix = self.group_train_matrix[:, :max_item_id + 1]
        self.group_test_matrix = self.group_test_matrix[:, :max_item_id + 1]
        self.user_train_matrix = self.user_train_matrix[:, :max_item_id + 1]
        self.user_test_matrix = self.user_test_matrix[:, :max_item_id + 1]
        
        train_group_matrix = self.group_train_matrix.tocsr() 
        test_group_matrix = self.group_test_matrix.tocsr()
        train_user_matrix = self.user_train_matrix.tocsr()
        test_user_matrix = self.user_test_matrix.tocsr()

        train_group_coo_matrix = train_group_matrix.tocoo()
        test_group_coo_matrix = test_group_matrix.tocoo()
        train_user_coo_matrix = train_user_matrix.tocoo()
        test_user_coo_matrix = test_user_matrix.tocoo()

        train_group_data = torch.FloatTensor(train_group_coo_matrix.data)
        test_group_data = torch.FloatTensor(test_group_coo_matrix.data)
        train_user_data = torch.FloatTensor(train_user_coo_matrix.data)
        test_user_data = torch.FloatTensor(test_user_coo_matrix.data)

        train_group_indices = torch.LongTensor(np.vstack((train_group_coo_matrix.row, train_group_coo_matrix.col)))
        test_group_indices = torch.LongTensor(np.vstack((test_group_coo_matrix.row, test_group_coo_matrix.col)))
        train_user_indices = torch.LongTensor(np.vstack((train_user_coo_matrix.row, train_user_coo_matrix.col)))
        test_user_indices = torch.LongTensor(np.vstack((test_user_coo_matrix.row, test_user_coo_matrix.col)))

        train_group_result = torch.sparse_coo_tensor(train_group_indices, train_group_data, torch.Size(train_group_coo_matrix.shape)).to_dense()
        test_group_result = torch.sparse_coo_tensor(test_group_indices, test_group_data, torch.Size(test_group_coo_matrix.shape)).to_dense()
        train_user_result = torch.sparse_coo_tensor(train_user_indices, train_user_data, torch.Size(train_user_coo_matrix.shape)).to_dense()
        test_user_result = torch.sparse_coo_tensor(test_user_indices, test_user_data, torch.Size(test_user_coo_matrix.shape)).to_dense()
        
        # 추가한 부분 
        self.train_g = train_group_result
        self.train_u = train_user_result
        self.test_g = test_group_result
        self.test_u = test_user_result
        self.unique_user_info = train_user_to_index
        self.unique_group_info = train_group_to_index
        self.unique_user_item_info = train_user_item_to_index
        self.unique_group_item_info = train_group_item_to_index
        
        group_members = self.loadGroupMembers() # 그룹 당 그 그룹에 속한 여러 멤버 아이디 모으기 / 딕셔너리 
        
        num_groups = len(train_groups) 
        group_member_matrix = torch.zeros((num_groups, self.num_users + 1)) # 그룹 수 x 유저 수 + 1
        
        for group_id, members in group_members.items(): # 그룹 아이디와 그룹에 속한 멤버 아이디
            if group_id in train_group_to_index: # 그룹 아이디가 트레인 그룹에 있으면
                group_index = train_group_to_index[group_id] # 그룹 아이디에 해당하는 인덱스
                for member_id in members: # 멤버 아이디
                    if member_id <= self.num_users: # 멤버 아이디가 유저 수보다 작거나 같으면
                        group_member_matrix[group_index, member_id] = 1 # 그룹 인덱스, 멤버 아이디 = 1


        num_users = train_user_result.shape[0]
        user_group_similarity = torch.zeros(num_users)

        for group_id, members in group_members.items():
            if group_id in train_group_to_index:
                group_index = train_group_to_index[group_id]
                for member_id in members:
                    if member_id in train_user_to_index:
                        user_index = train_user_to_index[member_id]
                        group_vector = train_group_result[group_index]
                        user_vector = train_user_result[user_index]
                        similarity = self.calculate_similarity(group_vector.unsqueeze(0), user_vector.unsqueeze(0))
                        user_group_similarity[user_index] = similarity
                
        return train_group_result, test_group_result, train_user_result, test_user_result, user_group_similarity, self.load_negative_file(self.group_negative_file, item_to_index), self.load_negative_file(self.user_negative_file, item_to_index), group_member_matrix
