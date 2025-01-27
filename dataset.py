import numpy as np
import torch
from datautil import get_max_item_id, load_rating_file_to_matrix, load_rating_file_to_list, load_negative, \
    load_group_member_to_dict, datasetToDataFrame, loadGroupMembers, load_negative_file, calculate_similarity \

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
        self.num_items = get_max_item_id(self.train_file, self.test_file, self.group_train_file, self.group_test_file)
        self.user_train_matrix = load_rating_file_to_matrix(self.train_file, path, self.num_items)
        self.user_test_matrix = load_rating_file_to_matrix(self.test_file, path, self.num_items)
        self.user_test_ratings = load_rating_file_to_list(self.test_file)
        self.user_test_negatives = load_negative(self.user_negative_file)
        self.num_users = self.user_train_matrix.shape[0] - 1
    
        # Group data
        self.group_train_matrix = load_rating_file_to_matrix(self.group_train_file, path, self.num_items)
        self.group_test_matrix = load_rating_file_to_matrix(self.group_test_file, path, self.num_items)
        self.group_test_ratings = load_rating_file_to_list(self.group_test_file)
        self.group_test_negatives = load_negative(self.group_negative_file)
        self.num_groups = self.group_train_matrix.shape[0] - 1
        self.group_member_dict = load_group_member_to_dict(self.group_member_file)

        self.train_g = None
        self.train_u = None
        self.test_g = None
        self.test_u = None
        self.unique_user_info = None
        self.unique_group_info = None 

    def getDataset(self):
        df_group_train = datasetToDataFrame(self.group_train_file, False) 
        df_user_train = datasetToDataFrame(self.train_file)

        train_groups = df_group_train['group_id'].unique() 
        max_item_id = max(df_group_train['item_id'].max(), df_user_train['item_id'].max())
        item_to_index = {item: idx for idx, item in enumerate(range(max_item_id + 1))}
        train_group_to_index = {group: index for index, group in enumerate(train_groups)}  
        
        # Update matrix with the same item index
        self.group_train_matrix = self.group_train_matrix[:, :max_item_id + 1]
        self.group_test_matrix = self.group_test_matrix[:, :max_item_id + 1]
        self.user_train_matrix = self.user_train_matrix[:, :max_item_id + 1]
        
        train_group_matrix = self.group_train_matrix.tocsr() 
        test_group_matrix = self.group_test_matrix.tocsr()
        train_user_matrix = self.user_train_matrix.tocsr()

        train_group_coo_matrix = train_group_matrix.tocoo()
        test_group_coo_matrix = test_group_matrix.tocoo()
        train_user_coo_matrix = train_user_matrix.tocoo()

        train_group_data = torch.FloatTensor(train_group_coo_matrix.data)
        test_group_data = torch.FloatTensor(test_group_coo_matrix.data)
        train_user_data = torch.FloatTensor(train_user_coo_matrix.data)

        train_group_indices = torch.LongTensor(np.vstack((train_group_coo_matrix.row, train_group_coo_matrix.col)))
        test_group_indices = torch.LongTensor(np.vstack((test_group_coo_matrix.row, test_group_coo_matrix.col)))
        train_user_indices = torch.LongTensor(np.vstack((train_user_coo_matrix.row, train_user_coo_matrix.col)))

        train_group_result = torch.sparse_coo_tensor(train_group_indices, train_group_data, torch.Size(train_group_coo_matrix.shape)).to_dense()
        test_group_result = torch.sparse_coo_tensor(test_group_indices, test_group_data, torch.Size(test_group_coo_matrix.shape)).to_dense()
        train_user_result = torch.sparse_coo_tensor(train_user_indices, train_user_data, torch.Size(train_user_coo_matrix.shape)).to_dense()
        
        group_members = loadGroupMembers(self.group_member_file) 
        num_groups = len(train_groups) 
        group_member_matrix = torch.zeros((num_groups, self.num_users + 1)) 
        
        for group_id, members in group_members.items(): 
            if group_id in train_group_to_index: 
                group_index = train_group_to_index[group_id] 
                for member_id in members: 
                    if member_id <= self.num_users: 
                        group_member_matrix[group_index, member_id] = 1 
                
        return train_group_result, test_group_result, train_user_result, load_negative_file(self.group_negative_file, item_to_index), group_member_matrix
