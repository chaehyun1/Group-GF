import torch
import numpy as np
import math


def count_users_in_groups(file_path): # 추가 
    group_user_count = {}
    
    with open(file_path, 'r') as f:
        for line in f:
            # 줄을 groupid와 users로 분리
            groupid, users = line.strip().split()
            # user 리스트를 쉼표로 분리하여 user 수를 셈
            user_count = len(users.split(','))
            # groupid를 키로, user 수를 값으로 딕셔너리에 저장
            group_user_count[int(groupid)] = user_count
    
    return group_user_count

def calculate_group_average_by_percentile(each_group_score, group_user_count, condition): # 추가
    # 그룹을 유저 수 기준으로 내림차순으로 정렬
    sorted_groups = sorted(group_user_count.items(), key=lambda x: x[1], reverse=True)
    
    # 전체 그룹 수 계산
    total_groups = len(sorted_groups)
    
    # 각 구간의 범위를 계산 (0~20%, 20~40%, ..., 80%~100%)
    ranges = [int(total_groups * (i / 5)) for i in range(6)]  # 5개 구간 (0%, 20%, 40%, 60%, 80%, 100%)
    
    # 각 구간에 속하는 그룹들의 hit 비율을 저장할 딕셔너리
    group_scores_by_percentile = {1: [], 2: [], 3: [], 4: [], 5: []}

    # 각 그룹이 어느 구간에 속하는지 계산
    for idx, (group, size) in enumerate(sorted_groups):
        # 그룹의 순서에 따라 구간을 할당
        if idx < ranges[1]:
            group_scores_by_percentile[1].append(each_group_score[group])  # 상위 20%
        elif idx < ranges[2]:
            group_scores_by_percentile[2].append(each_group_score[group])  # 20%~40%
        elif idx < ranges[3]:
            group_scores_by_percentile[3].append(each_group_score[group])  # 40%~60%
        elif idx < ranges[4]:
            group_scores_by_percentile[4].append(each_group_score[group])  # 60%~80%
        else:
            group_scores_by_percentile[5].append(each_group_score[group])  # 하위 20%

    # 각 구간의 hit 비율을 계산해서 출력
    for percentile in range(1, 6):
        group_scores = group_scores_by_percentile[percentile]
        if group_scores:
            average_score = sum(group_scores) / len(group_scores)
            print(f'상위 {percentile * 20}% 그룹의 평균 {condition} 비율: {average_score:.4f}')


def get_hit_k(pred_rank, k, groupSize=False):
    pred_rank_k = pred_rank[:, :k]
    hit = np.count_nonzero(pred_rank_k == 0)
    hit = hit / pred_rank.shape[0]
    
    if groupSize:
        group_user_count = count_users_in_groups('data/Mafengwo/groupMember.txt')
        each_group_score = {}
        # 각 그룹별 히트 여부를 기록
        for groupid, user_count in group_user_count.items():
            group_pred_rank = pred_rank_k[groupid]
            group_hit = np.count_nonzero(group_pred_rank == 0)
            each_group_score[groupid] = group_hit  # 유저 수가 아닌 그룹의 히트 횟수를 기록
        
        # 그룹 사이즈에 따른 평균 히트 비율 계산
        calculate_group_average_by_percentile(each_group_score, group_user_count, 'hit')
    
    return round(hit, 5)


def get_ndcg_k(pred_rank, k, groupSize=False):
    ndcgs = np.zeros(pred_rank.shape[0])
    for user in range(pred_rank.shape[0]):
        for j in range(k):
            if pred_rank[user][j] == 0:
                ndcgs[user] = math.log(2) / math.log(j+2)
    
    if groupSize:
        group_user_count = count_users_in_groups('data/Mafengwo/groupMember.txt')
        each_group_score = {}
        # 각 그룹별 NDCG 점수 계산
        for groupid, user_count in group_user_count.items():
            group_ndcg = ndcgs[groupid]
            each_group_score[groupid] = group_ndcg  # NDCG 점수를 그룹별로 저장

        # 그룹 사이즈에 따른 평균 NDCG 비율 계산
        calculate_group_average_by_percentile(each_group_score, group_user_count, 'ndcg')

    return np.round(np.mean(ndcgs), decimals=5)


def evaluate(model, test_ratings, test_negatives, device, k_list, type_m='group'):
    """Evaluate the performance (HitRatio, NDCG) of top-K recommendation"""
    model.eval()
    hits, ndcgs = [], []
    user_test, item_test = [], []

    for idx in range(len(test_ratings)):
        rating = test_ratings[idx]
        # Important
        # for testing, we put the ground-truth item as the first one and remaining are negative samples
        # for evaluation, we check whether prediction's idx is the ground-truth (idx with 0)
        items = [rating[1]]
        items.extend(test_negatives[idx])

        # an alternative
        # to avoid the dead relu issue where model predicts all candidate items with score 1.0 and thus lead to invalid predictions
        # we can put the ground-truth item to the last 
        # for evaluation, the checked ground-truth idx should be 100 in Line 17 & Line 8
        # items = test_negatives[idx] + [rating[1]]

        item_test.append(items)
        user_test.append(np.full(len(items), rating[0]))

    users_var = torch.LongTensor(user_test).to(device)
    items_var = torch.LongTensor(item_test).to(device)

    bsz = len(test_ratings)
    item_len = len(test_negatives[0]) + 1

    users_var = users_var.view(-1)
    items_var = items_var.view(-1)

    if type_m == 'group':
        predictions = model(users_var, None, items_var)
    elif type_m == 'user':
        predictions = model(None, users_var, items_var)

    predictions = torch.reshape(predictions, (bsz, item_len))

    pred_score = predictions.data.cpu().numpy()
    # print(pred_score[:10, ])
    pred_rank = np.argsort(pred_score * -1, axis=1)
    for k in k_list:
        hits.append(get_hit_k(pred_rank, k))
        ndcgs.append(get_ndcg_k(pred_rank, k))

    return hits, ndcgs
