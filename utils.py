from collections import defaultdict
import glob
import hashlib
import math
import os
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
from scipy.sparse.linalg import eigsh
import torch.nn as nn


def fit_delimiter(string='', length=80, delimiter="="):
    result_len = length - len(string)
    half_len = math.floor(result_len / 2)
    result = delimiter * half_len + string + delimiter * half_len
    return result


def init_torch_seeds(seed=0):
    torch.manual_seed(seed)
    if seed == 0:  # slower, more reproducible
        cudnn.benchmark, cudnn.deterministic = False, True
    else:  # faster, less reproducible
        cudnn.benchmark, cudnn.deterministic = True, False


def zipdir(path, ziph, include_format):
    for root, dirs, files in os.walk(path):
        for file in files:
            if os.path.splitext(file)[-1] in include_format:
                filename = os.path.join(root, file)
                arcname = os.path.relpath(os.path.join(root, file), os.path.join(path, '..'))
                ziph.write(filename, arcname)


def increment_path(path, exist_ok=True, sep=''):
    # Increment path, i.e. runs/exp --> runs/exp{sep}0, runs/exp{sep}1 etc.
    path = Path(path)  # os-agnostic
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}{sep}*")  # similar paths
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 2  # increment number
        return f"{path}{sep}{n}"  # update path


def get_normalized_features(X):
    # X.shape=(num_nodes, num_features)
    means = np.mean(X, axis=0)  # mean of features, shape:(num_features,)
    X = X - means.reshape((1, -1))
    stds = np.std(X, axis=0)  # std of features, shape:(num_features,)
    X = X / stds.reshape((1, -1))
    return X, means, stds


def calculate_laplacian_matrix(adj_mat, mat_type):
    n_vertex = adj_mat.shape[0]

    # row sum
    deg_mat_row = np.asmatrix(np.diag(np.sum(adj_mat, axis=1)))
    # column sum
    # deg_mat_col = np.asmatrix(np.diag(np.sum(adj_mat, axis=0)))
    deg_mat = deg_mat_row

    adj_mat = np.asmatrix(adj_mat)
    id_mat = np.asmatrix(np.identity(n_vertex))

    if mat_type == 'com_lap_mat':
        # Combinatorial
        com_lap_mat = deg_mat - adj_mat
        return com_lap_mat
    elif mat_type == 'wid_rw_normd_lap_mat':
        # For ChebConv
        rw_lap_mat = np.matmul(np.linalg.matrix_power(deg_mat, -1), adj_mat)
        rw_normd_lap_mat = id_mat - rw_lap_mat
        lambda_max_rw = eigsh(rw_lap_mat, k=1, which='LM', return_eigenvectors=False)[0]
        wid_rw_normd_lap_mat = 2 * rw_normd_lap_mat / lambda_max_rw - id_mat
        return wid_rw_normd_lap_mat
    elif mat_type == 'hat_rw_normd_lap_mat':
        # For GCNConv
        wid_deg_mat = deg_mat + id_mat
        wid_adj_mat = adj_mat + id_mat
        hat_rw_normd_lap_mat = np.matmul(np.linalg.matrix_power(wid_deg_mat, -1), wid_adj_mat)
        return hat_rw_normd_lap_mat
    else:
        raise ValueError(f'ERROR: {mat_type} is unknown.')


def maksed_mse_loss(input, target, mask_value=-1):
    mask = target == mask_value
    out = (input[~mask] - target[~mask]) ** 2
    loss = out.mean()
    return loss


def top_k_acc(y_true_seq, y_pred_seq, k):
    hit = 0
    # Convert to binary relevance (nonzero is relevant).
    for y_true, y_pred in zip(y_true_seq, y_pred_seq):
        top_k_rec = y_pred.argsort()[-k:][::-1]
        idx = np.where(top_k_rec == y_true)[0]
        if len(idx) != 0:
            hit += 1
    return hit / len(y_true_seq)


def mAP_metric(y_true_seq, y_pred_seq, k):
    # AP: area under PR curve
    # But in next POI rec, the number of positive sample is always 1. Precision is not well defined.
    # Take def of mAP from Personalized Long- and Short-term Preference Learning for Next POI Recommendation
    rlt = 0
    for y_true, y_pred in zip(y_true_seq, y_pred_seq):
        rec_list = y_pred.argsort()[-k:][::-1]
        r_idx = np.where(rec_list == y_true)[0]
        if len(r_idx) != 0:
            rlt += 1 / (r_idx[0] + 1)
    return rlt / len(y_true_seq)


def MRR_metric(y_true_seq, y_pred_seq):
    """Mean Reciprocal Rank: Reciprocal of the rank of the first relevant item """
    rlt = 0
    for y_true, y_pred in zip(y_true_seq, y_pred_seq):
        rec_list = y_pred.argsort()[-len(y_pred):][::-1]
        r_idx = np.where(rec_list == y_true)[0][0]
        rlt += 1 / (r_idx + 1)
    return rlt / len(y_true_seq)


def top_k_acc_last_timestep(y_true_seq, y_pred_seq, k):
    """ next poi metrics """
    y_true = y_true_seq[-1]
    y_pred = y_pred_seq[-1]
    top_k_rec = y_pred.argsort()[-k:][::-1]
    idx = np.where(top_k_rec == y_true)[0]
    if len(idx) != 0:
        return 1
    else:
        return 0


def mAP_metric_last_timestep(y_true_seq, y_pred_seq, k):
    """ next poi metrics """
    # AP: area under PR curve
    # But in next POI rec, the number of positive sample is always 1. Precision is not well defined.
    # Take def of mAP from Personalized Long- and Short-term Preference Learning for Next POI Recommendation
    y_true = y_true_seq[-1]
    y_pred = y_pred_seq[-1]
    rec_list = y_pred.argsort()[-k:][::-1]
    r_idx = np.where(rec_list == y_true)[0]
    if len(r_idx) != 0:
        return 1 / (r_idx[0] + 1)
    else:
        return 0


def MRR_metric_last_timestep(y_true_seq, y_pred_seq):
    """ next poi metrics """
    # Mean Reciprocal Rank: Reciprocal of the rank of the first relevant item
    y_true = y_true_seq[-1]
    y_pred = y_pred_seq[-1]
    rec_list = y_pred.argsort()[-len(y_pred):][::-1]
    r_idx = np.where(rec_list == y_true)[0][0]
    return 1 / (r_idx + 1)


def array_round(x, k=4):
    # For a list of float values, keep k decimals of each element
    return list(np.around(np.array(x), k))


def NDCG_metric_last_timestep(label_pois, pred_pois, k=20):
    dcg = 0
    idcg = 0
    sorted_pred_indices = np.argsort(-pred_pois[-1])[:k]
    relevant_items = set(label_pois)
    for i, index in enumerate(sorted_pred_indices):
        if index in relevant_items:
            dcg += 1 / np.log2(i + 2)
    ideal_order = sorted(relevant_items, key=lambda x: pred_pois[-1][x], reverse=True)[:k]
    for i, item in enumerate(ideal_order):
        idcg += 1 / np.log2(i + 2)
    if idcg == 0:
        return 0
    return dcg / idcg

# 定义计算 recall 的函数
def recall_metric_last_timestep(label_pois, pred_pois, k=20):
    sorted_pred_indices = np.argsort(-pred_pois[-1])[:k]
    relevant_items = set(label_pois)
    retrieved_relevant = len(set(sorted_pred_indices).intersection(relevant_items))
    total_relevant = len(relevant_items)
    if total_relevant == 0:
        return 0
    return retrieved_relevant / total_relevant


def calculate_prediction_stability(top1_accs, top5_accs, mrrs):
    """计算预测稳定性分数"""
    # 计算各个指标的标准差
    top1_std = np.std(top1_accs)
    top5_std = np.std(top5_accs)
    mrr_std = np.std(mrrs)
    
    # 计算稳定性分数 (1 - 归一化的标准差)
    stability = 1 - (top1_std + top5_std + mrr_std) / 3
    return max(0, stability)  # 确保非负

def calculate_loss_balance(poi_loss, time_loss, cat_loss):
    """计算损失平衡性分数"""
    # 计算各个损失的相对比例
    total_loss = poi_loss + time_loss + cat_loss
    poi_ratio = poi_loss / total_loss
    time_ratio = time_loss / total_loss
    cat_ratio = cat_loss / total_loss
    
    # 计算与理想平衡状态(1/3)的偏差
    ideal_ratio = 1/3
    balance = 1 - (abs(poi_ratio - ideal_ratio) + 
                  abs(time_ratio - ideal_ratio) + 
                  abs(cat_ratio - ideal_ratio)) / 2
    
    return max(0, balance)  # 确保非负

def calculate_user_cooccurrence(train_df):
    """计算用户共现矩阵,带缓存机制"""
    # 生成缓存文件名
    cache_path = os.path.join('dataset', f"cache_cooccurrence_{hashlib.md5(str(train_df['user_id'].tolist()).encode()).hexdigest()}.pkl")
    
    # 如果缓存存在,直接加载
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    user_ids = [each for each in list(set(train_df['user_id'].to_list()))]
    user_id2idx_dict = dict(zip(user_ids, range(len(user_ids))))
    
    # 否则计算共现矩阵
    # 1. 按时间区间和POI类别统计用户共现
    train_df['time_interval'] = pd.to_datetime(train_df['local_time']).dt.floor('1H')
    
    # 计算POI类别和POI细分类别的共现
    co_occurrence_catname = train_df.groupby(['time_interval', 'POI_catname'])['user_id'].nunique()
    co_occurrence_category = train_df.groupby(['time_interval', 'POI_category'])['user_id'].nunique()
    
    # 筛选有效共现
    co_occurrence_catname = co_occurrence_catname[co_occurrence_catname > 1]
    co_occurrence_category = co_occurrence_category[co_occurrence_category > 1]
    
    # 2. 计算用户对共现频率
    user_pairs = defaultdict(int)
    for (time_interval, category), _ in co_occurrence_catname.items():
        users = train_df[(train_df['time_interval'] == time_interval) & 
                        (train_df['POI_catname'] == category)]['user_id'].unique()
        for i in range(len(users)):
            for j in range(i + 1, len(users)):
                user_pairs[(users[i], users[j])] += 1
                
    for (time_interval, category), _ in co_occurrence_category.items():
        users = train_df[(train_df['time_interval'] == time_interval) & 
                        (train_df['POI_category'] == category)]['user_id'].unique()
        for i in range(len(users)):
            for j in range(i + 1, len(users)):
                user_pairs[(users[i], users[j])] += 1
    
    # 3. 计算用户活跃度
    user_activity = train_df['user_id'].value_counts().to_dict()
    
    # 4. 计算归一化共现矩阵
    num_users = len(user_activity)
    cooccurrence_matrix = torch.zeros((num_users, num_users))
    
    for (user1, user2), freq in user_pairs.items():
        # 使用Jaccard相似度进行归一化
        activity1 = user_activity[user1]
        activity2 = user_activity[user2]
        similarity = freq / (activity1 + activity2 - freq)
        
        # 转换为索引
        idx1 = user_id2idx_dict[user1]
        idx2 = user_id2idx_dict[user2]
        cooccurrence_matrix[idx1, idx2] = similarity
        cooccurrence_matrix[idx2, idx1] = similarity
    
    # 保存到缓存
    os.makedirs('dataset', exist_ok=True)
    with open(cache_path, 'wb') as f:
        pickle.dump(cooccurrence_matrix, f)
        
    return cooccurrence_matrix
