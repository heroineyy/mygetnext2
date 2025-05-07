import logging
import logging
import os
import pathlib
import pickle
import zipfile
from pathlib import Path
from sklearn.feature_extraction import FeatureHasher

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from sklearn.preprocessing import OneHotEncoder
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import torch.nn.functional as F
import hashlib
import requests
import json
from collections import defaultdict

from dataloader import load_graph_adj_mtx, load_graph_node_features
from model import GCN, GenericEmbeddings,Time2Vec, GatingNetwork,TransformerModel, POIFeatureFusion,EnhancedUserEmbedding
from param_parser import parameter_parser
from utils import increment_path, calculate_laplacian_matrix, zipdir, top_k_acc_last_timestep, \
    mAP_metric_last_timestep, MRR_metric_last_timestep, maksed_mse_loss, NDCG_metric_last_timestep,\
    recall_metric_last_timestep,calculate_user_cooccurrence,calculate_prediction_stability,calculate_loss_balance



def train(args):
    args.save_dir = increment_path(Path(args.project) / args.name, exist_ok=args.exist_ok, sep='-')
    if not os.path.exists(args.save_dir): os.makedirs(args.save_dir)

    # Setup logger
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        filename=os.path.join(args.save_dir, f"log_training.txt"),
                        filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    logging.getLogger('matplotlib.font_manager').disabled = True

    # Save run settings
    logging.info(args)
    with open(os.path.join(args.save_dir, 'args.yaml'), 'w') as f:
        yaml.dump(vars(args), f, sort_keys=False)

    # Save python code
    zipf = zipfile.ZipFile(os.path.join(args.save_dir, 'code.zip'), 'w', zipfile.ZIP_DEFLATED)
    zipdir(pathlib.Path().absolute(), zipf, include_format=['.py'])
    zipf.close()

    # %% ====================== Load data ======================
    # Read check-in train data
    train_df = pd.read_csv(args.data_train)
    val_df = pd.read_csv(args.data_val)


    # Build POI graph (built from train_df)
    print('Loading POI graph...')
    raw_A = load_graph_adj_mtx(args.data_adj_mtx)
    raw_X = load_graph_node_features(args.data_node_feats,
                                     args.feature1,
                                     args.feature2,
                                     args.feature3,
                                     args.feature4)
    logging.info(
        f"raw_X.shape: {raw_X.shape}; "
        f"Four features: {args.feature1}, {args.feature2}, {args.feature3}, {args.feature4}.")
    logging.info(f"raw_A.shape: {raw_A.shape}; Edge from row_index to col_index with weight (frequency).")
    num_pois = raw_X.shape[0]

    # One-hot encoding poi categories
    logging.info('One-hot encoding poi categories id')
    one_hot_encoder = OneHotEncoder()
    cat_list = list(raw_X[:, 1])
    one_hot_encoder.fit(list(map(lambda x: [x], cat_list)))
    one_hot_rlt = one_hot_encoder.transform(list(map(lambda x: [x], cat_list))).toarray()
    num_cats = one_hot_rlt.shape[-1]
    X = np.zeros((num_pois, raw_X.shape[-1] - 1 + num_cats), dtype=np.float32)
    X[:, 0] = raw_X[:, 0]
    X[:, 1:num_cats + 1] = one_hot_rlt
    X[:, num_cats + 1:] = raw_X[:, 2:]
    logging.info(f"After one hot encoding poi cat, X.shape: {X.shape}")
    logging.info(f'POI categories: {list(one_hot_encoder.categories_[0])}')


    # Normalization
    print('Laplician matrix...')
    A = calculate_laplacian_matrix(raw_A, mat_type='hat_rw_normd_lap_mat')

    # POI id to index
    nodes_df = pd.read_csv(args.data_node_feats)
    poi_ids = list(set(nodes_df['node_name/poi_id'].tolist()))
    poi_id2idx_dict = dict(zip(poi_ids, range(len(poi_ids))))

    # User id to index
    user_ids = [str(each) for each in list(set(train_df['user_id'].to_list()))]
    user_id2idx_dict = dict(zip(user_ids, range(len(user_ids))))
    num_users=len(user_ids)


    # Cat name to index
    cat_names = list(set(train_df['POI_catname'].tolist()))
    cat_name2id_dict = dict(zip(cat_names, range(len(cat_names))))
    num_cats = len(cat_names)

    # category name to index
    category_names = list(set(train_df['POI_category'].tolist()))
    category_name2id_dict = dict(zip(category_names, range(len(category_names))))
    num_categroys = len(category_names)


    poi_idx2cat_id_dict = {}
    poi_idx2category_id_dict = {}

    poi_idx2cat_name_dict = {}
    poi_idx2category_name_dict = {}

    for i, row in train_df.iterrows():
        poi_idx2cat_id_dict[poi_id2idx_dict[row['POI_id']]] = cat_name2id_dict[row['POI_catname']]
        poi_idx2category_id_dict[poi_id2idx_dict[row['POI_id']]]= category_name2id_dict[row['POI_category']]
        poi_idx2cat_name_dict[poi_id2idx_dict[row['POI_id']]] = row['POI_catname']
        poi_idx2category_name_dict[poi_id2idx_dict[row['POI_id']]] = row['POI_category']


    # %% ====================== Define Dataset ======================
    class TrajectoryDatasetTrain(Dataset):
        def __init__(self, train_df):
            self.df = train_df
            self.traj_seqs = []  # traj id: user id + traj no.
            self.input_seqs = []
            self.label_seqs = []

# 生成缓存文件名
            cache_path = os.path.join('dataset', f"cache_trajectory_train_{hashlib.md5(str(train_df['trajectory_id'].tolist()).encode()).hexdigest()}.pkl")
            
            # 如果缓存存在，直接加载
            if os.path.exists(cache_path):
                with open(cache_path, 'rb') as f:
                    cached_data = pickle.load(f)
                    self.traj_seqs = cached_data['traj_seqs']
                    self.input_seqs = cached_data['input_seqs']
                    self.label_seqs = cached_data['label_seqs']
                return

            # 否则处理数据并缓存
            for traj_id in tqdm(set(train_df['trajectory_id'].tolist())):
                traj_df = train_df[train_df['trajectory_id'] == traj_id]
                poi_ids = traj_df['POI_id'].to_list()
                poi_idxs = [poi_id2idx_dict[each] for each in poi_ids]
                time_feature = traj_df[args.time_feature].to_list()
                args.lat_feature = 'latitude'
                args.lon_feature = 'longitude'
                lat_feature = traj_df[args.lat_feature].to_list()
                lon_feature = traj_df[args.lon_feature].to_list()

                args.week_feature = 'day_of_week'
                week_feature = traj_df[args.week_feature].to_list()

                input_seq = []
                label_seq = []
                for i in range(len(poi_idxs) - 1):
                    input_seq.append((poi_idxs[i], time_feature[i], lat_feature[i], lon_feature[i], week_feature[i]))
                    label_seq.append((poi_idxs[i + 1], time_feature[i + 1], lat_feature[i+1], lon_feature[i+1], week_feature[i]))

                if len(input_seq) < args.short_traj_thres:
                    continue

                self.traj_seqs.append(traj_id)
                self.input_seqs.append(input_seq)
                self.label_seqs.append(label_seq)

            # 保存到缓存
            with open(cache_path, 'wb') as f:
                pickle.dump({
                    'traj_seqs': self.traj_seqs,
                    'input_seqs': self.input_seqs,
                    'label_seqs': self.label_seqs
                }, f)

        def __len__(self):
            assert len(self.input_seqs) == len(self.label_seqs) == len(self.traj_seqs)
            return len(self.traj_seqs)

        def __getitem__(self, index):
            return (self.traj_seqs[index], self.input_seqs[index], self.label_seqs[index])

    class TrajectoryDatasetVal(Dataset):
        def __init__(self, df):
            self.df = df
            self.traj_seqs = []
            self.input_seqs = []
            self.label_seqs = []


            cache_path = os.path.join('dataset', f"cache_trajectory_val_{hashlib.md5(str(df['trajectory_id'].tolist()).encode()).hexdigest()}.pkl")
            
            # 如果缓存存在，直接加载
            if os.path.exists(cache_path):
                with open(cache_path, 'rb') as f:
                    cached_data = pickle.load(f)
                    self.traj_seqs = cached_data['traj_seqs']
                    self.input_seqs = cached_data['input_seqs']
                    self.label_seqs = cached_data['label_seqs']
                return

            # 否则处理数据并缓存
            for traj_id in tqdm(set(df['trajectory_id'].tolist())):
                user_id = traj_id.split('_')[0]

                # Ignore user if not in training set
                if user_id not in user_id2idx_dict.keys():
                    continue

                # Get POIs idx in this trajectory
                traj_df = df[df['trajectory_id'] == traj_id]
                poi_ids = traj_df['POI_id'].to_list()
                poi_idxs = []
                time_feature = traj_df[args.time_feature].to_list()
                args.lat_feature = 'latitude'
                args.lon_feature = 'longitude'
                lat_feature = traj_df[args.lat_feature].to_list()
                lon_feature = traj_df[args.lon_feature].to_list()

                args.week_feature = 'day_of_week'
                week_feature = traj_df[args.week_feature].to_list()

                for each in poi_ids:
                    if each in poi_id2idx_dict.keys():
                        poi_idxs.append(poi_id2idx_dict[each])
                    else:
                        # Ignore poi if not in training set
                        continue

                # Construct input seq and label seq
                input_seq = []
                label_seq = []
                for i in range(len(poi_idxs) - 1):
                    input_seq.append((poi_idxs[i], time_feature[i], lat_feature[i], lon_feature[i], week_feature[i]))
                    label_seq.append((poi_idxs[i + 1], time_feature[i + 1], lat_feature[i+1], lon_feature[i+1], week_feature[i]))

                # Ignore seq if too short
                if len(input_seq) < args.short_traj_thres:
                    continue

                self.input_seqs.append(input_seq)
                self.label_seqs.append(label_seq)
                self.traj_seqs.append(traj_id)

            # 保存到缓存
            with open(cache_path, 'wb') as f:
                pickle.dump({
                    'traj_seqs': self.traj_seqs,
                    'input_seqs': self.input_seqs,
                    'label_seqs': self.label_seqs
                }, f)

        def __len__(self):
            assert len(self.input_seqs) == len(self.label_seqs) == len(self.traj_seqs)
            return len(self.traj_seqs)

        def __getitem__(self, index):
            return (self.traj_seqs[index], self.input_seqs[index], self.label_seqs[index])

    cooccurrence_matrix = calculate_user_cooccurrence(train_df).to(args.device)
    # %% ====================== Define dataloader ======================
    print('Prepare dataloader...')
    train_dataset = TrajectoryDatasetTrain(train_df)
    val_dataset = TrajectoryDatasetVal(val_df)

    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch,
                              shuffle=True, drop_last=False,
                              pin_memory=True, num_workers=args.workers,
                              collate_fn=lambda x: x)
    val_loader = DataLoader(val_dataset,
                            batch_size=args.batch,
                            shuffle=False, drop_last=False,
                            pin_memory=True, num_workers=args.workers,
                            collate_fn=lambda x: x)

    # %% ====================== Build Models ======================
    # Model1: POI embedding model
    if isinstance(X, np.ndarray):
        X = torch.from_numpy(X)
        A = torch.from_numpy(A)
    X = X.to(device=args.device, dtype=torch.float)
    A = A.to(device=args.device, dtype=torch.float)

    args.gcn_nfeat = X.shape[1]
    poi_gcn_model = GCN(ninput=args.gcn_nfeat,
                          nhid=args.gcn_nhid,
                          noutput=args.poi_embed_dim,
                          dropout=args.gcn_dropout)


    # %% Model2: User embedding model
    user_embed_model = EnhancedUserEmbedding(num_users,args.user_embed_dim,cooccurrence_matrix,args.similar_user_k)
    poi_embed_model = GenericEmbeddings(num_pois, args.poi_embed_dim)
    cat_embed_model = GenericEmbeddings(num_cats, args.cat_embed_dim)
    categroy_embed_model = GenericEmbeddings(num_categroys, args.cat2_embed_dim)


    # 初始化POI特征融合模块
    poi_fusion = POIFeatureFusion(args.poi_embed_dim, args.fuse_way)

    # %% Model3: Time week lat lon Model
    time_embed_model = Time2Vec('sin', out_dim=args.time_embed_dim)
    week_embed_model = GenericEmbeddings(7, args.week_embed_dim)
    lat_embed_model = Time2Vec('sin', out_dim=args.geo_embed_dim)
    lon_embed_model = Time2Vec('sin', out_dim=args.geo_embed_dim)



    # %% Model6: Sequence model
    args.seq_input_embed = args.user_embed_dim + args.time_embed_dim +args.week_embed_dim + +args.geo_embed_dim*2 + args.poi_embed_dim + args.cat_embed_dim + args.cat2_embed_dim
    args.seq_input_embed2 = args.time_embed_dim + args.week_embed_dim + +args.geo_embed_dim * 2 + args.cat_embed_dim + args.cat2_embed_dim
    args.seq_input_embed3 = args.time_embed_dim + args.week_embed_dim + +args.geo_embed_dim * 2
    seq_model = TransformerModel(num_pois,
                                 num_cats,
                                 args.seq_input_embed,
                                 args.transformer_nhead,
                                 args.transformer_nhid,
                                 args.transformer_nlayers,
                                 dropout=args.transformer_dropout)

    embed_fuse_model2 = GatingNetwork(args.seq_input_embed, args.gating_hidden_dim).to(args.device)
    align_layer = nn.Linear(args.seq_input_embed3, args.seq_input_embed).to(args.device) # 将seq_input_embed3线性转换为seq_input_embed,对齐
    multihead_attn = nn.MultiheadAttention(embed_dim=args.seq_input_embed, num_heads=4).to(args.device)
    layer_norm = nn.LayerNorm(args.seq_input_embed).to(args.device)


    # Define overall loss and optimizer
    optimizer = optim.Adam(params=list(poi_embed_model.parameters()) +
                                  list(poi_gcn_model.parameters()) +
                                  list(user_embed_model.parameters()) +
                                  list(time_embed_model.parameters()) +
                                  list(lat_embed_model.parameters()) +
                                  list(lon_embed_model.parameters()) +
                                  list(week_embed_model.parameters()) +
                                  list(cat_embed_model.parameters()) +
                                  list(categroy_embed_model.parameters()) +
                                  list(embed_fuse_model2.parameters()) +
                                  list(align_layer.parameters()) +
                                  list(multihead_attn.parameters()) +
                                  list(layer_norm.parameters()) +
                                  list(poi_fusion.parameters()) +
                                  list(seq_model.parameters()),
                           lr=args.lr,
                           weight_decay=args.weight_decay)

    criterion_poi = nn.CrossEntropyLoss(ignore_index=-1)  # -1 is padding
    criterion_cat = nn.CrossEntropyLoss(ignore_index=-1)  # -1 is padding
    criterion_categroy = nn.CrossEntropyLoss(ignore_index=-1)  # -1 is padding
    criterion_week = nn.CrossEntropyLoss(ignore_index=-1)  # -1 is padding
    criterion_time = maksed_mse_loss

    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', verbose=True, factor=args.lr_scheduler_factor)

    # %% Tool functions for training

    # 封装嵌入转换函数，支持数值型和类别型特征
    def get_embedding(value, embed_model, device, is_numeric=False):
        dtype = torch.float if is_numeric else torch.long
        input_tensor = torch.tensor(value, dtype=dtype).to(device=device)
        embedding = embed_model(input_tensor)
        return torch.squeeze(embedding).to(device=device)


    def input_traj_to_embeddings(sample, poi_gcn_embeddings):
        # todo:改进建议:
        # 添加特征重要性权重
        # 使用更复杂的特征融合方式
        # 考虑添加位置编码
        # 增加特征交互层
        # 添加dropout防止过拟合
        # Parse sample
        traj_id = sample[0]
        input_seq = [each[0] for each in sample[1]]
        input_seq_time = [each[1] for each in sample[1]]
        input_seq_lat = [each[2] for each in sample[1]]
        input_seq_lon = [each[3] for each in sample[1]]
        input_seq_week = [each[4] for each in sample[1]]
        input_seq_cat = [poi_idx2cat_id_dict[each] for each in input_seq]
        input_seq_category = [poi_idx2category_id_dict[each] for each in input_seq]

        # todo:如何让用到下面的信息?
        # input_seq_cat_name = [poi_idx2cat_name_dict[each] for each in input_seq]
        # input_seq_category_name = [poi_idx2category_name_dict[each] for each in input_seq]
        
        # 3. 获取用户和时空特征
        user_id = traj_id.split('_')[0]
        user_embedding = get_embedding(user_id2idx_dict[user_id], user_embed_model, args.device)  # shape: [user_embed_dim]
        
        # 4. 获取最后一个点的时空信息
        last_time_embedding = get_embedding([sample[2][-1][1]], time_embed_model, args.device, is_numeric=True)  # shape: [time_embed_dim]
        last_lat_embedding = get_embedding([np.radians(sample[2][-1][2])], lat_embed_model, args.device, is_numeric=True)  # shape: [geo_embed_dim]
        last_lon_embedding = get_embedding([np.radians(sample[2][-1][3])], lon_embed_model, args.device, is_numeric=True)  # shape: [geo_embed_dim]
        last_week_embedding = get_embedding([sample[2][-1][4]], week_embed_model, args.device)  # shape: [week_embed_dim]
        
        last_st_embeddingq = torch.cat((last_time_embedding, last_lat_embedding, last_lon_embedding, last_week_embedding), dim=-1)  # shape: [time_embed_dim + 2*geo_embed_dim + week_embed_dim]
        aligned_last_st_embedding = align_layer(last_st_embeddingq)  # shape: [seq_input_embed]
        
        # 5. 构建序列特征
        input_seq_embed = []
        for idx in range(len(input_seq)):
            # 获取 GCN embedding
            poi_gcn_embedding = poi_gcn_embeddings[input_seq[idx]]  # shape: [poi_embed_dim]
            poi_gcn_embedding = torch.squeeze(poi_gcn_embedding).to(device=args.device)

            poi_embedding = get_embedding(input_seq[idx], poi_embed_model, args.device)
            
            # 使用POI特征融合模块
            fused_poi_embedding = poi_fusion(poi_embedding, poi_gcn_embedding)
            
            # 获取时空特征
            time_embedding = get_embedding([input_seq_time[idx]], time_embed_model, args.device, is_numeric=True)  # shape: [time_embed_dim]
            lat_embedding = get_embedding([np.radians(input_seq_lat[idx])], lat_embed_model, args.device, is_numeric=True)  # shape: [geo_embed_dim]
            lon_embedding = get_embedding([np.radians(input_seq_lon[idx])], lon_embed_model, args.device, is_numeric=True)  # shape: [geo_embed_dim]
            week_embedding = get_embedding([input_seq_week[idx]], week_embed_model, args.device)  # shape: [week_embed_dim]
            cat_embedding = get_embedding([input_seq_cat[idx]], cat_embed_model, args.device)  # shape: [cat_embed_dim]
            cat2_embedding = get_embedding([input_seq_category[idx]], categroy_embed_model, args.device)  # shape: [cat2_embed_dim]
            
            # 融合时空特征
            st_features = torch.cat([user_embedding, fused_poi_embedding, time_embedding, week_embedding, lat_embedding, lon_embedding, cat_embedding, cat2_embedding], dim=-1)   # shape: [seq_input_embed]
            gate_values = embed_fuse_model2(st_features)   # shape: [seq_input_embed]
            gated_st_features = st_features * gate_values  # shape: [seq_input_embed]

            input_seq_embed.append(gated_st_features)
        
        # 6. 使用注意力机制融合序列特征和最后一个点的时空信息
        input_seq_embed = torch.stack(input_seq_embed, dim=0)  # shape: [seq_len, seq_input_embed]
        query = aligned_last_st_embedding.unsqueeze(0).expand(input_seq_embed.size(0), -1)  # shape: [seq_len, seq_input_embed]
        
        # 使用全局定义的注意力层和归一化层
        attn_output, _ = multihead_attn(query, input_seq_embed, input_seq_embed)  # shape: [seq_len, seq_input_embed]
        
        # 7. 添加残差连接
        attn_output = attn_output + input_seq_embed  # shape: [seq_len, seq_input_embed]
        
        # 使用全局定义的归一化层
        final_features = layer_norm(attn_output)  # shape: [seq_len, seq_input_embed]

        return final_features


    # %% ====================== Train ======================
    poi_embed_model = poi_embed_model.to(device=args.device)
    poi_gcn_model = poi_gcn_model.to(device=args.device)
    user_embed_model = user_embed_model.to(device=args.device)
    time_embed_model = time_embed_model.to(device=args.device)
    lat_embed_model = lat_embed_model.to(device=args.device)
    lon_embed_model = lon_embed_model.to(device=args.device)
    week_embed_model = week_embed_model.to(device=args.device)
    cat_embed_model = cat_embed_model.to(device=args.device)
    categroy_embed_model = categroy_embed_model.to(device=args.device)
    embed_fuse_model2 = embed_fuse_model2.to(device=args.device)
    align_layer = align_layer.to(device=args.device)
    seq_model = seq_model.to(device=args.device)
    multihead_attn = multihead_attn.to(device=args.device)
    layer_norm = layer_norm.to(device=args.device)
    poi_fusion = poi_fusion.to(device=args.device)

    # %% Loop epoch
    # For plotting
    train_epochs_top1_acc_list = []
    train_epochs_top5_acc_list = []
    train_epochs_top10_acc_list = []
    train_epochs_top20_acc_list = []
    train_epochs_mAP20_list = []
    train_epochs_mrr_list = []
    train_epochs_ndcg20_list = []
    train_epochs_recall20_list = []
    train_epochs_loss_list = []
    train_epochs_poi_loss_list = []
    train_epochs_time_loss_list = []
    train_epochs_cat_loss_list = []
    val_epochs_top1_acc_list = []
    val_epochs_top5_acc_list = []
    val_epochs_top10_acc_list = []
    val_epochs_top20_acc_list = []
    val_epochs_mAP20_list = []
    val_epochs_mrr_list = []
    val_epochs_ndcg20_list = []
    val_epochs_recall20_list = []
    val_epochs_loss_list = []
    val_epochs_poi_loss_list = []
    val_epochs_time_loss_list = []
    val_epochs_cat_loss_list = []
    # For saving ckpt
    max_val_score = -np.inf

    for epoch in range(args.epochs):
        logging.info(f"{'*' * 50}Epoch:{epoch:03d}{'*' * 50}\n")
        poi_gcn_model.train()
        user_embed_model.train()
        poi_embed_model.train()
        time_embed_model.train()
        lat_embed_model.train()
        lon_embed_model.train()
        week_embed_model.train()
        cat_embed_model.train()
        categroy_embed_model.train()
        embed_fuse_model2.train()
        align_layer.train()
        seq_model.train()
        multihead_attn.train()
        layer_norm.train()
        poi_fusion.train()

        train_batches_top1_acc_list = []
        train_batches_top5_acc_list = []
        train_batches_top10_acc_list = []
        train_batches_top20_acc_list = []
        train_batches_mAP20_list = []
        train_batches_mrr_list = []
        train_batches_ndcg20_list = []
        train_batches_recall20_list = []
        train_batches_loss_list = []
        train_batches_poi_loss_list = []
        train_batches_time_loss_list = []
        train_batches_week_loss_list = []
        train_batches_cat_loss_list = []
        train_batches_lat_loss_list = []
        train_batches_lon_loss_list = []
        train_batches_categroy_loss_list = []
        src_mask = seq_model.generate_square_subsequent_mask(args.batch).to(args.device)
        # Loop batch
        for b_idx, batch in enumerate(train_loader):
            if len(batch) != args.batch:
                src_mask = seq_model.generate_square_subsequent_mask(len(batch)).to(args.device)

            # For padding
            batch_input_seqs = []
            batch_seq_lens = []
            batch_seq_embeds = []
            batch_seq_labels_poi = []
            batch_seq_labels_time = []
            batch_seq_labels_lat = []
            batch_seq_labels_lon = []
            batch_seq_labels_week = []
            batch_seq_labels_cat = []
            batch_seq_labels_categroy = []

            poi_gcn_embeddings = poi_gcn_model(X, A)

            # Convert input seq to embeddings
            for sample in batch:
                input_seq = [each[0] for each in sample[1]]
                label_seq = [each[0] for each in sample[2]]
                label_seq_time = [each[1] for each in sample[2]]
                label_seq_lat = [np.radians(each[2]) for each in sample[2]]
                label_seq_lon = [np.radians(each[3]) for each in sample[2]]
                label_seq_week = [each[4] for each in sample[2]]

                label_seq_cats = [poi_idx2cat_id_dict[each] for each in label_seq]
                label_seq_category = [poi_idx2category_id_dict[each] for each in label_seq]

                input_seq_embed = input_traj_to_embeddings(sample,poi_gcn_embeddings)
                batch_seq_embeds.append(input_seq_embed)
                batch_seq_lens.append(len(input_seq))
                batch_input_seqs.append(input_seq)
                batch_seq_labels_poi.append(torch.LongTensor(label_seq))
                batch_seq_labels_time.append(torch.FloatTensor(label_seq_time))
                batch_seq_labels_lat.append(torch.FloatTensor(label_seq_lat))
                batch_seq_labels_lon.append(torch.FloatTensor(label_seq_lon))
                batch_seq_labels_week.append(torch.LongTensor(label_seq_week))
                batch_seq_labels_cat.append(torch.LongTensor(label_seq_cats))
                batch_seq_labels_categroy.append(torch.LongTensor(label_seq_category))

            # Pad seqs for batch training
            batch_padded = pad_sequence(batch_seq_embeds, batch_first=True, padding_value=-1)
            label_padded_poi = pad_sequence(batch_seq_labels_poi, batch_first=True, padding_value=-1)
            label_padded_time = pad_sequence(batch_seq_labels_time, batch_first=True, padding_value=-1)
            label_padded_lat = pad_sequence(batch_seq_labels_lat, batch_first=True, padding_value=-1)
            label_padded_lon = pad_sequence(batch_seq_labels_lon, batch_first=True, padding_value=-1)
            label_padded_week = pad_sequence(batch_seq_labels_week, batch_first=True, padding_value=-1)
            label_padded_cat = pad_sequence(batch_seq_labels_cat, batch_first=True, padding_value=-1)
            label_padded_categroy = pad_sequence(batch_seq_labels_categroy, batch_first=True, padding_value=-1)

            # Feedforward
            x = batch_padded.to(device=args.device, dtype=torch.float)
            y_poi = label_padded_poi.to(device=args.device, dtype=torch.long)
            y_time = label_padded_time.to(device=args.device, dtype=torch.float)
            y_lat = label_padded_lat.to(device=args.device, dtype=torch.float)
            y_lon = label_padded_lon.to(device=args.device, dtype=torch.float)
            y_week = label_padded_week.to(device=args.device, dtype=torch.long)
            y_cat = label_padded_cat.to(device=args.device, dtype=torch.long)
            y_categroy = label_padded_categroy.to(device=args.device, dtype=torch.long)
            # todo:主要还是为了预测下一个poi点,所以需要批判性分析loss_time , loss_cat +loss_lat,loss_lon ,loss_categroy , loss_week是否有必要
            y_pred_poi, y_pred_time, y_pred_lat, y_pred_lon, y_pred_cat, y_pred_categroy ,y_pred_week = seq_model(x, src_mask)

            loss_poi = criterion_poi(y_pred_poi.transpose(1, 2), y_poi)
            loss_time = criterion_time(torch.squeeze(y_pred_time), torch.squeeze(y_time))
            loss_lat = criterion_time(torch.squeeze(y_pred_lat), torch.squeeze(y_lat))
            loss_lon = criterion_time(torch.squeeze(y_pred_lon), torch.squeeze(y_lon))
            loss_week = criterion_week(y_pred_week.transpose(1, 2), y_week)
            loss_cat = criterion_cat(y_pred_cat.transpose(1, 2), y_cat)
            loss_categroy = criterion_categroy(y_pred_categroy.transpose(1, 2), y_categroy)

            # Final loss
            loss = loss_poi + loss_time * args.time_loss_weight + loss_cat +loss_lat + loss_lon +loss_categroy + loss_week
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

            # Performance measurement
            top1_acc = 0
            top5_acc = 0
            top10_acc = 0
            top20_acc = 0
            mAP20 = 0
            mrr = 0
            ndcg20 = 0
            recall20 = 0

            batch_label_pois = y_poi.detach().cpu().numpy()
            batch_pred_pois = y_pred_poi.detach().cpu().numpy()
            batch_pred_times = y_pred_time.detach().cpu().numpy()
            batch_pred_lats = y_pred_lat.detach().cpu().numpy()
            batch_pred_lons = y_pred_lon.detach().cpu().numpy()
            batch_pred_cats = y_pred_cat.detach().cpu().numpy()
            batch_pred_categroys = y_pred_categroy.detach().cpu().numpy()
            batch_pred_weeks = y_pred_week.detach().cpu().numpy()
            for label_pois, pred_pois, seq_len in zip(batch_label_pois, batch_pred_pois, batch_seq_lens):
                label_pois = label_pois[:seq_len]  # shape: (seq_len, )
                pred_pois = pred_pois[:seq_len, :]  # shape: (seq_len, num_poi)
                top1_acc += top_k_acc_last_timestep(label_pois, pred_pois, k=1)
                top5_acc += top_k_acc_last_timestep(label_pois, pred_pois, k=5)
                top10_acc += top_k_acc_last_timestep(label_pois, pred_pois, k=10)
                top20_acc += top_k_acc_last_timestep(label_pois, pred_pois, k=20)
                mAP20 += mAP_metric_last_timestep(label_pois, pred_pois, k=20)
                mrr += MRR_metric_last_timestep(label_pois, pred_pois)
                ndcg20 += NDCG_metric_last_timestep(label_pois, pred_pois, k=20)
                recall20 += recall_metric_last_timestep(label_pois, pred_pois, k=20)
            train_batches_top1_acc_list.append(top1_acc / len(batch_label_pois))
            train_batches_top5_acc_list.append(top5_acc / len(batch_label_pois))
            train_batches_top10_acc_list.append(top10_acc / len(batch_label_pois))
            train_batches_top20_acc_list.append(top20_acc / len(batch_label_pois))
            train_batches_mAP20_list.append(mAP20 / len(batch_label_pois))
            train_batches_mrr_list.append(mrr / len(batch_label_pois))
            train_batches_ndcg20_list.append(ndcg20 / len(batch_label_pois))
            train_batches_recall20_list.append(recall20 / len(batch_label_pois))
            train_batches_loss_list.append(loss.detach().cpu().numpy())
            train_batches_poi_loss_list.append(loss_poi.detach().cpu().numpy())
            train_batches_time_loss_list.append(loss_time.detach().cpu().numpy())
            train_batches_cat_loss_list.append(loss_cat.detach().cpu().numpy())
            train_batches_categroy_loss_list.append(loss_categroy.detach().cpu().numpy())
            train_batches_lat_loss_list.append(loss_lat.detach().cpu().numpy())
            train_batches_lon_loss_list.append(loss_lon.detach().cpu().numpy())
            train_batches_week_loss_list.append(loss_week.detach().cpu().numpy())

            # Report training progress
            if (b_idx % (args.batch * 5)) == 0:
                sample_idx = 0
                logging.info(f'Epoch:{epoch}, batch:{b_idx}, '
                             f'train_batch_loss:{loss.item():.2f}, '
                             f'train_batch_top1_acc:{top1_acc / len(batch_label_pois):.2f}, '
                             f'train_move_loss:{np.mean(train_batches_loss_list):.2f}\n'
                             f'train_move_poi_loss:{np.mean(train_batches_poi_loss_list):.2f}\n'
                             f'train_move_lat_loss:{np.mean(train_batches_lat_loss_list):.2f} \n'
                             f'train_move_lon_loss:{np.mean(train_batches_lon_loss_list):.2f} \n'
                             f'train_move_week_loss:{np.mean(train_batches_week_loss_list):.2f} \n'
                             f'train_move_cat_loss:{np.mean(train_batches_cat_loss_list):.2f} \n'
                             f'train_move_cat2_loss:{np.mean(train_batches_categroy_loss_list):.2f} \n'
                             f'train_move_time_loss:{np.mean(train_batches_time_loss_list):.2f}\n'
                             f'train_move_top1_acc:{np.mean(train_batches_top1_acc_list):.4f}\n'
                             f'train_move_top5_acc:{np.mean(train_batches_top5_acc_list):.4f}\n'
                             f'train_move_top10_acc:{np.mean(train_batches_top10_acc_list):.4f}\n'
                             f'train_move_top20_acc:{np.mean(train_batches_top20_acc_list):.4f}\n'
                             f'train_move_mAP20:{np.mean(train_batches_mAP20_list):.4f}\n'
                             f'train_move_MRR:{np.mean(train_batches_mrr_list):.4f}\n'
                             f'train_move_ndcg20:{np.mean(train_batches_ndcg20_list):.4f}\n'
                             f'train_move_recall20:{np.mean(train_batches_recall20_list):.4f}\n'
                             f'traj_id:{batch[sample_idx][0]}\n'
                             f'input_seq: {batch[sample_idx][1]}\n'
                             f'label_seq:{batch[sample_idx][2]}\n'
                             f'label_seq_poi:{[each[0] for each in batch[sample_idx][2]]}\n'
                             f'pred_seq_poi:{list(np.argmax(batch_pred_pois, axis=2)[sample_idx][:batch_seq_lens[sample_idx]])} \n'
                             f'label_seq_cat:{[poi_idx2cat_id_dict[each[0]] for each in batch[sample_idx][2]]}\n'
                             f'pred_seq_cat:{list(np.argmax(batch_pred_cats, axis=2)[sample_idx][:batch_seq_lens[sample_idx]])} \n'
                             f'label_seq_cat2:{[poi_idx2category_id_dict[each[0]] for each in batch[sample_idx][2]]}\n'
                             f'pred_seq_cat2:{list(np.argmax(batch_pred_categroys, axis=2)[sample_idx][:batch_seq_lens[sample_idx]])} \n'
                             f'label_seq_lat:{list(batch_seq_labels_lat[sample_idx].numpy()[:batch_seq_lens[sample_idx]])}\n'
                             f'pred_seq_lat:{list(np.squeeze(batch_pred_lats)[sample_idx][:batch_seq_lens[sample_idx]])} \n'
                             f'label_seq_lon:{list(batch_seq_labels_lon[sample_idx].numpy()[:batch_seq_lens[sample_idx]])}\n'
                             f'pred_seq_lat:{list(np.squeeze(batch_pred_lons)[sample_idx][:batch_seq_lens[sample_idx]])} \n'
                             f'label_seq_week:{list(batch_seq_labels_week[sample_idx].numpy()[:batch_seq_lens[sample_idx]])}\n'
                             f'pred_seq_week:{list(np.argmax(batch_pred_weeks, axis=2)[sample_idx][:batch_seq_lens[sample_idx]])} \n'
                             f'label_seq_time:{list(batch_seq_labels_time[sample_idx].numpy()[:batch_seq_lens[sample_idx]])}\n'
                             f'pred_seq_time:{list(np.squeeze(batch_pred_times)[sample_idx][:batch_seq_lens[sample_idx]])} \n' +
                             '=' * 100)

        # train end --------------------------------------------------------------------------------------------------------

        user_embed_model.eval()
        poi_embed_model.eval()
        poi_gcn_model.eval()
        time_embed_model.eval()
        lat_embed_model.eval()
        lon_embed_model.eval()
        week_embed_model.eval()
        cat_embed_model.eval()
        categroy_embed_model.eval()
        embed_fuse_model2.eval()
        align_layer.eval()
        seq_model.eval()
        multihead_attn.eval()
        layer_norm.eval()
        poi_fusion.eval()


        val_batches_top1_acc_list = []
        val_batches_top5_acc_list = []
        val_batches_top10_acc_list = []
        val_batches_top20_acc_list = []
        val_batches_mAP20_list = []
        val_batches_mrr_list = []
        val_batches_ndcg20_list = []
        val_batches_recall20_list = []
        val_batches_loss_list = []
        val_batches_poi_loss_list = []
        val_batches_time_loss_list = []
        val_batches_week_loss_list = []
        val_batches_cat_loss_list = []
        val_batches_lat_loss_list = []
        val_batches_lon_loss_list = []
        val_batches_categroy_loss_list = []
        src_mask = seq_model.generate_square_subsequent_mask(args.batch).to(args.device)
        for vb_idx, batch in enumerate(val_loader):
            if len(batch) != args.batch:
                src_mask = seq_model.generate_square_subsequent_mask(len(batch)).to(args.device)

            # For padding
            batch_input_seqs = []
            batch_seq_lens = []
            batch_seq_embeds = []
            batch_seq_labels_poi = []
            batch_seq_labels_time = []
            batch_seq_labels_lat = []
            batch_seq_labels_lon = []
            batch_seq_labels_week = []
            batch_seq_labels_cat = []
            batch_seq_labels_categroy = []

            poi_gcn_embeddings = poi_gcn_model(X, A)


            # Convert input seq to embeddings
            for sample in batch:
                input_seq = [each[0] for each in sample[1]]
                label_seq = [each[0] for each in sample[2]]
                label_seq_time = [each[1] for each in sample[2]]
                label_seq_lat = [np.radians(each[2]) for each in sample[2]]
                label_seq_lon = [np.radians(each[3]) for each in sample[2]]
                label_seq_week = [each[4] for each in sample[2]]
                label_seq_cats = [poi_idx2cat_id_dict[each] for each in label_seq]
                label_seq_category = [poi_idx2category_id_dict[each] for each in label_seq]
                input_seq_embed = input_traj_to_embeddings(sample,poi_gcn_embeddings)

                batch_seq_embeds.append(input_seq_embed)
                batch_seq_lens.append(len(input_seq))
                batch_input_seqs.append(input_seq)
                batch_seq_labels_poi.append(torch.LongTensor(label_seq))
                batch_seq_labels_time.append(torch.FloatTensor(label_seq_time))
                batch_seq_labels_lat.append(torch.FloatTensor(label_seq_lat))
                batch_seq_labels_lon.append(torch.FloatTensor(label_seq_lon))
                batch_seq_labels_week.append(torch.LongTensor(label_seq_week))
                batch_seq_labels_cat.append(torch.LongTensor(label_seq_cats))
                batch_seq_labels_categroy.append(torch.LongTensor(label_seq_category))

            # Pad seqs for batch training
            batch_padded = pad_sequence(batch_seq_embeds, batch_first=True, padding_value=-1)
            label_padded_poi = pad_sequence(batch_seq_labels_poi, batch_first=True, padding_value=-1)
            label_padded_time = pad_sequence(batch_seq_labels_time, batch_first=True, padding_value=-1)
            label_padded_lat = pad_sequence(batch_seq_labels_lat, batch_first=True, padding_value=-1)
            label_padded_lon = pad_sequence(batch_seq_labels_lon, batch_first=True, padding_value=-1)
            label_padded_week = pad_sequence(batch_seq_labels_week, batch_first=True, padding_value=-1)
            label_padded_cat = pad_sequence(batch_seq_labels_cat, batch_first=True, padding_value=-1)
            label_padded_categroy = pad_sequence(batch_seq_labels_categroy, batch_first=True, padding_value=-1)

            # Feedforward
            x = batch_padded.to(device=args.device, dtype=torch.float)
            y_poi = label_padded_poi.to(device=args.device, dtype=torch.long)
            y_time = label_padded_time.to(device=args.device, dtype=torch.float)
            y_lat = label_padded_lat.to(device=args.device, dtype=torch.float)
            y_lon = label_padded_lon.to(device=args.device, dtype=torch.float)
            y_week = label_padded_week.to(device=args.device, dtype=torch.long)
            y_cat = label_padded_cat.to(device=args.device, dtype=torch.long)
            y_categroy = label_padded_categroy.to(device=args.device, dtype=torch.long)
            y_pred_poi, y_pred_time, y_pred_lat, y_pred_lon, y_pred_cat, y_pred_categroy , y_pred_week = seq_model(x, src_mask)


            # Calculate loss
            loss_poi = criterion_poi(y_pred_poi.transpose(1, 2), y_poi)
            loss_time = criterion_time(torch.squeeze(y_pred_time), torch.squeeze(y_time))
            loss_lat = criterion_time(torch.squeeze(y_pred_lat), torch.squeeze(y_lat))
            loss_lon = criterion_time(torch.squeeze(y_pred_lon), torch.squeeze(y_lon))
            loss_week = criterion_week(y_pred_week.transpose(1, 2), y_week)
            loss_cat = criterion_cat(y_pred_cat.transpose(1, 2), y_cat)
            loss_categroy = criterion_categroy(y_pred_categroy.transpose(1, 2), y_categroy)
            loss = loss_poi + loss_time * args.time_loss_weight + loss_cat + loss_lat + loss_lon +loss_categroy + loss_week

            # Performance measurement
            top1_acc = 0
            top5_acc = 0
            top10_acc = 0
            top20_acc = 0
            mAP20 = 0
            mrr = 0
            ndcg20 = 0
            recall20 = 0
            batch_label_pois = y_poi.detach().cpu().numpy()
            batch_pred_pois = y_pred_poi.detach().cpu().numpy()
            batch_pred_times = y_pred_time.detach().cpu().numpy()
            batch_pred_lats = y_pred_lat.detach().cpu().numpy()
            batch_pred_lons = y_pred_lon.detach().cpu().numpy()
            batch_pred_cats = y_pred_cat.detach().cpu().numpy()
            batch_pred_categroys = y_pred_categroy.detach().cpu().numpy()
            batch_pred_weeks = y_pred_week.detach().cpu().numpy()
            for label_pois, pred_pois, seq_len in zip(batch_label_pois, batch_pred_pois, batch_seq_lens):
                label_pois = label_pois[:seq_len]  # shape: (seq_len, )
                pred_pois = pred_pois[:seq_len, :]  # shape: (seq_len, num_poi)
                top1_acc += top_k_acc_last_timestep(label_pois, pred_pois, k=1)
                top5_acc += top_k_acc_last_timestep(label_pois, pred_pois, k=5)
                top10_acc += top_k_acc_last_timestep(label_pois, pred_pois, k=10)
                top20_acc += top_k_acc_last_timestep(label_pois, pred_pois, k=20)
                mAP20 += mAP_metric_last_timestep(label_pois, pred_pois, k=20)
                mrr += MRR_metric_last_timestep(label_pois, pred_pois)
                ndcg20 += NDCG_metric_last_timestep(label_pois, pred_pois, k=20)
                recall20 += recall_metric_last_timestep(label_pois, pred_pois, k=20)
            val_batches_top1_acc_list.append(top1_acc / len(batch_label_pois))
            val_batches_top5_acc_list.append(top5_acc / len(batch_label_pois))
            val_batches_top10_acc_list.append(top10_acc / len(batch_label_pois))
            val_batches_top20_acc_list.append(top20_acc / len(batch_label_pois))
            val_batches_mAP20_list.append(mAP20 / len(batch_label_pois))
            val_batches_mrr_list.append(mrr / len(batch_label_pois))
            val_batches_ndcg20_list.append(ndcg20 / len(batch_label_pois))
            val_batches_recall20_list.append(recall20 / len(batch_label_pois))
            val_batches_loss_list.append(loss.detach().cpu().numpy())
            val_batches_poi_loss_list.append(loss_poi.detach().cpu().numpy())
            val_batches_time_loss_list.append(loss_time.detach().cpu().numpy())
            val_batches_cat_loss_list.append(loss_cat.detach().cpu().numpy())
            val_batches_categroy_loss_list.append(loss_categroy.detach().cpu().numpy())
            val_batches_lat_loss_list.append(loss_lat.detach().cpu().numpy())
            val_batches_lon_loss_list.append(loss_lon.detach().cpu().numpy())
            val_batches_week_loss_list.append(loss_week.detach().cpu().numpy())

            # Report validation progress
            if (vb_idx % (args.batch * 4)) == 0:
                sample_idx = 0
                logging.info(f'Epoch:{epoch}, batch:{vb_idx}, '
                             f'val_batch_loss:{loss.item():.2f}, '
                             f'val_batch_top1_acc:{top1_acc / len(batch_label_pois):.2f}, '
                             f'val_move_loss:{np.mean(val_batches_loss_list):.2f} \n'
                             f'val_move_poi_loss:{np.mean(val_batches_poi_loss_list):.2f} \n'
                             f'val_move_lat_loss:{np.mean(val_batches_lat_loss_list):.2f} \n'
                             f'val_move_lon_loss:{np.mean(val_batches_lon_loss_list):.2f} \n'
                             f'val_move_week_loss:{np.mean(val_batches_week_loss_list):.2f} \n'
                             f'val_move_cat_loss:{np.mean(val_batches_cat_loss_list):.2f} \n'
                             f'val_move_cat2_loss:{np.mean(val_batches_categroy_loss_list):.2f} \n'
                             f'val_move_time_loss:{np.mean(val_batches_time_loss_list):.2f} \n'
                             f'val_move_top1_acc:{np.mean(val_batches_top1_acc_list):.4f} \n'
                             f'val_move_top5_acc:{np.mean(val_batches_top5_acc_list):.4f} \n'
                             f'val_move_top10_acc:{np.mean(val_batches_top10_acc_list):.4f} \n'
                             f'val_move_top20_acc:{np.mean(val_batches_top20_acc_list):.4f} \n'
                             f'val_move_mAP20:{np.mean(val_batches_mAP20_list):.4f} \n'
                             f'val_move_MRR:{np.mean(val_batches_mrr_list):.4f} \n'
                             f'val_move_ndcg20:{np.mean(val_batches_ndcg20_list):.4f} \n'
                             f'val_move_recall20:{np.mean(val_batches_recall20_list):.4f} \n'
                             f'traj_id:{batch[sample_idx][0]}\n'
                             f'input_seq:{batch[sample_idx][1]}\n'
                             f'label_seq:{batch[sample_idx][2]}\n'
                             f'label_seq_poi:{[each[0] for each in batch[sample_idx][2]]}\n'
                             f'pred_seq_poi:{list(np.argmax(batch_pred_pois, axis=2)[sample_idx][:batch_seq_lens[sample_idx]])} \n'
                             f'label_seq_cat:{[poi_idx2cat_id_dict[each[0]] for each in batch[sample_idx][2]]}\n'
                             f'pred_seq_cat:{list(np.argmax(batch_pred_cats, axis=2)[sample_idx][:batch_seq_lens[sample_idx]])} \n'
                             f'label_seq_cat2:{[poi_idx2category_id_dict[each[0]] for each in batch[sample_idx][2]]}\n'
                             f'pred_seq_cat2:{list(np.argmax(batch_pred_categroys, axis=2)[sample_idx][:batch_seq_lens[sample_idx]])} \n'
                             f'label_seq_lat:{list(batch_seq_labels_lat[sample_idx].numpy()[:batch_seq_lens[sample_idx]])}\n'
                             f'pred_seq_lat:{list(np.squeeze(batch_pred_lats)[sample_idx][:batch_seq_lens[sample_idx]])} \n'
                             f'label_seq_lon:{list(batch_seq_labels_lon[sample_idx].numpy()[:batch_seq_lens[sample_idx]])}\n'
                             f'pred_seq_lat:{list(np.squeeze(batch_pred_lons)[sample_idx][:batch_seq_lens[sample_idx]])} \n'
                             f'label_seq_week:{list(batch_seq_labels_week[sample_idx].numpy()[:batch_seq_lens[sample_idx]])}\n'
                             f'pred_seq_week:{list(np.argmax(batch_pred_weeks, axis=2)[sample_idx][:batch_seq_lens[sample_idx]])} \n'
                             f'label_seq_time:{list(batch_seq_labels_time[sample_idx].numpy()[:batch_seq_lens[sample_idx]])}\n'
                             f'pred_seq_time:{list(np.squeeze(batch_pred_times)[sample_idx][:batch_seq_lens[sample_idx]])} \n' +
                             '=' * 100)
        # valid end --------------------------------------------------------------------------------------------------------

        # Calculate epoch metrics
        epoch_train_top1_acc = np.mean(train_batches_top1_acc_list)
        epoch_train_top5_acc = np.mean(train_batches_top5_acc_list)
        epoch_train_top10_acc = np.mean(train_batches_top10_acc_list)
        epoch_train_top20_acc = np.mean(train_batches_top20_acc_list)
        epoch_train_mAP20 = np.mean(train_batches_mAP20_list)
        epoch_train_mrr = np.mean(train_batches_mrr_list)
        epoch_train_ndcg20 = np.mean(train_batches_ndcg20_list)
        epoch_train_recall20 = np.mean(train_batches_recall20_list)
        epoch_train_loss = np.mean(train_batches_loss_list)
        epoch_train_poi_loss = np.mean(train_batches_poi_loss_list)
        epoch_train_time_loss = np.mean(train_batches_time_loss_list)
        epoch_train_cat_loss = np.mean(train_batches_cat_loss_list)
        epoch_val_top1_acc = np.mean(val_batches_top1_acc_list)
        epoch_val_top5_acc = np.mean(val_batches_top5_acc_list)
        epoch_val_top10_acc = np.mean(val_batches_top10_acc_list)
        epoch_val_top20_acc = np.mean(val_batches_top20_acc_list)
        epoch_val_mAP20 = np.mean(val_batches_mAP20_list)
        epoch_val_mrr = np.mean(val_batches_mrr_list)
        epoch_val_ndcg20 = np.mean(val_batches_ndcg20_list)
        epoch_val_recall20 = np.mean(val_batches_recall20_list)
        epoch_val_loss = np.mean(val_batches_loss_list)
        epoch_val_poi_loss = np.mean(val_batches_poi_loss_list)
        epoch_val_time_loss = np.mean(val_batches_time_loss_list)
        epoch_val_cat_loss = np.mean(val_batches_cat_loss_list)

        # Save metrics to list
        train_epochs_loss_list.append(epoch_train_loss)
        train_epochs_poi_loss_list.append(epoch_train_poi_loss)
        train_epochs_time_loss_list.append(epoch_train_time_loss)
        train_epochs_cat_loss_list.append(epoch_train_cat_loss)
        train_epochs_top1_acc_list.append(epoch_train_top1_acc)
        train_epochs_top5_acc_list.append(epoch_train_top5_acc)
        train_epochs_top10_acc_list.append(epoch_train_top10_acc)
        train_epochs_top20_acc_list.append(epoch_train_top20_acc)
        train_epochs_mAP20_list.append(epoch_train_mAP20)
        train_epochs_mrr_list.append(epoch_train_mrr)
        train_epochs_ndcg20_list.append(epoch_train_ndcg20)
        train_epochs_recall20_list.append(epoch_train_recall20)

        val_epochs_loss_list.append(epoch_val_loss)
        val_epochs_poi_loss_list.append(epoch_val_poi_loss)
        val_epochs_time_loss_list.append(epoch_val_time_loss)
        val_epochs_cat_loss_list.append(epoch_val_cat_loss)
        val_epochs_top1_acc_list.append(epoch_val_top1_acc)
        val_epochs_top5_acc_list.append(epoch_val_top5_acc)
        val_epochs_top10_acc_list.append(epoch_val_top10_acc)
        val_epochs_top20_acc_list.append(epoch_val_top20_acc)
        val_epochs_mAP20_list.append(epoch_val_mAP20)
        val_epochs_mrr_list.append(epoch_val_mrr)
        val_epochs_ndcg20_list.append(epoch_val_ndcg20)
        val_epochs_recall20_list.append(epoch_val_recall20)

        # Monitor loss and score
        # 1. 基础指标评分
        base_metrics = {
            'top1_acc': epoch_val_top1_acc,
            'top5_acc': epoch_val_top5_acc,
            'top10_acc': epoch_val_top10_acc,
            'top20_acc': epoch_val_top20_acc,
            'mrr': epoch_val_mrr,
            'ndcg20': epoch_val_ndcg20,
            'recall20': epoch_val_recall20
        }
        
        # 2. 计算综合评分
        weights = {
            'top1_acc': 0.3,    # 最相关预测的权重
            'top5_acc': 0.2,    # 前5个预测的权重
            'mrr': 0.2,         # 排序质量的权重
            'ndcg20': 0.2,      # 排序准确性的权重
            'recall20': 0.1     # 覆盖率的权重
        }
        
        monitor_score = sum(base_metrics[k] * v for k, v in weights.items())
        
        # 3. 计算预测稳定性 
        stability_score = calculate_prediction_stability(
            val_batches_top1_acc_list,
            val_batches_top5_acc_list,
            val_batches_mrr_list
        )
        
        # 4. 计算损失平衡性
        loss_balance = calculate_loss_balance(
            epoch_val_poi_loss,
            epoch_val_time_loss,
            epoch_val_cat_loss
        )
        
        # 5. 最终监控指标
        monitor_loss = epoch_val_loss
        monitor_score = monitor_score * (1 + stability_score) * (1 + loss_balance)
        
        # 记录详细指标
        logging.info(f"Detailed monitoring metrics:")
        logging.info(f"Base metrics: {base_metrics}")
        logging.info(f"Stability score: {stability_score:.4f}")
        logging.info(f"Loss balance: {loss_balance:.4f}")
        logging.info(f"Final monitor score: {monitor_score:.4f}")
        
        # Learning rate scheduler
        lr_scheduler.step(monitor_loss)

        # Print epoch results
        logging.info(f"Epoch {epoch}/{args.epochs}\n"
                     f"train_loss:{epoch_train_loss:.4f}, "
                     f"train_poi_loss:{epoch_train_poi_loss:.4f}, "
                     f"train_time_loss:{epoch_train_time_loss:.4f}, "
                     f"train_cat_loss:{epoch_train_cat_loss:.4f}, "
                     f"train_top1_acc:{epoch_train_top1_acc:.4f}, "
                     f"train_top5_acc:{epoch_train_top5_acc:.4f}, "
                     f"train_top10_acc:{epoch_train_top10_acc:.4f}, "
                     f"train_top20_acc:{epoch_train_top20_acc:.4f}, "
                     f"train_mAP20:{epoch_train_mAP20:.4f}, "
                     f"train_mrr:{epoch_train_mrr:.4f}\n"
                     f"train_ndcg20:{epoch_train_ndcg20:.4f}, "
                     f"train_recall20:{epoch_train_recall20:.4f}\n"
                     f"val_loss: {epoch_val_loss:.4f}, "
                     f"val_poi_loss: {epoch_val_poi_loss:.4f}, "
                     f"val_time_loss: {epoch_val_time_loss:.4f}, "
                     f"val_cat_loss: {epoch_val_cat_loss:.4f}, "
                     f"val_top1_acc:{epoch_val_top1_acc:.4f}, "
                     f"val_top5_acc:{epoch_val_top5_acc:.4f}, "
                     f"val_top10_acc:{epoch_val_top10_acc:.4f}, "
                     f"val_top20_acc:{epoch_val_top20_acc:.4f}, "
                     f"val_mAP20:{epoch_val_mAP20:.4f}, "
                     f"val_mrr:{epoch_val_mrr:.4f},"
                     f"val_ndcg20:{epoch_val_ndcg20:.4f}, "
                     f"val_recall20:{epoch_val_recall20:.4f}"
                     )

        # Save model state dict
        if args.save_weights:
            state_dict = {
                'epoch': epoch,
                'epoch_train_metrics': {
                    'epoch_train_loss': epoch_train_loss,
                    'epoch_train_poi_loss': epoch_train_poi_loss,
                    'epoch_train_time_loss': epoch_train_time_loss,
                    'epoch_train_cat_loss': epoch_train_cat_loss,
                    'epoch_train_top1_acc': epoch_train_top1_acc,
                    'epoch_train_top5_acc': epoch_train_top5_acc,
                    'epoch_train_top10_acc': epoch_train_top10_acc,
                    'epoch_train_top20_acc': epoch_train_top20_acc,
                    'epoch_train_mAP20': epoch_train_mAP20,
                    'epoch_train_mrr': epoch_train_mrr,
                    'epoch_train_ncdg20': epoch_val_ndcg20,
                    'epoch_train_recall20': epoch_val_recall20
                },
                'epoch_val_metrics': {
                    'epoch_val_loss': epoch_val_loss,
                    'epoch_val_poi_loss': epoch_val_poi_loss,
                    'epoch_val_time_loss': epoch_val_time_loss,
                    'epoch_val_cat_loss': epoch_val_cat_loss,
                    'epoch_val_top1_acc': epoch_val_top1_acc,
                    'epoch_val_top5_acc': epoch_val_top5_acc,
                    'epoch_val_top10_acc': epoch_val_top10_acc,
                    'epoch_val_top20_acc': epoch_val_top20_acc,
                    'epoch_val_mAP20': epoch_val_mAP20,
                    'epoch_val_mrr': epoch_val_mrr,
                    'epoch_val_ncdg20': epoch_val_ndcg20,
                    'epoch_val_recall20': epoch_val_recall20
                }
            }
            model_save_dir = os.path.join(args.save_dir, 'checkpoints')
            # Save best val score epoch
            if monitor_score >= max_val_score:
                if not os.path.exists(model_save_dir): os.makedirs(model_save_dir)
                torch.save(state_dict, rf"{model_save_dir}/best_epoch.state.pt")
                with open(rf"{model_save_dir}/best_epoch.txt", 'w') as f:
                    print(state_dict['epoch_val_metrics'], file=f)
                max_val_score = monitor_score

        # Save train/val metrics for plotting purpose
        with open(os.path.join(args.save_dir, 'metrics-train.txt'), "w") as f:
            print(f'train_epochs_loss_list={[float(f"{each:.4f}") for each in train_epochs_loss_list]}', file=f)
            print(f'train_epochs_poi_loss_list={[float(f"{each:.4f}") for each in train_epochs_poi_loss_list]}', file=f)
            print(f'train_epochs_time_loss_list={[float(f"{each:.4f}") for each in train_epochs_time_loss_list]}',
                  file=f)
            print(f'train_epochs_cat_loss_list={[float(f"{each:.4f}") for each in train_epochs_cat_loss_list]}', file=f)
            print(f'train_epochs_top1_acc_list={[float(f"{each:.4f}") for each in train_epochs_top1_acc_list]}', file=f)
            print(f'train_epochs_top5_acc_list={[float(f"{each:.4f}") for each in train_epochs_top5_acc_list]}', file=f)
            print(f'train_epochs_top10_acc_list={[float(f"{each:.4f}") for each in train_epochs_top10_acc_list]}',
                  file=f)
            print(f'train_epochs_top20_acc_list={[float(f"{each:.4f}") for each in train_epochs_top20_acc_list]}',
                  file=f)
            print(f'train_epochs_mAP20_list={[float(f"{each:.4f}") for each in train_epochs_mAP20_list]}', file=f)
            print(f'train_epochs_mrr_list={[float(f"{each:.4f}") for each in train_epochs_mrr_list]}', file=f)
            print(f'train_epochs_ncdg20_list={[float(f"{each:.4f}") for each in train_epochs_ndcg20_list]}', file=f)
            print(f'train_epochs_recall20_list={[float(f"{each:.4f}") for each in train_epochs_recall20_list]}', file=f)
        with open(os.path.join(args.save_dir, 'metrics-val.txt'), "w") as f:
            print(f'val_epochs_loss_list={[float(f"{each:.4f}") for each in val_epochs_loss_list]}', file=f)
            print(f'val_epochs_poi_loss_list={[float(f"{each:.4f}") for each in val_epochs_poi_loss_list]}', file=f)
            print(f'val_epochs_time_loss_list={[float(f"{each:.4f}") for each in val_epochs_time_loss_list]}', file=f)
            print(f'val_epochs_cat_loss_list={[float(f"{each:.4f}") for each in val_epochs_cat_loss_list]}', file=f)
            print(f'val_epochs_top1_acc_list={[float(f"{each:.4f}") for each in val_epochs_top1_acc_list]}', file=f)
            print(f'val_epochs_top5_acc_list={[float(f"{each:.4f}") for each in val_epochs_top5_acc_list]}', file=f)
            print(f'val_epochs_top10_acc_list={[float(f"{each:.4f}") for each in val_epochs_top10_acc_list]}', file=f)
            print(f'val_epochs_top20_acc_list={[float(f"{each:.4f}") for each in val_epochs_top20_acc_list]}', file=f)
            print(f'val_epochs_mAP20_list={[float(f"{each:.4f}") for each in val_epochs_mAP20_list]}', file=f)
            print(f'val_epochs_mrr_list={[float(f"{each:.4f}") for each in val_epochs_mrr_list]}', file=f)
            print(f'val_epochs_ncdg20_list={[float(f"{each:.4f}") for each in val_epochs_ndcg20_list]}', file=f)
            print(f'val_epochs_recall20_list={[float(f"{each:.4f}") for each in val_epochs_recall20_list]}', file=f)



if __name__ == '__main__':
    args = parameter_parser()
    # The name of node features in NYC/graph_X.csv
    args.feature1 = 'checkin_cnt'
    args.feature2 = 'poi_catid'
    args.feature3 = 'latitude'
    args.feature4 = 'longitude'
    train(args)

