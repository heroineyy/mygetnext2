""" Build the user-agnostic global trajectory flow map from the sequence data """
import os
import math
import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.cluster import KMeans

import itertools



def build_global_User_graph(df, exclude_poi=None):
    G = nx.Graph()  # 使用无向图
    pois = list(set(df['POI_id'].to_list()))
    if exclude_poi in pois:
        pois.remove(exclude_poi)
    loop = tqdm(pois)
    df['local_time'] = pd.to_datetime(df['local_time'])

    for poi_id in loop:
        poi_df = df[df['POI_id'] == poi_id]
        users = poi_df['user_id'].unique()  # 获取访问该 POI 的所有用户

        # 为每个用户添加节点
        for user_id in users:
            if user_id not in G.nodes():
                G.add_node(user_id, checkin_cnt=1)
            else:
                G.nodes[user_id]['checkin_cnt'] += 1

        # 添加边
        for user_a, user_b in itertools.combinations(users, 2):
            if G.has_edge(user_a, user_b):
                G.edges[user_a, user_b]['weight'] += 1
            else:
                G.add_edge(user_a, user_b, weight=1)

    return G


def save_graph_to_csv(G, dst_dir):
    # Save adj matrix
    nodelist = G.nodes()
    A = nx.adjacency_matrix(G, nodelist=nodelist, weight='weight')
    np.savetxt(os.path.join(dst_dir, 'user_graph_adj.csv'), A.todense(), delimiter=',')

    # Save nodes list
    nodes_data = list(G.nodes.data())
    with open(os.path.join(dst_dir, 'user_graph_X.csv'), 'w') as f:
        print('user_id,checkin_cnt', file=f)
        for each in nodes_data:
            usr_id = each[0]
            checkin_cnt = each[1]['checkin_cnt']
            print(f'{usr_id},{checkin_cnt}', file=f)

if __name__ == '__main__':
    dst_dir = r'dataset/NYC/'
    dataset_name = 'NYC'

    # Build POI checkin trajectory graph
    train_df = pd.read_csv(os.path.join(dst_dir, f'{dataset_name}_train.csv'))


    print('Build global POI checkin graph -----------------------------------')

    G = build_global_User_graph(train_df)
    save_graph_to_csv(G, dst_dir=dst_dir)
    print("创建完成")



