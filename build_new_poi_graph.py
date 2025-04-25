""" Build the user-agnostic global trajectory flow map from the sequence data """
import os
import math
import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.cluster import KMeans

import itertools



def build_global_new_poi_graph(df, exclude_user=None):
    G = nx.DiGraph()
    users = list(set(df['user_id'].to_list()))
    if exclude_user in users: users.remove(exclude_user)
    loop = tqdm(users)
    for user_id in loop:
        user_df = df[df['user_id'] == user_id]

        # Add node (POI)
        for i, row in user_df.iterrows():
            node = row['POI_category']
            if node not in G.nodes():
                G.add_node(row['POI_category'],
                           checkin_cnt=1
                           )
            else:
                G.nodes[node]['checkin_cnt'] += 1

        # Add edges (Check-in seq)
        previous_poi_id = 0
        previous_traj_id = 0
        for i, row in user_df.iterrows():
            poi_id = row['POI_category']
            traj_id = row['trajectory_id']
            # No edge for the begin of the seq or different traj
            if (previous_poi_id == 0) or (previous_traj_id != traj_id):
                previous_poi_id = poi_id
                previous_traj_id = traj_id
                continue

            # Add edges
            if G.has_edge(previous_poi_id, poi_id):
                G.edges[previous_poi_id, poi_id]['weight'] += 1
            else:  # Add new edge
                G.add_edge(previous_poi_id, poi_id, weight=1)
            previous_traj_id = traj_id
            previous_poi_id = poi_id

    return G

def save_graph_to_csv(G, dst_dir):
    # Save adj matrix
    nodelist = G.nodes()
    A = nx.adjacency_matrix(G, nodelist=nodelist, weight='weight')
    np.savetxt(os.path.join(dst_dir, 'new_poi_graph_adj.csv'), A.todense(), delimiter=',')

    # Save nodes list
    nodes_data = list(G.nodes.data())
    with open(os.path.join(dst_dir, 'new_poi_graph_X.csv'), 'w') as f:
        print('poi_category,checkin_cnt', file=f)
        for each in nodes_data:
            poi_category = each[0]
            checkin_cnt = each[1]['checkin_cnt']
            print(f'{poi_category},{checkin_cnt}', file=f)

if __name__ == '__main__':
    dst_dir = r'dataset/NYC'
    dataset_name = 'NYC'

    # Build POI checkin trajectory graph
    train_df = pd.read_csv(os.path.join(dst_dir, f'{dataset_name}_train.csv'))


    print('Build global POI checkin graph -----------------------------------')

    G = build_global_new_poi_graph(train_df)
    save_graph_to_csv(G, dst_dir=dst_dir)
    print("创建完成")



