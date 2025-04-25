""" Build the user-agnostic global trajectory flow map from the sequence data """
import os
import math
import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.cluster import KMeans

def augmented_data(raw_df):
    # # 聚类地理位置
    kmeans = KMeans(n_clusters=60, random_state=42)
    raw_df['location_cluster'] = kmeans.fit_predict(raw_df[['latitude', 'longitude']])
    # 转换时间格式并提取时间段
    raw_df['local_time'] = pd.to_datetime(raw_df['local_time'], errors='coerce')  # 将无效时间转换为 NaT
    raw_df = raw_df.dropna(subset=['local_time'])  # 去掉时间无效的行
    raw_df['hour'] = raw_df['local_time'].dt.hour  # 提取小时信息
    raw_df['weekday'] = raw_df['local_time'].dt.weekday
    raw_df['time_period'] = raw_df['hour'].apply(
        lambda x: '0-5' if 0 <= x < 6 else ('6-11' if 6 <= x < 12 else ('12-17' if 12 <= x < 18 else '18-23'))
    )
    return raw_df

def build_global_POI_checkin_graph(df, exclude_user=None):
    G = nx.DiGraph()
    users = list(set(df['user_id'].to_list()))
    if exclude_user in users:
        users.remove(exclude_user)
    loop = tqdm(users)
    df['local_time'] = pd.to_datetime(df['local_time'])
    for user_id in loop:
        user_df = df[df['user_id'] == user_id]

        # 为每个签到记录添加 POI 节点
        for i, row in user_df.iterrows():
            node = row['POI_id'] # 获取 POI 节点ID
            if node not in G.nodes():
                G.add_node(node,
                           checkin_cnt=1,  # 初始化签到次数为 1
                           poi_catid=row['POI_catid'],
                           poi_catid_code=row['POI_catid_code'],
                           poi_catname=row['POI_catname'],
                           latitude=row['latitude'],  # 纬度
                           longitude=row['longitude'],  # 经度
                           poi_category=row['POI_category'])
            else:
                G.nodes[node]['checkin_cnt'] += 1

        previous_poi_id = -1
        previous_traj_id = 0
        for i, row in user_df.iterrows():
            poi_id = row['POI_id']
            traj_id = row['trajectory_id']
            lat = row['latitude']  # 当前 POI 纬度
            lon = row['longitude']  # 当前 POI 经度
            # 如果是签到序列的起点或者轨迹不同，不添加边
            if previous_poi_id == -1 or (previous_traj_id != traj_id):
                previous_poi_id = poi_id
                previous_traj_id = traj_id
                previous_lat = lat
                previous_lon = lon
                continue

            # 计算 POI 之间的地理距离（Haversine 公式）
            a = (math.sin(math.radians(lon / 2 - previous_lon / 2))) ** 2
            b = math.cos(lon * math.pi / 180) * math.cos(previous_lon * math.pi / 180) * (
                math.sin((lat / 2 - previous_lat / 2) * math.pi / 180)) ** 2
            L = 2 * 6371.393 * math.asin((a + b) ** 0.5)  # 地理距离 L，单位为公里

            # 根据距离 L 计算地理奖励
            if L < 1:
                Lbonus = 1  # 如果 POI 之间的距离小于1公里，给予距离奖励1
            else:
                Lbonus = 0.76 / math.tanh(L)  # 否则根据距离 L 计算 tanh 奖励


            # 如果 G 中已有从 previous_poi_id 到 poi_id 的边，则累加权重
            if G.has_edge(previous_poi_id, poi_id):
                G.edges[previous_poi_id, poi_id]['geo_weight'] += Lbonus  # 更新 G 中边的权重
            else:
                # 如果 G 中没有这条边，则创建新的边并设置初始权重
                G.add_edge(previous_poi_id, poi_id, geo_weight=Lbonus)  # 添加地理距离权重的边

            # 更新前一个 POI 的信息
            previous_poi_id = poi_id
            previous_traj_id = traj_id
            previous_lat = lat
            previous_lon = lon

    return G


def save_graph_to_csv(G, dst_dir):
    # Save graph to an adj matrix file and a nodes file
    # Adj matrix file: edge from row_idx to col_idx with weight; Rows and columns are ordered according to nodes file.
    # Nodes file: node_name/poi_id, node features (category, location); Same node order with adj matrix.

    # Save adj matrix
    nodelist = G.nodes()
    A_geo = nx.adjacency_matrix(G, nodelist=nodelist,weight='geo_weight')
    # A_time = nx.adjacency_matrix(G, nodelist=nodelist, weight='time_weight')
    np.savetxt(os.path.join(dst_dir, 'geo_graph_adj.csv'), A_geo.todense(), delimiter=',')
    # np.savetxt(os.path.join(dst_dir, 'time_graph_adj2.csv'), A_time.todense(), delimiter=',')

    # Save nodes list
    nodes_data = list(G.nodes.data())
    with open(os.path.join(dst_dir, 'geo_graph_X.csv'), 'w') as f:
        print('node_name/poi_id,checkin_cnt,poi_catid,poi_catid_code,poi_catname,latitude,longitude,POI_category', file=f)
        for each in nodes_data:
            node_name = each[0]
            checkin_cnt = each[1]['checkin_cnt']
            poi_catid = each[1]['poi_catid']
            poi_catid_code = each[1]['poi_catid_code']
            poi_catname = each[1]['poi_catname']
            latitude = each[1]['latitude']
            longitude = each[1]['longitude']
            poi_category = each[1]['poi_category']
            print(f'{node_name},{checkin_cnt},'
                  f'{poi_catid},{poi_catid_code},{poi_catname},{latitude},{longitude},'
                  f'{poi_category}', file=f)
if __name__ == '__main__':
    dst_dir = r'dataset/NYC'
    dataset_name = 'NYC'

    # Build POI checkin trajectory graph
    train_df = pd.read_csv(os.path.join(dst_dir, f'{dataset_name}_train_with_categories.csv'))

    # train_df = augmented_data(train_df)

    print('Build global POI checkin graph -----------------------------------')

    G = build_global_POI_checkin_graph(train_df)
    save_graph_to_csv(G, dst_dir=dst_dir)
    print("创建完成")



