import numpy as np
import pandas as pd


def load_graph_adj_mtx(path):
    """A.shape: (num_node, num_node), edge from row_index to col_index with weight"""
    A = np.loadtxt(path, delimiter=',')
    return A


def load_graph_node_features(path, flag, feature1='checkin_cnt', feature2='poi_catid_code',
                             feature3='latitude'):
    """X.shape: (num_node, 4), four features: checkin cnt, poi cat, latitude, longitude"""
    df = pd.read_csv(path)
    if flag == 1:
        rlt_df = df[[feature1, feature2,feature3]]
    elif flag == 2:
        rlt_df = df[[feature1]]
    else:
        rlt_df = df[[feature1, feature2]]
        # todo 对rlt_df的第一列进行归一化(假设从0开始),为什么要归一化,主要是考虑第一列是签到次数,第2列到769列都是第0列的编码值,为了防止输入给模型误认为第一列权重很高
    return rlt_df.to_numpy()
