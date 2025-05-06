import pandas as pd
import numpy as np
from datetime import datetime
import json
from collections import defaultdict

# 输入：
# dataset/NYC/NYC_train_with_categories.csv（原始数据）
# 输出：
# processed_data.json（处理后的结构化数据）
# dataset_metadata.json（数据集的元数据，包含有效的地点类型和名称）


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def process_nyc_data(input_file, output_file):
    # 读取数据
    df = pd.read_csv(input_file)
    
    # 构建地点和类别的映射表
    place_mapping = {
        'place_names': sorted(df['POI_catname'].unique().tolist()),
        'place_types': sorted(df['POI_category'].unique().tolist())
    }
    
    # 保存映射表到单独的metadata文件
    metadata = {
        'valid_place_names': place_mapping['place_names'],
        'valid_place_types': place_mapping['place_types'],
        'total_samples': 0,
        'total_users': int(len(df['user_id'].unique())),
        'total_trajectories': int(len(df['trajectory_id'].unique()))
    }
    
    with open('fine_tune_data/dataset_metadata.json', 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
    
    # 按用户和轨迹ID分组
    grouped = df.groupby(['user_id', 'trajectory_id'])
    
    processed_data = []
    
    for (user_id, trajectory_id), group in grouped:
        # 按时间排序
        group = group.sort_values('UTC_time')
        
        # 构建完整轨迹
        trajectory = []
        for _, row in group.iterrows():
            # 验证地点名称和类型是否在映射表中
            if row['POI_catname'] not in place_mapping['place_names'] or \
               row['POI_category'] not in place_mapping['place_types']:
                continue
                
            trajectory.append({
                'timestamp': row['UTC_time'],
                'longitude': float(row['longitude']),
                'latitude': float(row['latitude']),
                'place_name': row['POI_catname'],
                'place_type': row['POI_category'],
                'day_of_week': int(row['day_of_week']),  # 添加星期几信息
                'time_of_day': float(row['norm_in_day_time'])  # 添加一天中的时间信息
            })
        
        # 如果轨迹太短（少于2个点），跳过
        if len(trajectory) < 2:
            continue
            
        # 获取最后一个点的时间信息
        last_point = trajectory[-1]
        last_row = group.iloc[-1]  # 获取原始数据中的最后一行
        
        # 为整条轨迹创建一个训练样本
        sample = {
            'input': {
                'user_id': str(user_id),
                'trajectory_id': trajectory_id,
                'history': trajectory[:-1],  # 之前的历史轨迹点
                'context': {
                    'day_of_week': int(last_row['day_of_week']),  # 直接使用原始数据
                    'time_of_day': float(last_row['norm_in_day_time']),  # 直接使用原始数据
                    'date': last_point['timestamp'].split('T')[0]  # 只提取日期部分
                }
            },
            'output': {
                'next_location': {
                    'place_name': trajectory[-1]['place_name'],
                    'place_type': trajectory[-1]['place_type']
                }
            }
        }
        
        processed_data.append(sample)
    
    # 更新metadata中的样本总数
    metadata['total_samples'] = len(processed_data)
    with open('fine_tune_data/dataset_metadata.json', 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
    
    # 保存处理后的数据
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)

if __name__ == "__main__":
    input_file = "../dataset/NYC/NYC_train_with_categories.csv"
    output_file = "fine_tune_data/processed_data.json"
    process_nyc_data(input_file, output_file) 