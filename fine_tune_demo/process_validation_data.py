import json
import pandas as pd
from datetime import datetime

def process_validation_data():
    """处理验证数据集"""
    # 读取验证集CSV文件
    df = pd.read_csv('../dataset/NYC/NYC_val_with_categories.csv')
    
    # 加载训练集生成的metadata
    with open('fine_tune_data/dataset_metadata.json', 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    # 获取有效的地点类型和名称
    valid_place_types = set(metadata['valid_place_types'])
    valid_place_names = set(metadata['valid_place_names'])
    
    # 按用户和轨迹ID分组
    grouped = df.groupby(['user_id', 'trajectory_id'])
    
    validation_samples = []
    for (user_id, trajectory_id), group in grouped:
        # 按时间排序
        group = group.sort_values('UTC_time')
        
        # 构建完整轨迹
        trajectory = []
        for _, row in group.iterrows():
            # 验证地点名称和类型是否在训练集的字典中
            if row['POI_catname'] not in valid_place_names or \
               row['POI_category'] not in valid_place_types:
                continue
                
            trajectory.append({
                'timestamp': row['UTC_time'],
                'longitude': float(row['longitude']),
                'latitude': float(row['latitude']),
                'place_name': row['POI_catname'],
                'place_type': row['POI_category'],
                'day_of_week': int(row['day_of_week']),
                'time_of_day': float(row['norm_in_day_time'])
            })
        
        # 如果轨迹太短（少于2个点），跳过
        if len(trajectory) < 2:
            continue
            
        # 获取最后一个点的时间信息
        last_point = trajectory[-1]
        last_row = group.iloc[-1]
        
        # 构建验证样本
        sample = {
            'input': {
                'user_id': str(user_id),
                'trajectory_id': trajectory_id,
                'history': trajectory[:-1],
                'context': {
                    'day_of_week': int(last_row['day_of_week']),
                    'time_of_day': float(last_row['norm_in_day_time']),
                    'date': last_point['timestamp'].split('T')[0]
                }
            },
            'output': {
                'next_location': {
                    'place_name': trajectory[-1]['place_name'],
                    'place_type': trajectory[-1]['place_type']
                }
            }
        }
        
        validation_samples.append(sample)
    
    # 保存处理后的验证数据
    with open('fine_tune_data/validation_data.json', 'w', encoding='utf-8') as f:
        json.dump(validation_samples, f, ensure_ascii=False, indent=2)
    
    print(f"Processed {len(validation_samples)} validation samples")

if __name__ == "__main__":
    process_validation_data() 