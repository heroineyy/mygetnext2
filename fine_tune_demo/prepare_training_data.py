import json
from pathlib import Path


def format_training_sample(sample, metadata):
    """将单个样本转换为训练格式"""
    input_data = sample['input']
    output_data = sample['output']
    
    # 格式化历史记录
    history_str = "\n".join([
        f"- Time: {h['timestamp']}, Location: {h['place_name']} ({h['place_type']}), " \
        f"Weekday: {h['day_of_week']}, Time of day: {h['time_of_day']:.2f}"
        for h in input_data['history']
    ])
    
    # 构建提示
    prompt = f"""User ID: {input_data['user_id']}
Trajectory ID: {input_data['trajectory_id']}

Historical visits: 
{history_str}

Current context:
- Date: {input_data['context']['date']}
- Weekday: {input_data['context']['day_of_week']}
- Time of day: {input_data['context']['time_of_day']:.2f}

Please predict the user's next visit location."""
    
    # 构建完成
    completion = json.dumps(output_data['next_location'], ensure_ascii=False)
    
    return {
        "prompt": prompt,
        "completion": completion
    }

def prepare_training_data():
    # 加载metadata
    with open('fine_tune_data/dataset_metadata.json', 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    # 加载处理后的数据
    with open('fine_tune_data/processed_data.json', 'r', encoding='utf-8') as f:
        processed_data = json.load(f)
    
    # 准备训练数据
    training_data = []
    for sample in processed_data:
        training_sample = format_training_sample(sample, metadata)
        training_data.append(training_sample)
    
    # 保存训练数据为Ollama格式
    with open('fine_tune_data/training_data.jsonl', 'w', encoding='utf-8') as f:
        for item in training_data:
            # Ollama期望的格式是每行一个JSON对象，包含prompt和completion
            f.write(json.dumps({
                "prompt": item["prompt"],
                "completion": item["completion"]
            }, ensure_ascii=False) + '\n')
    
    print(f"Generated {len(training_data)} training samples")

if __name__ == "__main__":
    prepare_training_data() 