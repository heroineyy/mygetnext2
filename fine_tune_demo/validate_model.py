import json
import subprocess
import random
from typing import Dict, List

def load_validation_data():
    """加载处理后的验证数据"""
    with open('fine_tune_data/validation_data.json', 'r', encoding='utf-8') as f:
        return json.load(f)

def format_test_prompt(sample: Dict) -> str:
    """格式化测试提示"""
    input_data = sample['input']
    
    # 格式化历史记录
    history_str = "\n".join([
        f"- Time: {h['timestamp']}, Location: {h['place_name']} ({h['place_type']}), " \
        f"Weekday: {h['day_of_week']}, Time of day: {h['time_of_day']:.2f}"
        for h in input_data['history']
    ])
    
    return f"""User ID: {input_data['user_id']}
Trajectory ID: {input_data['trajectory_id']}

Historical visits: 
{history_str}

Current context:
- Date: {input_data['context']['date']}
- Weekday: {input_data['context']['day_of_week']}
- Time of day: {input_data['context']['time_of_day']:.2f}

Please predict the user's next visit location."""

def run_model(prompt: str) -> Dict:
    """运行模型并获取预测结果"""
    result = subprocess.run(
        ['ollama', 'run', 'next_location_predictor', prompt],
        capture_output=True,
        text=True
    )
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        print(f"Error parsing model output: {result.stdout}")
        return None

def validate_model(num_samples: int = None):
    """验证模型"""
    validation_data = load_validation_data()
    
    if num_samples is None:
        # 使用所有验证样本
        test_samples = validation_data
    else:
        # 随机选择指定数量的样本
        test_samples = random.sample(validation_data, min(num_samples, len(validation_data)))
    
    correct_predictions = 0
    total_predictions = 0
    
    for sample in test_samples:
        prompt = format_test_prompt(sample)
        expected = sample['output']['next_location']
        predicted = run_model(prompt)
        
        if predicted is None:
            continue
            
        total_predictions += 1
        if (predicted['place_name'] == expected['place_name'] and 
            predicted['place_type'] == expected['place_type']):
            correct_predictions += 1
        else:
            print(f"\nTest case failed:")
            print(f"Prompt: {prompt}")
            print(f"Expected: {expected}")
            print(f"Predicted: {predicted}")
    
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    print(f"\nValidation Results:")
    print(f"Total samples tested: {total_predictions}")
    print(f"Correct predictions: {correct_predictions}")
    print(f"Accuracy: {accuracy:.2%}")

if __name__ == "__main__":
    # 如果不指定num_samples，将使用所有验证样本
    validate_model() 