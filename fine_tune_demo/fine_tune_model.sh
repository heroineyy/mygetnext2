#!/bin/bash

# 设置环境变量
export OLLAMA_HOST=127.0.0.1
export OLLAMA_MODELS=~/.ollama/models

# 下载基础模型（如果还没有）
ollama pull mistral:7b

# 读取metadata以获取有效的地点类型和名称
VALID_PLACE_TYPES=$(cat fine_tune_data/dataset_metadata.json | jq -r '.valid_place_types | @json')
VALID_PLACE_NAMES=$(cat fine_tune_data/dataset_metadata.json | jq -r '.valid_place_names | @json')

# 创建微调配置文件
cat > Modelfile << EOF
FROM mistral:7b

# 设置系统提示
SYSTEM """
You are a location prediction assistant. Your task is to predict the next possible location based on user's historical trajectory.
You should analyze the user's historical visits, current context, and time information to make predictions.

You MUST ONLY choose from these valid place types:
${VALID_PLACE_TYPES}

And these valid place names:
${VALID_PLACE_NAMES}

Your response must be a JSON object with 'place_name' and 'place_type' fields, both values must be from the lists above.
"""

# 设置模板
TEMPLATE """
{{.System}}

User information:
{{.Prompt}}

Please predict the next location:
{{.completion}}
"""

# 设置参数
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER num_ctx 2048
EOF

# 创建基础模型
ollama create next_location_predictor -f Modelfile

# 安装必要的Python包
echo "Installing required packages..."
pip install --upgrade pip
pip install torch transformers datasets peft accelerate bitsandbytes tqdm

# 使用Python脚本进行LoRA微调
echo "Starting LoRA fine-tuning with training data..."
python3 fine_tune.py \
    --model_name_or_path "mistralai/Mistral-7B-v0.1" \
    --train_file "fine_tune_data/training_data.jsonl" \
    --output_dir "./results" \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-4 \
    --fp16 \
    --logging_steps 10 \
    --save_steps 100 \
    --warmup_steps 100 \
    --weight_decay 0.01

# 测试微调后的模型
echo "Testing the fine-tuned model..."
ollama run next_location_predictor "User ID: 123
Trajectory ID: 456

Historical visits: 
- Time: 2024-01-01T10:00:00, Location: Home (Residential), Weekday: 1, Time of day: 0.42
- Time: 2024-01-01T12:00:00, Location: Office (Work), Weekday: 1, Time of day: 0.50

Current context:
- Date: 2024-01-01
- Weekday: 1
- Time of day: 0.75"