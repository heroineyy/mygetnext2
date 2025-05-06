import os
import json
from llama_factory import LlamaFactory, LlamaConfig

def load_training_data(file_path):
    """加载训练数据"""
    with open(file_path, "r") as f:
        return [json.loads(line) for line in f]

def main():
    # 设置模型和训练参数
    config = LlamaConfig(
        model_name_or_path="mistralai/Mistral-7B-v0.1",
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        save_steps=100,
        warmup_steps=100,
        weight_decay=0.01,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
    )

    # 加载和处理数据
    print("Loading and processing data...")
    train_data = load_training_data("fine_tune_data/training_data.jsonl")

    # 初始化 Llama-Factory
    llama_factory = LlamaFactory(config)

    # 开始训练
    print("Starting training...")
    llama_factory.train(train_data)

    # 保存模型
    print("Saving model...")
    llama_factory.save_model()

    print("Training completed!")

if __name__ == "__main__":
    main() 