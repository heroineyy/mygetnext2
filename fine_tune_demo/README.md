# 基于 Ollama 的地点预测模型微调指南

本指南将帮助您完成基于 Ollama 的地点预测模型的微调过程。我们将使用 LoRA 方法对 Mistral 7B 模型进行微调，使其能够根据用户的历史轨迹预测下一个可能访问的地点。

## 环境要求

- Python 3.8+
- Ollama
- jq (用于处理 JSON)

## 数据准备

### 1. 训练数据
- 位置：`../dataset/NYC/NYC_train_with_categories.csv`
- 格式：CSV 文件，包含用户轨迹数据

### 2. 验证数据
- 位置：`../dataset/NYC/NYC_val_with_categories.csv`
- 格式：CSV 文件，与训练数据格式相同

## 执行流程

### 1. 准备训练数据
```bash
# 处理训练数据
python process_data.py 

# 准备训练样本
python prepare_training_data.py 
```

### 2. 准备验证数据
```bash
# 处理验证数据
python process_validation_data.py 
```

### 3. 微调模型
```bash
# 执行微调
./fine_tune_model.sh
```

### 4. 验证模型
```bash
# 运行验证
python validate_model.py 
```

## 文件说明

- `process_data.py`: 处理原始数据，生成结构化数据
- `prepare_training_data.py`: 准备训练样本
- `process_validation_data.py`: 处理验证数据
- `fine_tune_model.sh`: 执行模型微调
- `validate_model.py`: 验证模型性能

## 配置说明

### 模型参数
- 基础模型：Mistral 7B
- 微调方法：LoRA
- LoRA 参数：
  - rank: 8
  - alpha: 16
  - dropout: 0.1

### 数据参数
- 最小轨迹长度：2
- 时间信息：包含星期几和一天中的时间
- 地点信息：包含地点名称和类型

## 注意事项

1. 确保所有输入文件路径正确
2. 验证数据会使用训练集生成的地点字典
3. 微调过程可能需要较长时间，取决于数据量和计算资源
4. 建议先使用小规模数据进行测试

## 常见问题

1. **如何处理新的地点类型？**
   - 需要重新处理训练数据，更新地点字典
   - 重新训练模型

2. **如何调整模型性能？**
   - 修改 LoRA 参数
   - 调整训练数据量
   - 优化提示词

3. **如何评估模型效果？**
   - 使用验证集进行测试
   - 查看准确率和错误案例
   - 分析预测结果

## 维护建议

1. 使用配置文件管理路径和参数
2. 定期备份训练数据和模型
3. 记录每次微调的结果和参数
4. 保持代码和文档的同步更新 