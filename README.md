# 模型集成与API服务项目

本项目实现了一个基于集成学习的机器学习模型系统，采用模块化架构设计，提供模型训练、评估和REST API预测服务。

## 架构设计

### 架构师决策

本项目采用**三层架构**设计，遵循**关注点分离**原则：

1. **训练层** (`model_training.py`) - 负责模型构建与训练
2. **评估层** (`model_evaluation.py`) - 负责模型性能评估
3. **服务层** (`api_service.py`) - 负责对外提供REST API服务

### 架构设计原则：
- **模块化**：各模块职责单一，便于维护和扩展
- **可扩展**：通过文件解耦，可独立替换各模块
- **可测试**：各模块可独立测试
- **可部署**：支持独立部署各服务

## 功能特性

1. **模型集成方案**：使用随机森林和逻辑回归两种算法进行集成
2. **模型评估模块**：输出准确率、F1-score、混淆矩阵等评估指标
3. **REST API服务**：提供HTTP接口供外部调用预测服务

## 核心模块说明

| 文件名称 | 核心职责 | 主要函数 | 依赖关系 |
|---------|---------|---------|---------|
| `model_training.py` | 负责数据加载、预处理、集成模型训练和模型保存 | `load_data()` - 从CSV文件加载数据<br>`preprocess_data()` - 数据标准化<br>`train_ensemble_model()` - 训练集成模型 | pandas, numpy, scikit-learn, joblib |
| `model_evaluation.py` | 负责模型性能评估，计算并输出各种评估指标 | `evaluate_model()` - 计算评估指标<br>`print_evaluation_results()` - 格式化输出评估结果 | numpy, scikit-learn, joblib, model_training |
| `api_service.py` | 提供REST API服务，接收预测请求并返回结果 | `predict()` - 处理预测请求<br>`health()` - 健康检查接口<br>`model_info()` - 查询模型信息接口 | flask, joblib, numpy |

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 训练模型

```bash
python model_training.py
```

### 2. 评估模型

```bash
python model_evaluation.py
```

### 3. 启动API服务

```bash
python api_service.py
```

## API接口

### 健康检查
```
GET /health
```

### 预测接口
```
POST /predict
Content-Type: application/json

{
  "features": [5.1, 3.5, 1.4, 0.2]
}
```

### 模型信息查询接口
```
GET /model_info
```
