from flask import Flask, request, jsonify
from model_training import ModelTrainer
import pandas as pd
import os

app = Flask(__name__)

# 初始化模型训练器
trainer = ModelTrainer()

# 检查模型是否存在，如果不存在则训练
if not os.path.exists(trainer.model_path):
    print("模型不存在，开始训练...")
    trainer.train_model()
else:
    print("加载现有模型...")

# 加载模型
model = trainer.load_model()


@app.route('/predict', methods=['POST'])
def predict():
    """
    预测API接口
    请求格式：JSON
    {
        "age": 25,
        "visit_frequency": 2,
        "avg_spend": 30,
        "satisfaction": 4,
        "healthy_options_importance": 3
    }
    """
    try:
        # 获取请求数据
        data = request.json
        
        # 验证输入数据
        required_fields = ['age', 'visit_frequency', 'avg_spend', 'satisfaction', 'healthy_options_importance']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'缺少必填字段: {field}'}), 400
        
        # 转换为DataFrame
        df = pd.DataFrame([data])
        
        # 进行预测
        prediction = model.predict(df)[0]
        probabilities = model.predict_proba(df)[0]
        confidence = float(max(probabilities))
        
        # 准备响应
        response = {
            'prediction': int(prediction),
            'confidence': confidence,
            'probabilities': {
                'class_0': float(probabilities[0]),
                'class_1': float(probabilities[1])
            },
            'model_type': 'ensemble (RandomForest + GradientBoosting)'
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    return jsonify({'status': 'healthy', 'message': 'API服务运行正常'})


@app.route('/model-info', methods=['GET'])
def model_info():
    """获取模型信息"""
    return jsonify({
        'model_type': 'ensemble',
        'classifiers': ['RandomForestClassifier', 'GradientBoostingClassifier'],
        'voting_method': 'soft',
        'model_path': trainer.model_path
    })

@app.route('/feature-importance', methods=['GET'])
def feature_importance():
    """获取模型特征重要性"""
    try:
        # 获取特征重要性（从集成模型中的第一个分类器获取）
        classifier = model.named_steps['classifier']
        rf_importance = classifier.estimators_[0].feature_importances_
        gb_importance = classifier.estimators_[1].feature_importances_
        
        # 平均重要性
        avg_importance = (rf_importance + gb_importance) / 2
        
        # 获取特征名称
        feature_names = ['age', 'gender', 'income', 'visit_frequency', 'satisfaction_level']
        
        # 排序并返回
        importance_dict = dict(zip(feature_names, avg_importance))
        sorted_importance = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
        
        return jsonify({
            'feature_importance': sorted_importance,
            'random_forest_importance': dict(zip(feature_names, rf_importance)),
            'gradient_boosting_importance': dict(zip(feature_names, gb_importance))
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
