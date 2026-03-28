import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from model_training import ModelTrainer


class ModelEvaluator:
    def __init__(self, model_path='ensemble_model.joblib'):
        self.trainer = ModelTrainer(model_path)
        self.model = self.trainer.load_model()
        
    def _preprocess(self, data):
        """数据预处理"""
        # 过滤年龄
        filtered_data = data[(data['age'] >= 18)].dropna()
        
        # 转换visit_frequency为数值（频率越高数值越大）
        frequency_mapping = {
            'rarely': 1,
            'monthly': 2,
            'weekly': 3,
            'daily': 4
        }
        filtered_data['visit_frequency'] = filtered_data['visit_frequency'].map(frequency_mapping)
        
        # 转换satisfaction_level为数值
        satisfaction_mapping = {
            'low': 1,
            'medium': 2,
            'high': 3
        }
        filtered_data['satisfaction_level'] = filtered_data['satisfaction_level'].map(satisfaction_mapping)
        
        # 转换gender为数值
        gender_mapping = {
            'male': 0,
            'female': 1
        }
        filtered_data['gender'] = filtered_data['gender'].map(gender_mapping)
        
        return filtered_data
    
    def load_and_split_data(self, data_path='mcdonald_data.csv', test_size=0.2, random_state=42):
        """加载数据并划分训练集和测试集"""
        # 加载数据
        data = pd.read_csv(data_path)
        processed_data = self._preprocess(data)
        
        # 分离特征和标签
        X = processed_data.drop('liked_mcdonalds', axis=1)
        y = processed_data['liked_mcdonalds']
        
        # 划分数据集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        return X_train, X_test, y_train, y_test
    
    def evaluate_model(self, X_test, y_test):
        """评估模型性能"""
        # 预测
        y_pred = self.model.predict(X_test)
        
        # 计算评估指标
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # 混淆矩阵
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        # 分类报告
        class_report = classification_report(y_test, y_pred)
        
        # 整理结果
        evaluation_results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': conf_matrix,
            'classification_report': class_report
        }
        
        return evaluation_results
    
    def print_evaluation_results(self, results):
        """打印评估结果"""
        print("=" * 50)
        print("模型评估结果")
        print("=" * 50)
        print(f"准确率 (Accuracy): {results['accuracy']:.4f}")
        print(f"精确率 (Precision): {results['precision']:.4f}")
        print(f"召回率 (Recall): {results['recall']:.4f}")
        print(f"F1值 (F1-Score): {results['f1_score']:.4f}")
        print("\n混淆矩阵:")
        print(results['confusion_matrix'])
        print("\n分类报告:")
        print(results['classification_report'])
        print("=" * 50)
    
    def run_evaluation(self, data_path='mcdonald_data.csv'):
        """执行完整的评估流程"""
        # 加载并划分数据
        _, X_test, _, y_test = self.load_and_split_data(data_path)
        
        # 评估模型
        results = self.evaluate_model(X_test, y_test)
        
        # 打印结果
        self.print_evaluation_results(results)
        
        return results


if __name__ == '__main__':
    evaluator = ModelEvaluator()
    evaluator.run_evaluation()
