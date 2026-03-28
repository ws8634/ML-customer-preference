import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import VotingClassifier
import joblib
import os


class ModelTrainer:
    def __init__(self, model_path='ensemble_model.joblib'):
        self.model_path = model_path
        
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
    
    def _create_ensemble_model(self):
        """创建集成模型，使用随机森林和梯度提升树"""
        # 随机森林分类器
        rf_clf = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42
        )
        
        # 梯度提升树分类器
        gb_clf = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=3,
            random_state=42
        )
        
        # 投票集成模型
        ensemble_clf = VotingClassifier(
            estimators=[
                ('random_forest', rf_clf),
                ('gradient_boosting', gb_clf)
            ],
            voting='soft'  # 使用软投票，考虑概率
        )
        
        # 创建包含预处理的管道
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', ensemble_clf)
        ])
        
        return model
    
    def train_model(self, data_path='mcdonald_data.csv'):
        """训练模型并保存"""
        # 加载数据
        data = pd.read_csv(data_path)
        processed_data = self._preprocess(data)
        
        # 分离特征和标签
        X = processed_data.drop('liked_mcdonalds', axis=1)
        y = processed_data['liked_mcdonalds']
        
        # 创建并训练模型
        self.model = self._create_ensemble_model()
        self.model.fit(X, y)
        
        # 保存模型
        joblib.dump(self.model, self.model_path)
        print(f"模型已训练并保存到 {self.model_path}")
        
        return self.model
    
    def load_model(self):
        """加载已训练的模型"""
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
            return self.model
        else:
            raise FileNotFoundError(f"模型文件 {self.model_path} 不存在，请先训练模型")


if __name__ == '__main__':
    trainer = ModelTrainer()
    trainer.train_model()
