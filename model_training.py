import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
import joblib

def load_data(file_path):
    data = pd.read_csv(file_path)
    X = data.drop('target', axis=1)
    y = data['target']
    return X, y

def preprocess_data(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler

def train_ensemble_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    X_train_scaled, scaler = preprocess_data(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
    lr_clf = LogisticRegression(random_state=42)
    
    voting_clf = VotingClassifier(
        estimators=[('random_forest', rf_clf), ('logistic_regression', lr_clf)],
        voting='soft'
    )
    
    voting_clf.fit(X_train_scaled, y_train)
    
    joblib.dump(voting_clf, 'ensemble_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    
    return voting_clf, scaler, X_test_scaled, y_test

if __name__ == "__main__":
    X, y = load_data('data.csv')
    model, scaler, X_test, y_test = train_ensemble_model(X, y)
    print("模型训练完成，已保存为 ensemble_model.pkl")
    print("数据标准化器已保存为 scaler.pkl")
