import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import joblib

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    
    return {
        'accuracy': accuracy,
        'f1_score': f1,
        'confusion_matrix': conf_matrix,
        'classification_report': class_report
    }

def print_evaluation_results(results):
    print("="*50)
    print("模型评估结果")
    print("="*50)
    print(f"准确率 (Accuracy): {results['accuracy']:.4f}")
    print(f"F1 分数 (F1-Score): {results['f1_score']:.4f}")
    print("\n混淆矩阵:")
    print(results['confusion_matrix'])
    print("\n分类报告:")
    print(results['classification_report'])
    print("="*50)

if __name__ == "__main__":
    from model_training import load_data, train_ensemble_model
    
    X, y = load_data('data.csv')
    model, scaler, X_test, y_test = train_ensemble_model(X, y)
    
    results = evaluate_model(model, X_test, y_test)
    print_evaluation_results(results)
