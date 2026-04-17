#!/usr/bin/env python3
"""
测试脚本 - 验证 McDonaldPredictor 的核心功能
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mcdonald_predictor import (
    McDonaldPredictor,
    InputValidationError,
    PredictionError
)


def test_model_training():
    """测试模型训练"""
    print("=" * 60)
    print("测试 1: 模型训练和评估")
    print("=" * 60)

    try:
        predictor = McDonaldPredictor()
        print("✅ 模型训练成功!")

        eval_results = predictor.get_evaluation_results()
        print(f"\n📊 评估结果:")
        if 'train' in eval_results:
            print(f"  训练集准确率: {eval_results['train']['accuracy']:.4f}")
            print(f"  训练集 F1-score: {eval_results['train']['f1_score']:.4f}")
        if 'test' in eval_results:
            print(f"  测试集准确率: {eval_results['test']['accuracy']:.4f}")
            print(f"  测试集 F1-score: {eval_results['test']['f1_score']:.4f}")
        elif 'overall' in eval_results:
            print(f"  整体准确率: {eval_results['overall']['accuracy']:.4f}")
            print(f"  整体 F1-score: {eval_results['overall']['f1_score']:.4f}")

        feature_importance = predictor.get_feature_importance()
        print(f"\n🎯 特征重要性 (Top 5):")
        for i, (feature, importance) in enumerate(list(feature_importance.items())[:5]):
            print(f"  {i+1}. {feature}: {importance:.4f}")

        model_info = predictor.get_model_info()
        print(f"\n🤖 模型信息:")
        print(f"  类型: {model_info['model_type']}")
        print(f"  基学习器: {', '.join(model_info['base_estimators'])}")
        print(f"  投票方式: {model_info['voting']}")
        print(f"  特征数量: {len(model_info['features'])}")

        return predictor

    except Exception as e:
        print(f"❌ 模型训练失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_prediction(predictor):
    """测试预测功能"""
    print("\n" + "=" * 60)
    print("测试 2: 预测功能")
    print("=" * 60)

    if predictor is None:
        print("❌ 跳过测试: 模型未初始化")
        return

    test_cases = [
        {
            "name": "正常用户 - 男性",
            "data": {
                "age": 30,
                "income": 5000,
                "gender": "male",
                "visit_frequency": "weekly",
                "satisfaction_level": "high"
            }
        },
        {
            "name": "正常用户 - 女性",
            "data": {
                "age": 25,
                "income": 4500,
                "gender": "female",
                "visit_frequency": "monthly",
                "satisfaction_level": "medium"
            }
        },
        {
            "name": "高频用户",
            "data": {
                "age": 40,
                "income": 8000,
                "gender": "male",
                "visit_frequency": "daily",
                "satisfaction_level": "high"
            }
        }
    ]

    for i, test_case in enumerate(test_cases):
        print(f"\n📋 测试用例 {i+1}: {test_case['name']}")
        try:
            result = predictor.predict_single(test_case['data'])
            print(f"  ✅ 预测成功!")
            print(f"     预测结果: {result['prediction_label']} ({result['prediction']})")
            print(f"     置信度: {result['confidence_percent']}%")
        except Exception as e:
            print(f"  ❌ 预测失败: {e}")


def test_input_validation():
    """测试输入验证"""
    print("\n" + "=" * 60)
    print("测试 3: 输入验证")
    print("=" * 60)

    try:
        predictor = McDonaldPredictor()
    except Exception as e:
        print(f"❌ 模型初始化失败: {e}")
        return

    invalid_cases = [
        {
            "name": "年龄为负数",
            "data": {
                "age": -5,
                "income": 5000,
                "gender": "male",
                "visit_frequency": "weekly",
                "satisfaction_level": "high"
            },
            "expected_error": True
        },
        {
            "name": "收入为负数",
            "data": {
                "age": 30,
                "income": -1000,
                "gender": "male",
                "visit_frequency": "weekly",
                "satisfaction_level": "high"
            },
            "expected_error": True
        },
        {
            "name": "无效性别",
            "data": {
                "age": 30,
                "income": 5000,
                "gender": "unknown",
                "visit_frequency": "weekly",
                "satisfaction_level": "high"
            },
            "expected_error": True
        },
        {
            "name": "无效访问频率",
            "data": {
                "age": 30,
                "income": 5000,
                "gender": "male",
                "visit_frequency": "invalid",
                "satisfaction_level": "high"
            },
            "expected_error": True
        },
        {
            "name": "缺少特征",
            "data": {
                "age": 30,
                "income": 5000
            },
            "expected_error": True
        }
    ]

    for i, test_case in enumerate(invalid_cases):
        print(f"\n📋 测试用例 {i+1}: {test_case['name']}")
        try:
            result = predictor.predict_single(test_case['data'])
            if test_case['expected_error']:
                print(f"  ❌ 应该抛出错误但没有抛出")
            else:
                print(f"  ✅ 预测成功 (预期行为)")
        except PredictionError as e:
            if test_case['expected_error']:
                print(f"  ✅ 正确抛出错误: {e}")
            else:
                print(f"  ❌ 意外错误: {e}")
        except InputValidationError as e:
            if test_case['expected_error']:
                print(f"  ✅ 正确抛出输入验证错误: {e}")
            else:
                print(f"  ❌ 意外错误: {e}")
        except Exception as e:
            print(f"  ⚠️  其他错误: {type(e).__name__}: {e}")


def main():
    """主测试函数"""
    print("\n" + "=" * 60)
    print("🚀 开始测试 McDonaldPredictor")
    print("=" * 60)

    predictor = test_model_training()

    if predictor:
        test_prediction(predictor)
        test_input_validation()

    print("\n" + "=" * 60)
    print("✅ 测试完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
