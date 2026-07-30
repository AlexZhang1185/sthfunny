import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Union
from corner_data_processor import CornerDataProcessor
from corner_model_persistence import ModelPersistence
from corner_model_definition import CornerMultiTaskModel

class CornerPredictor:
    """
    角球预测推理接口
    提供简单易用的API，无需了解内部实现细节
    """

    def __init__(self, model_package_path: str = None):
        """
        初始化预测器
        :param model_package_path: 模型包路径(.zip)，如果为None则尝试加载最新模型
        """
        self.data_processor = None
        self.model = None
        self.scaler = None
        self.config = None
        self.metrics = None

        if model_package_path is None:
            # 查找最新的模型包
            models = ModelPersistence.list_available_models()
            if len(models) == 0:
                raise ValueError("未找到可用的模型包，请指定模型包路径")
            model_package_path = models[0]['path']
            print(f"自动加载最新模型: {model_package_path}")

        self._load_model(model_package_path)

    def _load_model(self, package_path: str):
        """加载模型包"""
        package = ModelPersistence.load_model_package(package_path)
        self.model = package['model']
        self.scaler = package['scaler']
        self.config = package['config']
        self.metrics = package['metrics']

        # 初始化数据处理器
        self.data_processor = CornerDataProcessor(
            use_enhanced_features=self.config.get('use_enhanced_features', False)
        )
        # 覆盖特征名称
        self.data_processor.feature_names = self.config.get('feature_names', self.data_processor.feature_names)
        # 设置标准化器
        self.data_processor.scaler = self.scaler

    def predict(self,
               pos_list: List[float],
               time_list: List[Union[int, str]],
               odds_data: Dict = None) -> Dict:
        """
        预测单场比赛的终场角球情况
        :param pos_list: 盘口序列，例如 [9.0, 8.5, 10.0, 9.5]
        :param time_list: 对应的时间序列（分钟），例如 ["38", "46", "50", "61"] 或 [38, 46, 50, 61]
        :param odds_data: 赔率数据（可选，只有使用增强特征的模型需要）
                         格式: {'big_odds': 1.92, 'small_odds': 1.92}
        :return: 预测结果字典
        """
        # 输入校验
        if len(pos_list) == 0:
            raise ValueError("盘口序列不能为空")
        if len(time_list) != len(pos_list):
            raise ValueError("时间序列长度必须与盘口序列一致")

        # 提取特征
        features_df = self.data_processor.extract_features_from_match(pos_list, time_list, odds_data)

        # 标准化
        X_scaled = self.scaler.transform(features_df)

        # 预测
        prediction = self.model.predict_single_match(X_scaled)

        # 添加相对于每个历史盘口的高低概率
        prediction['vs_each_handicap'] = []
        for h in pos_list:
            h_pred = self.model.predict_vs_handicap(X_scaled, h)
            prediction['vs_each_handicap'].append({
                'handicap': h,
                'over_probability': h_pred['over_probability'],
                'under_probability': h_pred['under_probability'],
                'is_over': h_pred['is_over']
            })

        return prediction

    def predict_vs_handicap(self,
                           pos_list: List[float],
                           time_list: List[Union[int, str]],
                           handicap: float,
                           odds_data: Dict = None) -> Dict:
        """
        预测相对于指定盘口的高低概率
        :param pos_list: 盘口序列
        :param time_list: 时间序列
        :param handicap: 要比较的盘口值
        :param odds_data: 赔率数据（可选）
        :return: 预测结果
        """
        # 提取特征
        features_df = self.data_processor.extract_features_from_match(pos_list, time_list, odds_data)
        X_scaled = self.scaler.transform(features_df)

        # 预测
        return self.model.predict_vs_handicap(X_scaled, handicap)

    def predict_batch(self, matches: List[Dict]) -> List[Dict]:
        """
        批量预测多场比赛
        :param matches: 比赛列表，每个元素格式:
            {
                'pos': [盘口序列],
                'time': [时间序列],
                'odds': {赔率数据} (可选)
            }
        :return: 预测结果列表
        """
        results = []
        for match in matches:
            try:
                pred = self.predict(
                    pos_list=match['pos'],
                    time_list=match['time'],
                    odds_data=match.get('odds')
                )
                results.append({
                    'status': 'success',
                    'prediction': pred
                })
            except Exception as e:
                results.append({
                    'status': 'error',
                    'error': str(e)
                })
        return results

    def get_model_info(self) -> Dict:
        """获取模型信息"""
        return {
            'version': self.config.get('version'),
            'created_at': self.config.get('created_at'),
            'input_dim': self.config.get('input_dim'),
            'use_enhanced_features': self.config.get('use_enhanced_features'),
            'feature_names': self.config.get('feature_names'),
            'metrics': self.metrics
        }

    def print_prediction_report(self, prediction: Dict, show_all_handicaps: bool = False):
        """
        友好打印预测报告
        :param prediction: predict()返回的预测结果
        :param show_all_handicaps: 是否显示相对于所有历史盘口的概率
        """
        print("\n" + "="*60)
        print("⚽ 角球预测报告")
        print("="*60)

        print(f"\n🎯 预测终场角球数: {prediction['predicted_corners']:.1f}")
        print(f"📊 95%置信区间: [{prediction['confidence_interval'][0]:.1f}, {prediction['confidence_interval'][1]:.1f}]")
        print(f"📈 超过最终盘口概率: {prediction['over_probability']:.1%}")
        print(f"📉 低于最终盘口概率: {prediction['under_probability']:.1%}")

        print("\n📊 角球区间概率:")
        for interval, prob in prediction['interval_probabilities'].items():
            print(f"  {interval}: {prob:.1%}")
        print(f"✅ 最可能区间: {prediction['predicted_interval']}")

        if show_all_handicaps and 'vs_each_handicap' in prediction:
            print("\n🔍 相对于每个历史盘口的预测:")
            for idx, h in enumerate(prediction['vs_each_handicap']):
                status = "✅ OVER" if h['is_over'] else "❌ UNDER"
                print(f"  盘口 {h['handicap']:4.1f}: {h['over_probability']:5.1%} OVER | {h['under_probability']:5.1%} UNDER | {status}")

        print("\n" + "="*60)
        print("⚠️  预测仅供参考，足球比赛存在不确定性")
        print("="*60)
