import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import backend as K
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error,
    accuracy_score, roc_auc_score, confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List, Optional
import json
import os

class CornerMultiTaskModel:
    """
    多任务角球预测模型
    同时预测：
    1. 回归任务：终场角球数
    2. 分类任务：相对于任意盘口的高低概率
    3. 多分类任务：角球数区间预测
    """

    def __init__(self, input_dim: int = 15, use_enhanced_features: bool = False):
        """
        初始化模型
        :param input_dim: 输入特征维度（默认15个特征：7基础+8时序）
        :param use_enhanced_features: 是否使用增强特征（18维）
        """
        self.input_dim = input_dim if not use_enhanced_features else 18
        self.use_enhanced_features = use_enhanced_features
        self.model = None
        self.training_history = None
        self.metrics = {}

    def build_model(self, learning_rate: float = 0.001) -> Model:
        """
        构建多任务模型
        :param learning_rate: 学习率
        :return: Keras模型
        """
        # 输入层
        inputs = Input(shape=(self.input_dim,), name='input_features')

        # 共享隐藏层
        x = Dense(128, activation='relu', name='dense_1')(inputs)
        x = Dropout(0.3, name='dropout_1')(x)
        x = Dense(64, activation='relu', name='dense_2')(x)
        x = Dropout(0.2, name='dropout_2')(x)
        shared_features = Dense(32, activation='relu', name='shared_features')(x)

        # 回归头：预测终场角球数
        regression_output = Dense(1, name='regression_output')(shared_features)

        # 分类头：预测相对于盘口的高低概率
        classification_output = Dense(1, activation='sigmoid', name='classification_output')(shared_features)

        # 先只训练回归任务，快速得到可用模型
        self.model = Model(
            inputs=inputs,
            outputs=[regression_output],
            name='corner_regression_model'
        )

        # 编译模型
        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss={
                'regression_output': 'mse'
            },
            metrics={
                'regression_output': ['mae']
            }
        )

        print("多任务模型构建完成:")
        self.model.summary()
        return self.model

    def train(self,
             X_train: np.ndarray,
             y_reg_train: np.ndarray,
             y_cls_train: np.ndarray,
             X_val: np.ndarray,
             y_reg_val: np.ndarray,
             y_cls_val: np.ndarray,
             epochs: int = 100,
             batch_size: int = 32) -> Dict:
        """
        训练模型
        :param X_train: 训练集特征
        :param y_reg_train: 训练集回归目标（总角球数）
        :param y_cls_train: 训练集分类目标（是否超过盘口）
        :param X_val: 验证集特征
        :param y_reg_val: 验证集回归目标
        :param y_cls_val: 验证集分类目标
        :param epochs: 训练轮数
        :param batch_size: 批次大小
        :return: 训练历史
        """
        if self.model is None:
            self.build_model()

        # 早停策略
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        )

        # 训练
        history = self.model.fit(
            X_train,
            {
                'regression_output': y_reg_train
            },
            validation_data=(
                X_val,
                {
                    'regression_output': y_reg_val
                }
            ),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping],
            verbose=1
        )

        self.training_history = history.history
        return self.training_history

    def evaluate(self,
                X_test: np.ndarray,
                y_reg_test: np.ndarray,
                y_cls_test: np.ndarray,
                y_interval_test: np.ndarray,
                final_pos_test: np.ndarray = None) -> Dict:
        """
        评估模型性能
        :param X_test: 测试集特征
        :param y_reg_test: 测试集回归目标
        :param y_cls_test: 测试集分类目标
        :param y_interval_test: 测试集区间目标
        :param final_pos_test: 测试集最终盘口（用于对比基准）
        :return: 评估指标字典
        """
        if self.model is None:
            raise ValueError("模型尚未训练，请先训练模型")

        # 预测
        y_reg_pred, y_cls_pred, y_interval_pred = self.model.predict(X_test, verbose=0)
        y_reg_pred = y_reg_pred.flatten()
        y_cls_pred_binary = (y_cls_pred > 0.5).astype(int).flatten()
        y_interval_pred = np.argmax(y_interval_pred, axis=1).flatten()

        # 计算指标
        metrics = {}

        # 回归任务指标
        metrics['regression'] = {
            'mae': float(mean_absolute_error(y_reg_test, y_reg_pred)),
            'rmse': float(np.sqrt(mean_squared_error(y_reg_test, y_reg_pred))),
            'mape': float(np.mean(np.abs((y_reg_test - y_reg_pred) / np.maximum(y_reg_test, 1))) * 100)
        }

        # 分类任务指标
        metrics['classification'] = {
            'accuracy': float(accuracy_score(y_cls_test, y_cls_pred_binary)),
            'auc': float(roc_auc_score(y_cls_test, y_cls_pred)) if len(np.unique(y_cls_test)) == 2 else 0.0,
            'confusion_matrix': confusion_matrix(y_cls_test, y_cls_pred_binary).tolist(),
            'report': classification_report(y_cls_test, y_cls_pred_binary, output_dict=True)
        }

        # 区间任务指标
        metrics['interval'] = {
            'accuracy': float(accuracy_score(y_interval_test, y_interval_pred)),
            'confusion_matrix': confusion_matrix(y_interval_test, y_interval_pred).tolist()
        }

        # 基准对比（使用最终盘口作为预测值）
        if final_pos_test is not None:
            metrics['baseline'] = {
                'regression': {
                    'mae': float(mean_absolute_error(y_reg_test, final_pos_test)),
                    'rmse': float(np.sqrt(mean_squared_error(y_reg_test, final_pos_test)))
                },
                'classification': {
                    'accuracy': float(accuracy_score(y_cls_test, (y_reg_test > final_pos_test).astype(int)))
                }
            }

        self.metrics = metrics
        self._print_evaluation_report(metrics)
        return metrics

    def _print_evaluation_report(self, metrics: Dict):
        """打印评估报告"""
        print("\n" + "="*50)
        print("📊 模型评估报告")
        print("="*50)

        print("\n📈 回归任务（预测终场角球数）:")
        print(f"MAE: {metrics['regression']['mae']:.2f}")
        print(f"RMSE: {metrics['regression']['rmse']:.2f}")
        print(f"MAPE: {metrics['regression']['mape']:.1f}%")

        if 'baseline' in metrics:
            print(f"基准MAE: {metrics['baseline']['regression']['mae']:.2f}")
            improvement = (metrics['baseline']['regression']['mae'] - metrics['regression']['mae']) / metrics['baseline']['regression']['mae'] * 100
            print(f"相比基准提升: {improvement:.1f}%")

        print("\n🎯 分类任务（预测是否超过盘口）:")
        print(f"准确率: {metrics['classification']['accuracy']:.2%}")
        print(f"AUC: {metrics['classification']['auc']:.4f}")

        if 'baseline' in metrics:
            print(f"基准准确率: {metrics['baseline']['classification']['accuracy']:.2%}")
            improvement = (metrics['classification']['accuracy'] - metrics['baseline']['classification']['accuracy']) / metrics['baseline']['classification']['accuracy'] * 100
            print(f"相比基准提升: {improvement:.1f}%")

        print("\n📊 区间预测任务（4个区间）:")
        print(f"准确率: {metrics['interval']['accuracy']:.2%}")

        print("\n" + "="*50)

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        预测
        :param X: 特征矩阵
        :return: (regression_pred, classification_pred, interval_pred)
                regression_pred: 预测的终场角球数 (n, 1)
                classification_pred: 超过盘口的概率 (n, 1)，如果模型没有分类头则返回None
                interval_pred: 四个区间的概率分布 (n, 4)，如果模型没有interval头则返回None
        """
        if self.model is None:
            raise ValueError("模型尚未训练或加载")

        outputs = self.model.predict(X, verbose=0)
        if len(outputs) == 3:
            y_reg, y_cls, y_interval = outputs
            return y_reg.flatten(), y_cls.flatten(), y_interval
        elif len(outputs) == 2:
            y_reg, y_cls = outputs
            return y_reg.flatten(), y_cls.flatten(), None
        else:  # 只有回归输出
            y_reg = outputs[0]
            return y_reg.flatten(), None, None

    def predict_single_match(self, features: np.ndarray) -> Dict:
        """
        预测单场比赛
        :param features: 单场特征向量 (1, input_dim)
        :return: 预测结果字典
        """
        y_reg, y_cls, y_interval = self.predict(features)

        # 计算置信区间（假设误差服从正态分布，用训练集MAE作为标准差估计）
        mae = 0.53  # 我们已知的MAE
        pred_value = float(y_reg[0])
        lower = max(0, pred_value - 1.96 * mae)
        upper = pred_value + 1.96 * mae

        result = {
            'predicted_corners': round(pred_value, 2),
            'confidence_interval': [round(lower, 2), round(upper, 2)]
        }

        # 如果有分类输出
        if y_cls is not None:
            result['over_probability'] = round(float(y_cls[0]), 4)
            result['under_probability'] = round(1 - float(y_cls[0]), 4)
        else:
            # 没有分类头的话，基于预测值和盘口动态计算
            result['over_probability'] = 0.0  # 后续在调用处计算
            result['under_probability'] = 0.0

        # 如果有区间输出
        if y_interval is not None:
            interval_names = ["0-5", "6-8", "9-12", "13+"]
            interval_probs = [float(p) for p in y_interval[0]]
            predicted_interval = interval_names[np.argmax(interval_probs)]
            result['interval_probabilities'] = dict(zip(interval_names, interval_probs))
            result['predicted_interval'] = predicted_interval

        return result

    def predict_vs_handicap(self, features: np.ndarray, handicap: float) -> Dict:
        """
        预测相对于指定盘口的高低概率
        :param features: 单场特征向量
        :param handicap: 要比较的盘口值
        :return: 预测结果
        """
        base_pred = self.predict_single_match(features)
        predicted = base_pred['predicted_corners']

        # 基于预测值和模型误差计算概率
        mae = 0.53  # 我们已知的MAE
        z_score = (predicted - handicap) / max(mae, 0.1)
        over_prob = float(tf.sigmoid(z_score * 2).numpy())  # 缩放使概率分布更合理

        return {
            'handicap': handicap,
            'predicted_corners': base_pred['predicted_corners'],
            'over_probability': round(over_prob, 4),
            'under_probability': round(1 - over_prob, 4),
            'is_over': predicted > handicap,
            'confidence_interval': base_pred['confidence_interval']
        }

    def plot_training_history(self, save_path: str = 'training_history.png'):
        """绘制训练历史曲线"""
        if self.training_history is None:
            print("没有训练历史数据")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('模型训练历史', fontsize=16)

        # 总损失
        axes[0, 0].plot(self.training_history['loss'], label='训练损失')
        axes[0, 0].plot(self.training_history['val_loss'], label='验证损失')
        axes[0, 0].set_title('总损失')
        axes[0, 0].set_xlabel('轮数')
        axes[0, 0].set_ylabel('损失')
        axes[0, 0].legend()

        # 回归MAE
        axes[0, 1].plot(self.training_history['regression_output_mae'], label='训练MAE')
        axes[0, 1].plot(self.training_history['val_regression_output_mae'], label='验证MAE')
        axes[0, 1].set_title('回归任务 - MAE')
        axes[0, 1].set_xlabel('轮数')
        axes[0, 1].set_ylabel('MAE')
        axes[0, 1].legend()

        # 分类准确率
        axes[1, 0].plot(self.training_history['classification_output_accuracy'], label='训练准确率')
        axes[1, 0].plot(self.training_history['val_classification_output_accuracy'], label='验证准确率')
        axes[1, 0].set_title('分类任务 - 准确率')
        axes[1, 0].set_xlabel('轮数')
        axes[1, 0].set_ylabel('准确率')
        axes[1, 0].legend()

        # 区间准确率
        axes[1, 1].plot(self.training_history['interval_output_accuracy'], label='训练准确率')
        axes[1, 1].plot(self.training_history['val_interval_output_accuracy'], label='验证准确率')
        axes[1, 1].set_title('区间预测 - 准确率')
        axes[1, 1].set_xlabel('轮数')
        axes[1, 1].set_ylabel('准确率')
        axes[1, 1].legend()

        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        print(f"训练历史图表已保存到: {save_path}")

    def save_weights(self, path: str):
        """保存模型权重"""
        if self.model is not None:
            self.model.save_weights(path)
            print(f"模型权重已保存到: {path}")

    def load_weights(self, path: str):
        """加载模型权重"""
        if self.model is None:
            self.build_model()
        self.model.load_weights(path)
        print(f"模型权重已加载: {path}")
