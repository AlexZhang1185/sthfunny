import os
import json
import pickle
import zipfile
from datetime import datetime
from typing import Dict, Optional
from sklearn.preprocessing import StandardScaler
from corner_model_definition import CornerMultiTaskModel
from corner_data_processor import CornerDataProcessor

class ModelPersistence:
    """
    模型持久化管理器
    负责将模型、标准化器、配置信息打包保存/加载
    生成自包含的模型包，推理时无需依赖训练数据
    """

    VERSION = "1.0.0"
    MODEL_WEIGHTS_FILE = "model_weights.weights.h5"
    SCALER_FILE = "scaler.pkl"
    CONFIG_FILE = "config.json"
    METRICS_FILE = "metrics.json"

    @classmethod
    def save_model_package(cls,
                          model: CornerMultiTaskModel,
                          scaler: StandardScaler,
                          feature_names: list,
                          metrics: Dict = None,
                          save_path: str = None,
                          model_name: str = "corner_prediction_model") -> str:
        """
        保存完整模型包
        :param model: 训练好的多任务模型
        :param scaler: 拟合好的标准化器
        :param feature_names: 特征名称列表
        :param metrics: 评估指标
        :param save_path: 保存路径（默认当前目录）
        :param model_name: 模型名称
        :return: 保存的文件路径
        """
        # 生成默认保存路径
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d")
            save_path = f"{model_name}_v{cls.VERSION}_{timestamp}.zip"

        # 创建临时目录
        temp_dir = f".temp_model_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        os.makedirs(temp_dir, exist_ok=True)

        try:
            # 1. 保存模型权重
            weights_path = os.path.join(temp_dir, cls.MODEL_WEIGHTS_FILE)
            model.save_weights(weights_path)

            # 2. 保存标准化器
            scaler_path = os.path.join(temp_dir, cls.SCALER_FILE)
            with open(scaler_path, 'wb') as f:
                pickle.dump(scaler, f)

            # 3. 保存配置信息
            config = {
                'version': cls.VERSION,
                'input_dim': model.input_dim,
                'use_enhanced_features': model.use_enhanced_features,
                'feature_names': feature_names,
                'created_at': datetime.now().isoformat(),
                'model_type': 'multi_task_corner_predictor'
            }
            config_path = os.path.join(temp_dir, cls.CONFIG_FILE)
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)

            # 4. 保存评估指标
            if metrics is not None:
                metrics_path = os.path.join(temp_dir, cls.METRICS_FILE)
                with open(metrics_path, 'w', encoding='utf-8') as f:
                    json.dump(metrics, f, indent=2, ensure_ascii=False)

            # 5. 打包成zip
            with zipfile.ZipFile(save_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for root, dirs, files in os.walk(temp_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, temp_dir)
                        zipf.write(file_path, arcname)

            print(f"✅ 模型包已保存到: {save_path}")
            print(f"📦 包内容: 模型权重、标准化器、配置文件、评估指标")
            return save_path

        finally:
            # 清理临时目录
            import shutil
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

    @classmethod
    def load_model_package(cls, package_path: str) -> Dict:
        """
        加载完整模型包
        :param package_path: 模型包路径(.zip)
        :return: 包含所有组件的字典
            {
                'model': CornerMultiTaskModel实例,
                'scaler': StandardScaler实例,
                'config': 配置字典,
                'metrics': 评估指标字典(如果有)
            }
        """
        if not os.path.exists(package_path):
            raise FileNotFoundError(f"模型包不存在: {package_path}")

        # 创建临时目录
        temp_dir = f".temp_load_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        os.makedirs(temp_dir, exist_ok=True)

        try:
            # 解压zip包
            with zipfile.ZipFile(package_path, 'r') as zipf:
                zipf.extractall(temp_dir)

            # 1. 加载配置
            config_path = os.path.join(temp_dir, cls.CONFIG_FILE)
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)

            # 版本检查
            if config.get('version') != cls.VERSION:
                print(f"⚠️  模型版本不兼容: 包版本 {config.get('version')}, 当前代码版本 {cls.VERSION}")

            # 2. 加载标准化器
            scaler_path = os.path.join(temp_dir, cls.SCALER_FILE)
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)

            # 3. 构建并加载模型
            model = CornerMultiTaskModel(
                input_dim=config['input_dim'],
                use_enhanced_features=config['use_enhanced_features']
            )
            weights_path = os.path.join(temp_dir, cls.MODEL_WEIGHTS_FILE)
            model.load_weights(weights_path)

            # 4. 加载评估指标（如果有）
            metrics = None
            metrics_path = os.path.join(temp_dir, cls.METRICS_FILE)
            if os.path.exists(metrics_path):
                with open(metrics_path, 'r', encoding='utf-8') as f:
                    metrics = json.load(f)

            # 加载模型的metrics属性
            if metrics is not None:
                model.metrics = metrics

            print(f"✅ 模型包加载成功: {package_path}")
            print(f"📋 版本: {config.get('version')}")
            print(f"📅 创建时间: {config.get('created_at')}")
            print(f"🔢 特征维度: {config['input_dim']}")

            return {
                'model': model,
                'scaler': scaler,
                'config': config,
                'metrics': metrics
            }

        finally:
            # 清理临时目录
            import shutil
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

    @classmethod
    def get_package_info(cls, package_path: str) -> Optional[Dict]:
        """
        获取模型包信息而不加载完整模型
        :param package_path: 模型包路径
        :return: 配置信息字典
        """
        if not os.path.exists(package_path):
            return None

        try:
            with zipfile.ZipFile(package_path, 'r') as zipf:
                with zipf.open(cls.CONFIG_FILE) as f:
                    config = json.load(f)

                metrics = None
                if cls.METRICS_FILE in zipf.namelist():
                    with zipf.open(cls.METRICS_FILE) as f:
                        metrics = json.load(f)

            return {
                'config': config,
                'metrics': metrics
            }

        except Exception as e:
            print(f"读取模型包信息失败: {str(e)}")
            return None

    @classmethod
    def list_available_models(cls, directory: str = '.') -> list:
        """
        列出目录下所有可用的模型包
        :param directory: 搜索目录
        :return: 模型包信息列表
        """
        models = []
        for filename in os.listdir(directory):
            if filename.endswith('.zip') and ('corner' in filename.lower() or 'model' in filename.lower()):
                file_path = os.path.join(directory, filename)
                info = cls.get_package_info(file_path)
                if info is not None:
                    models.append({
                        'file': filename,
                        'path': file_path,
                        'version': info['config'].get('version'),
                        'created_at': info['config'].get('created_at'),
                        'input_dim': info['config'].get('input_dim'),
                        'accuracy': info['metrics'].get('classification', {}).get('accuracy', 0) if info.get('metrics') else 0
                    })

        # 按创建时间排序
        models.sort(key=lambda x: x['created_at'] if x['created_at'] else '', reverse=True)
        return models
