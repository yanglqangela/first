import os
import numpy as np
import pandas as pd
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.static import create_parameter
from typing import Dict, List, Optional, Tuple, Union
import json
from pathlib import Path

from protonets.models import register_model
from protonets.utils import euclidean_dist


class MultiModalFeatureLoader:
    """
    多模态特征加载器
    支持从comprehensive_features文件夹加载步态、压力、耦合特征
    支持静态和时序两种模式
    """
    
    def __init__(self, base_path: str = "comprehensive_features1"):
        self.base_path = Path(base_path)
        print(f"加载多模态特征，base_path: {self.base_path}")
        self.static_path = self.base_path / "static"
        self.sliding_path = self.base_path / "sliding"
        
        # 特征类型映射
        self.feature_types = {
            'gait': 'gait_features',
            'pressure': 'pressure_features', 
            'coupling': 'coupling_features'
        }
        
        # 缓存加载的特征
        self._feature_cache = {}
        
    def load_features(self, 
                     feature_types: List[str], 
                     mode: str = 'static',
                     data_range: Optional[Tuple[int, int]] = None) -> Dict[str, np.ndarray]:
        """
        加载指定类型的特征
        
        Args:
            feature_types: 特征类型列表 ['gait', 'pressure', 'coupling']
            mode: 特征模式 'static' 或 'sliding'
            data_range: 数据范围 (start_idx, end_idx)，如果为None则加载所有数据
            
        Returns:
            Dict[str, np.ndarray]: 特征字典，键为特征类型，值为特征数组
        """
        if mode not in ['static', 'sliding']:
            raise ValueError(f"模式必须是 'static' 或 'sliding'，得到: {mode}")
            
        # 检查特征类型
        invalid_types = set(feature_types) - set(self.feature_types.keys())
        if invalid_types:
            raise ValueError(f"无效的特征类型: {invalid_types}")
            
        features_dict = {}
        labels_dict = {}
        
        for feature_type in feature_types:
            cache_key = f"{feature_type}_{mode}"
            
            if cache_key not in self._feature_cache:
                # 加载特征
                feature_data, labels = self._load_single_feature_type(feature_type, mode)
                self._feature_cache[cache_key] = (feature_data, labels)
            else:
                feature_data, labels = self._feature_cache[cache_key]
                
            # 应用数据范围过滤
            if data_range is not None:
                start_idx, end_idx = data_range
                feature_data = feature_data[start_idx:end_idx]
                labels = labels[start_idx:end_idx]
                
            features_dict[feature_type] = feature_data
            labels_dict[feature_type] = labels
            
        # 验证标签一致性
        self._validate_labels_consistency(labels_dict)
        
        # 返回特征和统一的标签
        unified_labels = list(labels_dict.values())[0]
        return features_dict, unified_labels
        
    def _load_single_feature_type(self, feature_type: str, mode: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        加载单一类型的特征
        
        Args:
            feature_type: 特征类型
            mode: 特征模式
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (特征数组, 标签数组)
        """
        # 构建路径
        if mode == 'static':
            feature_dir = self.static_path / f"{self.feature_types[feature_type]}_{mode}"
        else:  # sliding
            feature_dir = self.sliding_path / f"{self.feature_types[feature_type]}_{mode}"
            
        if not feature_dir.exists():
            raise FileNotFoundError(f"特征目录不存在: {feature_dir}")
            
        # 获取所有CSV文件
        csv_files = sorted(list(feature_dir.glob("*.csv")))
        if not csv_files:
            raise FileNotFoundError(f"在 {feature_dir} 中未找到CSV文件")
            
        all_features = []
        all_labels = []
        
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                
                if df.empty:
                    print(f"警告: 文件 {csv_file} 为空，跳过")
                    continue
                    
                # 分离特征和标签
                if 'label' in df.columns:
                    features = df.drop('label', axis=1).values
                    labels = df['label'].values
                else:
                    # 如果没有标签列，假设最后一列是标签
                    features = df.iloc[:, :-1].values
                    labels = df.iloc[:, -1].values
                    
                all_features.append(features)
                all_labels.append(labels)
                
            except Exception as e:
                print(f"加载文件 {csv_file} 时出错: {e}")
                continue
                
        if not all_features:
            raise ValueError(f"未能加载任何 {feature_type} 特征")
            
        # 合并所有特征
        combined_features = np.vstack(all_features)
        combined_labels = np.hstack(all_labels)
        
        print(f"成功加载 {feature_type} 特征: {combined_features.shape}, 标签: {len(combined_labels)}")
        
        return combined_features, combined_labels
        
    def _validate_labels_consistency(self, labels_dict: Dict[str, np.ndarray]):
        """
        验证不同特征类型的标签是否一致
        
        Args:
            labels_dict: 标签字典
        """
        if len(labels_dict) <= 1:
            return
            
        reference_labels = None
        reference_type = None
        
        for feature_type, labels in labels_dict.items():
            if reference_labels is None:
                reference_labels = labels
                reference_type = feature_type
            else:
                if not np.array_equal(labels, reference_labels):
                    print(f"警告: {feature_type} 的标签与 {reference_type} 不一致")
                    print(f"{reference_type} 标签: {reference_labels[:5]}...")
                    print(f"{feature_type} 标签: {labels[:5]}...")
                    
    def get_feature_info(self) -> Dict[str, Dict[str, int]]:
        """
        获取特征信息
        
        Returns:
            Dict: 特征信息字典
        """
        info = {}
        
        for feature_type in self.feature_types.keys():
            info[feature_type] = {}
            
            for mode in ['static', 'sliding']:
                try:
                    features, labels = self._load_single_feature_type(feature_type, mode)
                    info[feature_type][mode] = {
                        'samples': len(features),
                        'features': features.shape[1] if len(features.shape) > 1 else 1,
                        'classes': len(np.unique(labels))
                    }
                except Exception as e:
                    info[feature_type][mode] = {'error': str(e)}
                    
        return info


class MultiHeadAttentionFusion(nn.Layer):
    """
    多头注意力机制特征融合模块
    对不同特征动态分配权重，突出重要特征
    """
    
    def __init__(self,
                 feature_dims: Dict[str, int],
                 hidden_dim: int = 128,
                 num_heads: int = 8,
                 dropout_rate: float = 0.2):
        """
        初始化多头注意力融合模块
        
        Args:
            feature_dims: 各特征类型的维度字典
            hidden_dim: 隐藏层维度
            num_heads: 注意力头数
            dropout_rate: dropout比率
        """
        super(MultiHeadAttentionFusion, self).__init__()
        
        self.feature_dims = feature_dims
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.feature_types = list(feature_dims.keys())
        
        assert hidden_dim % num_heads == 0, "hidden_dim必须能被num_heads整除"
        
        # 特征投影层 - 将不同维度的特征投影到统一空间
        self.feature_projections = nn.LayerDict()
        for feature_type, dim in feature_dims.items():
            self.feature_projections[feature_type] = nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            )
            
        # 多头注意力层
        self.query_projection = nn.Linear(hidden_dim, hidden_dim)
        self.key_projection = nn.Linear(hidden_dim, hidden_dim)
        self.value_projection = nn.Linear(hidden_dim, hidden_dim)
        
        # 输出投影
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)
        
        # 层归一化和dropout
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        
        # 前馈网络
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        # 最终融合层
        self.final_fusion = nn.Sequential(
            nn.Linear(hidden_dim * len(feature_dims), hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

    def forward(self, features: Dict[str, paddle.Tensor]) -> Tuple[paddle.Tensor, Dict[str, paddle.Tensor]]:
        """
        前向传播

        Args:
            features: 特征字典，键为特征类型，值为特征张量

        Returns:
            Tuple[paddle.Tensor, Dict[str, paddle.Tensor]]: (融合特征, 注意力权重)
        """
        batch_size = list(features.values())[0].shape[0]

        # 1. 特征投影到统一空间
        projected_features = {}
        for feature_type in self.feature_types:
            if feature_type in features and features[feature_type] is not None:
                projected = self.feature_projections[feature_type](features[feature_type])
                projected_features[feature_type] = projected
            else:
                # 如果特征不存在，用零填充
                projected_features[feature_type] = paddle.zeros([batch_size, self.hidden_dim])

        # 2. 准备多头注意力输入
        # 将所有特征堆叠为序列 [batch_size, num_features, hidden_dim]
        feature_sequence = paddle.stack(list(projected_features.values()), axis=1)

        # 3. 多头注意力计算
        attended_features, attention_weights = self._multi_head_attention(feature_sequence)

        # 4. 残差连接和层归一化
        attended_features = self.layer_norm1(attended_features + feature_sequence)

        # 5. 前馈网络
        ff_output = self.feed_forward(attended_features)
        attended_features = self.layer_norm2(attended_features + ff_output)

        # 6. 最终融合
        # 将序列展平 [batch_size, num_features * hidden_dim]
        flattened_features = attended_features.reshape([batch_size, -1])
        fused_features = self.final_fusion(flattened_features)

        # 7. 整理注意力权重
        attention_dict = {}
        for i, feature_type in enumerate(self.feature_types):
            attention_dict[feature_type] = attention_weights[:, :, i, :]  # [batch_size, num_heads, seq_len]

        return fused_features, attention_dict

    def _multi_head_attention(self, x: paddle.Tensor) -> Tuple[paddle.Tensor, paddle.Tensor]:
        """
        多头注意力计算

        Args:
            x: 输入张量 [batch_size, seq_len, hidden_dim]

        Returns:
            Tuple[paddle.Tensor, paddle.Tensor]: (输出特征, 注意力权重)
        """
        batch_size, seq_len, _ = x.shape

        # 计算Q, K, V
        Q = self.query_projection(x)  # [batch_size, seq_len, hidden_dim]
        K = self.key_projection(x)
        V = self.value_projection(x)

        # 重塑为多头格式
        Q = Q.reshape([batch_size, seq_len, self.num_heads, self.head_dim]).transpose([0, 2, 1, 3])
        K = K.reshape([batch_size, seq_len, self.num_heads, self.head_dim]).transpose([0, 2, 1, 3])
        V = V.reshape([batch_size, seq_len, self.num_heads, self.head_dim]).transpose([0, 2, 1, 3])

        # 计算注意力分数
        attention_scores = paddle.matmul(Q, K.transpose([0, 1, 3, 2])) / np.sqrt(self.head_dim)
        attention_weights = F.softmax(attention_scores, axis=-1)
        attention_weights = self.dropout(attention_weights)

        # 应用注意力权重
        attended_values = paddle.matmul(attention_weights, V)

        # 重塑回原始格式
        attended_values = attended_values.transpose([0, 2, 1, 3]).reshape([batch_size, seq_len, self.hidden_dim])

        # 输出投影
        output = self.output_projection(attended_values)

        return output, attention_weights


class MultiModalProtoNet(nn.Layer):
    """
    多模态ProtoNet模型
    集成多头注意力特征融合和原型网络分类
    """

    def __init__(self,
                 feature_dims: Dict[str, int],
                 hidden_dim: int = 128,
                 z_dim: int = 64,
                 num_heads: int = 4,
                 dropout_rate: float = 0.3):
        """
        初始化多模态ProtoNet

        Args:
            feature_dims: 各特征类型的维度字典
            hidden_dim: 隐藏层维度
            z_dim: 最终特征维度
            num_heads: 注意力头数
            dropout_rate: dropout比率
        """
        super(MultiModalProtoNet, self).__init__()

        self.feature_dims = feature_dims
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim

        # 多头注意力融合模块
        self.attention_fusion = MultiHeadAttentionFusion(
            feature_dims=feature_dims,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout_rate=dropout_rate
        )

        # 编码器 - 将融合特征映射到最终特征空间
        self.encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, z_dim),
            nn.LayerNorm(z_dim)
        )

        # 初始化距离度量权重
        self.dist_weights = create_parameter(
            shape=[3],
            dtype='float32',
            default_initializer=nn.initializer.Constant(1.0/3)
        )

        # 可学习的缩放因子
        self.learnable_scale = create_parameter(
            shape=[1],
            dtype='float32',
            default_initializer=nn.initializer.Constant(10.0)  # 增加初始缩放
        )

        # 温度参数用于控制softmax锐度
        self.temperature = create_parameter(
            shape=[1],
            dtype='float32',
            default_initializer=nn.initializer.Constant(0.1)  # 较小的温度使分布更锐利
        )

    def forward(self, features: Dict[str, paddle.Tensor]) -> Tuple[paddle.Tensor, Dict[str, paddle.Tensor]]:
        """
        前向传播

        Args:
            features: 特征字典

        Returns:
            Tuple[paddle.Tensor, Dict[str, paddle.Tensor]]: (编码特征, 注意力权重)
        """
        # 多头注意力融合
        fused_features, attention_weights = self.attention_fusion(features)

        # 编码到最终特征空间
        encoded_features = self.encoder(fused_features)

        return encoded_features, attention_weights

    def compute_multi_metric_distances(self, z_query: paddle.Tensor, z_proto: paddle.Tensor) -> paddle.Tensor:
        """
        计算多度量距离

        Args:
            z_query: 查询特征 [n_query, z_dim]
            z_proto: 原型特征 [n_class, z_dim]

        Returns:
            paddle.Tensor: 距离矩阵 [n_query, n_class]
        """
        # 计算欧氏距离
        euclidean = euclidean_dist(z_query, z_proto)

        # 计算曼哈顿距离
        manhattan = self._manhattan_dist(z_query, z_proto)

        # 计算余弦距离
        cosine = self._cosine_dist(z_query, z_proto)

        # 归一化各个距离
        euclidean = euclidean / (euclidean.max() + 1e-8)
        manhattan = manhattan / (manhattan.max() + 1e-8)
        cosine = cosine / (cosine.max() + 1e-8)

        # 使用softmax确保权重和为1
        weights = F.softmax(self.dist_weights, axis=0)

        # 加权聚合距离
        combined_dist = (weights[0] * euclidean +
                        weights[1] * manhattan +
                        weights[2] * cosine)

        return self.learnable_scale * combined_dist

    def _manhattan_dist(self, x: paddle.Tensor, y: paddle.Tensor) -> paddle.Tensor:
        """计算曼哈顿距离"""
        n = x.shape[0]
        m = y.shape[0]
        d = x.shape[1]
        x_expanded = x.unsqueeze(1).expand([n, m, d])
        y_expanded = y.unsqueeze(0).expand([n, m, d])
        return paddle.abs(x_expanded - y_expanded).sum(2)

    def _cosine_dist(self, x: paddle.Tensor, y: paddle.Tensor) -> paddle.Tensor:
        """计算余弦距离"""
        n = x.shape[0]
        m = y.shape[0]
        d = x.shape[1]
        x_expanded = x.unsqueeze(1).expand([n, m, d])
        y_expanded = y.unsqueeze(0).expand([n, m, d])
        cos_sim = F.cosine_similarity(x_expanded, y_expanded, axis=2)
        return 1 - cos_sim

    def loss(self, sample: Dict[str, paddle.Tensor]) -> Tuple[paddle.Tensor, Dict]:
        """
        计算损失函数

        Args:
            sample: 包含support和query数据的字典

        Returns:
            Tuple[paddle.Tensor, Dict]: (损失值, 统计信息)
        """
        # 获取support和query数据
        support_features = {}
        query_features = {}

        # 分离support和query特征
        for feature_type in self.feature_dims.keys():
            if f'{feature_type}_support' in sample:
                support_features[feature_type] = sample[f'{feature_type}_support']
            if f'{feature_type}_query' in sample:
                query_features[feature_type] = sample[f'{feature_type}_query']

        # 获取数据维度
        n_class = list(support_features.values())[0].shape[0]
        n_support = list(support_features.values())[0].shape[1]
        n_query = list(query_features.values())[0].shape[1]

        # 处理support set
        support_reshaped = {}
        for feature_type, features in support_features.items():
            support_reshaped[feature_type] = features.reshape([-1, features.shape[-1]])

        # 编码support特征
        z_support, support_attention = self.forward(support_reshaped)
        z_support = z_support.reshape([n_class, n_support, -1])
        z_proto = z_support.mean(axis=1)  # 计算原型

        # 处理query set
        query_reshaped = {}
        for feature_type, features in query_features.items():
            query_reshaped[feature_type] = features.reshape([-1, features.shape[-1]])

        # 编码query特征
        z_query, query_attention = self.forward(query_reshaped)

        # 计算多度量距离
        dists = self.compute_multi_metric_distances(z_query, z_proto)

        # 计算目标索引
        target_inds = paddle.arange(0, n_class).reshape([n_class, 1]).expand([n_class, n_query])
        target_inds = target_inds.reshape([-1])
        target_inds.stop_gradient = True

        # 计算损失和准确率 - 使用温度参数
        scaled_dists = -dists / paddle.abs(self.temperature)  # 使用温度缩放
        log_p_y = F.log_softmax(scaled_dists, axis=1)
        loss_val = F.nll_loss(log_p_y, target_inds)

        _, y_hat = paddle.topk(scaled_dists, k=1, axis=1)
        acc_val = paddle.equal(y_hat.squeeze(), target_inds).astype('float32').mean()

        return loss_val, {
            'loss': loss_val.item(),
            'acc': acc_val.item(),
            'dist_weights': self.dist_weights.numpy(),
            'temperature': self.temperature.numpy().item(),
            'scale': self.learnable_scale.numpy().item(),
            'support_attention': support_attention,
            'query_attention': query_attention
        }


class MultiModalConfig:
    """
    多模态配置类
    用于管理用户选择的特征类型和模式
    """

    def __init__(self):
        self.available_features = ['gait', 'pressure', 'coupling']
        self.available_modes = ['static', 'sliding']
        self.available_fusion_types = ['attention', 'concat', 'average']

        # 默认配置
        self.selected_features = ['gait']
        self.selected_mode = 'static'
        self.fusion_type = 'attention'

        # 优化后的模型参数
        self.hidden_dim = 128  # 增加隐藏层维度
        self.z_dim = 64        # 增加最终特征维度
        self.num_heads = 8     # 增加注意力头数
        self.dropout_rate = 0.3 # 适当增加dropout防止过拟合

    def set_features(self, features: List[str]):
        """设置选择的特征类型"""
        invalid_features = set(features) - set(self.available_features)
        if invalid_features:
            raise ValueError(f"无效的特征类型: {invalid_features}")
        self.selected_features = features

    def set_mode(self, mode: str):
        """设置特征模式"""
        if mode not in self.available_modes:
            raise ValueError(f"无效的模式: {mode}")
        self.selected_mode = mode

    def set_fusion_type(self, fusion_type: str):
        """设置融合类型"""
        if fusion_type not in self.available_fusion_types:
            raise ValueError(f"无效的融合类型: {fusion_type}")
        self.fusion_type = fusion_type

    def get_config_dict(self) -> Dict:
        """获取配置字典"""
        return {
            'selected_features': self.selected_features,
            'selected_mode': self.selected_mode,
            'fusion_type': self.fusion_type,
            'hidden_dim': self.hidden_dim,
            'z_dim': self.z_dim,
            'num_heads': self.num_heads,
            'dropout_rate': self.dropout_rate
        }

    def save_config(self, filepath: str):
        """保存配置到文件"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.get_config_dict(), f, indent=2, ensure_ascii=False)

    def load_config(self, filepath: str):
        """从文件加载配置"""
        with open(filepath, 'r', encoding='utf-8') as f:
            config = json.load(f)

        self.selected_features = config.get('selected_features', ['gait'])
        self.selected_mode = config.get('selected_mode', 'static')
        self.fusion_type = config.get('fusion_type', 'attention')
        self.hidden_dim = config.get('hidden_dim', 64)
        self.z_dim = config.get('z_dim', 32)
        self.num_heads = config.get('num_heads', 4)
        self.dropout_rate = config.get('dropout_rate', 0.3)
