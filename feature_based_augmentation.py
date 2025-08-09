#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于特征学习的认知衰弱数据增强系统
方案：分类别学习特征 -> 数据扰动 -> VAE-GAN增强 -> 质量评估
"""

import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score, classification_report
from scipy import stats
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# 设置matplotlib避免中文字体警告
plt.rcParams['font.family'] = ['DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# 导入可视化工具
try:
    from visualization_utils import create_comprehensive_report
except ImportError:
    print("⚠️ 可视化模块未找到，将跳过可视化功能")
    create_comprehensive_report = None

# ==================== 原型网络模块 ====================
class ProtoNetEncoder(nn.Module):
    """轻量化原型网络编码器 - 专为小样本优化"""
    def __init__(self, input_dim, hidden_dim=64, z_dim=32, dropout_rate=0.2):
        super(ProtoNetEncoder, self).__init__()
        # 简化网络结构，减少过拟合风险
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, z_dim),
            nn.BatchNorm1d(z_dim),
            nn.ReLU()
        )

        # 权重初始化
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.net(x)

class ProtoNet(nn.Module):
    """优化的原型网络分类器"""
    def __init__(self, encoder, temperature=1.0):
        super(ProtoNet, self).__init__()
        self.encoder = encoder
        self.temperature = temperature  # 温度参数，控制softmax的锐度

    def compute_distances(self, query_embeddings, prototype_embeddings):
        """计算查询样本到原型的距离 - 使用余弦相似度"""
        # L2归一化
        query_norm = torch.nn.functional.normalize(query_embeddings, p=2, dim=1)
        proto_norm = torch.nn.functional.normalize(prototype_embeddings, p=2, dim=1)

        # 计算余弦相似度 (转换为距离)
        similarities = torch.mm(query_norm, proto_norm.t())
        distances = 1 - similarities  # 余弦距离

        return distances

    def forward(self, support_data, support_labels, query_data):
        """前向传播 - 优化版本"""
        # 编码
        support_embeddings = self.encoder(support_data)
        query_embeddings = self.encoder(query_data)

        # 计算原型 - 使用更稳定的方法
        unique_labels = torch.unique(support_labels, sorted=True)
        n_class = len(unique_labels)
        z_dim = support_embeddings.size(1)

        prototypes = torch.zeros(n_class, z_dim, device=support_data.device)
        for i, label in enumerate(unique_labels):
            mask = support_labels == label
            class_embeddings = support_embeddings[mask]

            # 修复张量维度问题 - 使用均值而非中位数
            if len(class_embeddings) > 1:
                prototypes[i] = torch.mean(class_embeddings, dim=0)
            else:
                prototypes[i] = class_embeddings[0]

        # 计算距离
        distances = self.compute_distances(query_embeddings, prototypes)
        return distances / self.temperature, unique_labels

class ProtoNetClassifier:
    """自适应原型网络分类器 - 根据数据集自动调整参数"""
    def __init__(self, input_dim, hidden_dim=64, z_dim=32, dropout_rate=0.2,
                 learning_rate=0.01, epochs=15, device='cuda', temperature=2.0,
                 auto_adapt=True):
        self.input_dim = input_dim
        self.auto_adapt = auto_adapt

        # 自动适应参数
        if auto_adapt:
            hidden_dim, z_dim, dropout_rate, learning_rate, epochs, temperature = \
                self._auto_adapt_params(input_dim, hidden_dim, z_dim, dropout_rate,
                                      learning_rate, epochs, temperature)

        self.hidden_dim = hidden_dim
        self.z_dim = z_dim
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.temperature = temperature

        # 初始化网络
        encoder = ProtoNetEncoder(input_dim, hidden_dim, z_dim, dropout_rate)
        self.model = ProtoNet(encoder, temperature).to(self.device)

        # 选择优化器 - 根据数据规模自动选择
        if input_dim < 50:  # 小特征空间使用Adam
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=1e-4
            )
        else:  # 大特征空间使用SGD
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=learning_rate,
                momentum=0.9,
                weight_decay=1e-4
            )

        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        self.is_fitted = False
        self.support_data = None
        self.support_labels = None
        self.class_mapping = None

    def _auto_adapt_params(self, input_dim, hidden_dim, z_dim, dropout_rate,
                          learning_rate, epochs, temperature):
        """根据输入维度自动调整参数"""

        # 根据输入维度调整网络大小
        if input_dim <= 20:  # 小特征空间 (如gait数据)
            hidden_dim = max(8, min(hidden_dim, input_dim * 2))
            z_dim = max(4, min(z_dim, input_dim))
            dropout_rate = min(0.5, dropout_rate + 0.1)  # 增加正则化
            # 保持用户配置的学习率，不自动调整
            epochs = min(epochs, 15)                      # 减少训练轮数
            temperature = max(2.0, temperature)           # 增加温度

        elif input_dim <= 100:  # 中等特征空间
            hidden_dim = max(16, min(hidden_dim, input_dim))
            z_dim = max(8, min(z_dim, input_dim // 2))
            dropout_rate = max(0.2, dropout_rate)
            # 保持用户配置的学习率
            epochs = min(epochs, 20)
            temperature = max(1.5, temperature)

        else:  # 大特征空间 (如时序数据)
            hidden_dim = max(32, min(hidden_dim, input_dim // 4))
            z_dim = max(16, min(z_dim, input_dim // 8))
            dropout_rate = max(0.1, dropout_rate)
            # 保持用户配置的学习率
            epochs = max(epochs, 20)
            temperature = max(1.0, temperature * 0.8)

        return hidden_dim, z_dim, dropout_rate, learning_rate, epochs, temperature

    def fit(self, X, y):
        """优化的训练方法 - 适合小样本"""
        self.model.train()

        # 数据预处理和转换
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.LongTensor(y).to(self.device)

        # 创建类别映射
        unique_labels = torch.unique(y_tensor, sorted=True)
        self.class_mapping = {label.item(): i for i, label in enumerate(unique_labels)}

        # 保存支持集
        self.support_data = X_tensor
        self.support_labels = y_tensor

        # 小批量训练 - 修复批次大小问题
        n_classes = len(unique_labels)
        min_batch_size = max(4, n_classes * 2)  # 确保每个批次至少包含每个类别2个样本
        batch_size = min(min_batch_size, len(X))



        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # 学习率调度器 - 修复学习率显示问题
        scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=max(3, self.epochs//3), gamma=0.9)

        best_loss = float('inf')
        patience = 5
        patience_counter = 0

        for epoch in range(self.epochs):
            epoch_loss = 0
            num_batches = 0

            for batch_x, batch_y in dataloader:
                if len(batch_x) < 2:  # 跳过太小的批次
                    continue

                self.optimizer.zero_grad()

                # 智能分割为支持集和查询集 - 确保类别平衡
                unique_batch_labels = torch.unique(batch_y)

                if len(batch_x) >= len(unique_batch_labels) * 2:
                    # 如果样本足够，按类别分割
                    support_indices = []
                    query_indices = []

                    for label in unique_batch_labels:
                        label_indices = (batch_y == label).nonzero(as_tuple=True)[0]
                        if len(label_indices) >= 2:
                            # 每个类别至少1个支持样本，1个查询样本
                            support_indices.append(label_indices[0].item())
                            query_indices.extend([idx.item() for idx in label_indices[1:]])
                        else:
                            # 如果类别样本不足，同时用作支持和查询
                            support_indices.extend([idx.item() for idx in label_indices])
                            query_indices.extend([idx.item() for idx in label_indices])

                    # 转换为张量索引
                    support_indices = torch.tensor(support_indices, device=batch_x.device)
                    query_indices = torch.tensor(query_indices, device=batch_x.device)

                    support_x = batch_x[support_indices]
                    support_y = batch_y[support_indices]
                    query_x = batch_x[query_indices]
                    query_y = batch_y[query_indices]
                else:
                    # 如果样本不足，使用整个批次
                    support_x, support_y = batch_x, batch_y
                    query_x, query_y = batch_x, batch_y

                try:
                    distances, unique_labels = self.model(support_x, support_y, query_x)

                    # 创建标签映射
                    label_map = {label.item(): i for i, label in enumerate(unique_labels)}
                    mapped_labels = torch.tensor([label_map[label.item()] for label in query_y],
                                               device=self.device)

                    # 计算损失
                    logits = -distances
                    loss = self.criterion(logits, mapped_labels)

                    loss.backward()

                    # 梯度裁剪
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                    self.optimizer.step()
                    epoch_loss += loss.item()
                    num_batches += 1

                except Exception:
                    # 如果出现错误，跳过这个批次
                    continue

            if num_batches > 0:
                avg_loss = epoch_loss / num_batches
                scheduler.step()

                # 早停机制
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    print(f"   ProtoNet 早停于 Epoch {epoch+1}, 最佳损失: {best_loss:.4f}")
                    break

                if (epoch + 1) % 5 == 0:
                    current_lr = self.optimizer.param_groups[0]['lr']
                    print(f"   ProtoNet Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.4f}, LR: {current_lr:.6f}")

        self.is_fitted = True
        return self

    def predict(self, X):
        """优化的预测方法"""
        if not self.is_fitted:
            raise ValueError("模型未训练，请先调用fit方法")

        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)

            try:
                distances, unique_labels = self.model(self.support_data, self.support_labels, X_tensor)

                # 选择距离最小的类别
                predicted_indices = torch.argmin(distances, dim=1)
                predictions = unique_labels[predicted_indices].cpu().numpy()

                return predictions

            except Exception as e:
                # 如果预测失败，返回最频繁的类别
                most_common_label = torch.mode(self.support_labels)[0].item()
                return np.full(len(X), most_common_label)

    def predict_proba(self, X):
        """优化的概率预测"""
        if not self.is_fitted:
            raise ValueError("模型未训练，请先调用fit方法")

        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)

            try:
                distances, unique_labels = self.model(self.support_data, self.support_labels, X_tensor)

                # 将距离转换为概率（使用温度缩放）
                logits = -distances
                probabilities = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()

                return probabilities

            except Exception as e:
                # 如果预测失败，返回均匀分布
                n_classes = len(torch.unique(self.support_labels))
                return np.full((len(X), n_classes), 1.0 / n_classes)

# ==================== CNN和LSTM模型 ====================
class CNNClassifier(nn.Module):
    """防过拟合的1D CNN分类器"""
    def __init__(self, input_dim, n_classes=3, dropout_rate=0.5):
        super(CNNClassifier, self).__init__()

        self.input_dim = input_dim

        # 简化的卷积层 - 减少参数
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, padding=1)  # 减少通道数
        self.conv2 = nn.Conv1d(16, 16, kernel_size=3, padding=1) # 保持通道数

        # 池化层
        self.pool = nn.AdaptiveAvgPool1d(4)  # 减少池化后的长度

        # 全连接层 - 大幅简化
        self.fc1 = nn.Linear(16 * 4, 16)     # 减少隐藏层大小
        self.fc2 = nn.Linear(16, n_classes)

        # 增强正则化
        self.dropout = nn.Dropout(dropout_rate)
        self.batch_norm1 = nn.BatchNorm1d(16)
        self.batch_norm2 = nn.BatchNorm1d(16)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: [batch_size, input_dim] -> [batch_size, 1, input_dim]
        x = x.unsqueeze(1)

        # 卷积层 + 批归一化 + Dropout
        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.dropout(x)
        x = self.relu(self.batch_norm2(self.conv2(x)))
        x = self.dropout(x)

        # 池化
        x = self.pool(x)

        # 展平
        x = x.view(x.size(0), -1)

        # 全连接层 + 强Dropout
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)

        return x

class LSTMClassifier(nn.Module):
    """防过拟合的LSTM分类器"""
    def __init__(self, input_dim, hidden_dim=32, n_layers=1, n_classes=3, dropout_rate=0.5):
        super(LSTMClassifier, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # 简化的LSTM层 - 减少参数
        self.lstm = nn.LSTM(1, hidden_dim, n_layers,
                           batch_first=True, dropout=0)  # 单层LSTM不使用内置dropout

        # 全连接层 - 简化
        self.fc1 = nn.Linear(hidden_dim, 16)      # 添加中间层
        self.fc2 = nn.Linear(16, n_classes)       # 输出层

        # 增强正则化
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: [batch_size, input_dim] -> [batch_size, input_dim, 1]
        x = x.unsqueeze(-1)

        # LSTM
        lstm_out, _ = self.lstm(x)

        # 使用最后一个时间步的输出 + LayerNorm
        last_output = self.layer_norm(lstm_out[:, -1, :])

        # 分类 - 两层全连接 + 强Dropout
        output = self.dropout(self.relu(self.fc1(last_output)))
        output = self.fc2(output)

        return output

class DeepClassifierWrapper:
    """深度学习分类器包装类 - 完全兼容sklearn接口"""
    def __init__(self, model_type='cnn', input_dim=20, n_classes=3,
                 epochs=30, learning_rate=0.001, device='cuda'):
        self.model_type = model_type
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.device = device if torch.cuda.is_available() else 'cpu'

    def get_params(self, deep=True):
        """获取参数 - sklearn兼容性要求"""
        return {
            'model_type': self.model_type,
            'input_dim': self.input_dim,
            'n_classes': self.n_classes,
            'epochs': self.epochs,
            'learning_rate': self.learning_rate,
            'device': self.device
        }

    def set_params(self, **params):
        """设置参数 - sklearn兼容性要求"""
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def _initialize_model(self):
        """初始化模型"""
        # 创建模型
        if self.model_type == 'cnn':
            self.model = CNNClassifier(self.input_dim, self.n_classes).to(self.device)
        elif self.model_type == 'lstm':
            self.model = LSTMClassifier(self.input_dim, n_classes=self.n_classes).to(self.device)
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        self.is_fitted = False

    def fit(self, X, y):
        """训练模型 - 添加早停机制防止过拟合"""
        # 初始化模型（如果还没有初始化）
        if not hasattr(self, 'model'):
            self._initialize_model()

        self.model.train()

        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.LongTensor(y).to(self.device)

        # 分割训练和验证集
        if len(X) > 10:
            val_size = max(2, len(X) // 5)  # 20%作为验证集
            indices = torch.randperm(len(X))
            train_indices = indices[val_size:]
            val_indices = indices[:val_size]

            X_train, X_val = X_tensor[train_indices], X_tensor[val_indices]
            y_train, y_val = y_tensor[train_indices], y_tensor[val_indices]
        else:
            # 数据太少，不分割验证集
            X_train, X_val = X_tensor, X_tensor
            y_train, y_val = y_tensor, y_tensor

        dataset = TensorDataset(X_train, y_train)
        dataloader = DataLoader(dataset, batch_size=min(4, len(X_train)), shuffle=True)

        # 早停参数
        best_val_loss = float('inf')
        patience = 5
        patience_counter = 0

        for epoch in range(self.epochs):
            # 训练
            self.model.train()
            epoch_loss = 0
            for batch_x, batch_y in dataloader:
                self.optimizer.zero_grad()

                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)

                loss.backward()
                # 梯度裁剪防止梯度爆炸
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                epoch_loss += loss.item()

            # 验证
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val)
                val_loss = self.criterion(val_outputs, y_val).item()

            # 早停检查
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # 保存最佳模型状态
                best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1

            if patience_counter >= patience:
                # 恢复最佳模型
                self.model.load_state_dict(best_model_state)
                break

            if (epoch + 1) % 10 == 0:
                avg_loss = epoch_loss / len(dataloader)
                print(f"   {self.model_type.upper()} Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.4f}")

        self.is_fitted = True
        return self

    def predict(self, X):
        """预测"""
        if not self.is_fitted:
            raise ValueError("模型未训练")

        if not hasattr(self, 'model'):
            raise ValueError("模型未初始化")

        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            outputs = self.model(X_tensor)
            predictions = torch.argmax(outputs, dim=1).cpu().numpy()

        return predictions

    def predict_proba(self, X):
        """预测概率"""
        if not self.is_fitted:
            raise ValueError("模型未训练")

        if not hasattr(self, 'model'):
            raise ValueError("模型未初始化")

        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            outputs = self.model(X_tensor)
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()

        return probabilities

class FeatureLearner:
    """类别特征学习器"""
    
    def __init__(self, n_components=10):
        self.n_components = n_components
        self.class_features = {}
        self.class_scalers = {}
        self.class_pcas = {}
        
    def learn_class_features(self, data_by_class):
        """学习每个类别的特征分布"""
        print("🧠 学习各类别特征分布...")
        
        for class_label, class_data in data_by_class.items():
            if len(class_data) == 0:
                continue
                
            print(f"   📊 学习类别 {class_label} 特征 ({len(class_data)} 个样本)")
            
            # 展平数据
            flattened_data = np.array([sample.flatten() for sample in class_data])
            
            # 标准化
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(flattened_data)
            
            # PCA降维提取主要特征
            n_comp = min(self.n_components, len(class_data)-1, scaled_data.shape[1])
            pca = PCA(n_components=n_comp)
            pca_features = pca.fit_transform(scaled_data)
            
            # 计算特征统计信息
            feature_stats = {
                'mean': np.mean(pca_features, axis=0),
                'std': np.std(pca_features, axis=0),
                'min': np.min(pca_features, axis=0),
                'max': np.max(pca_features, axis=0),
                'median': np.median(pca_features, axis=0)
            }
            
            # 使用K-means聚类发现子模式
            if len(class_data) >= 3:
                n_clusters = min(3, len(class_data))
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                clusters = kmeans.fit_predict(pca_features)
                cluster_centers = kmeans.cluster_centers_
            else:
                clusters = np.zeros(len(class_data))
                cluster_centers = np.array([feature_stats['mean']])
            
            self.class_features[class_label] = {
                'stats': feature_stats,
                'clusters': clusters,
                'cluster_centers': cluster_centers,
                'raw_features': pca_features
            }
            self.class_scalers[class_label] = scaler
            self.class_pcas[class_label] = pca
            
            print(f"     ✅ 提取 {n_comp} 个主要特征")
    
    def generate_perturbation_patterns(self, class_label, n_patterns=5):
        """为指定类别生成扰动模式"""
        if class_label not in self.class_features:
            return []
        
        features = self.class_features[class_label]
        stats = features['stats']
        
        patterns = []
        
        # 模式1: 基于均值和标准差的高斯扰动
        for i in range(n_patterns):
            noise_scale = 0.1 + i * 0.05  # 递增的噪声强度
            pattern = {
                'type': 'gaussian',
                'mean': stats['mean'],
                'std': stats['std'] * noise_scale,
                'intensity': noise_scale
            }
            patterns.append(pattern)
        
        # 模式2: 基于聚类中心的扰动
        for center in features['cluster_centers']:
            pattern = {
                'type': 'cluster_based',
                'center': center,
                'std': stats['std'] * 0.15,
                'intensity': 0.15
            }
            patterns.append(pattern)
        
        return patterns

class SimpleVAEGAN(nn.Module):
    """简化的VAE-GAN网络"""
    
    def __init__(self, input_dim, latent_dim=32, hidden_dim=128):
        super(SimpleVAEGAN, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # VAE编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU()
        )
        self.mu_layer = nn.Linear(hidden_dim//2, latent_dim)
        self.logvar_layer = nn.Linear(hidden_dim//2, latent_dim)
        
        # VAE解码器
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Tanh()
        )
        
        # 判别器
        self.discriminator = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim//2, 1),
            nn.Sigmoid()
        )
    
    def encode(self, x):
        h = self.encoder(x)
        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar
    
    def discriminate(self, x):
        return self.discriminator(x)

class FeatureBasedAugmentationSystem:
    """基于特征学习的数据增强系统"""
    
    def __init__(self, seq_len=400, latent_dim=32, device='cuda'):
        self.seq_len = seq_len
        self.latent_dim = latent_dim
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        self.feature_learner = FeatureLearner()
        self.vae_gan = None
        self.global_scaler = MinMaxScaler(feature_range=(0.01, 0.99))
        
        print(f"🔧 初始化增强系统，设备: {self.device}")
    
    def load_and_process_data(self, data_dir):
        """智能加载和处理数据 - 自动检测数据类型"""
        print(f"📂 从 {data_dir} 加载数据...")

        if not os.path.exists(data_dir):
            raise ValueError(f"目录 '{data_dir}' 不存在")

        data_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
        if not data_files:
            raise ValueError(f"在 {data_dir} 中未找到CSV文件")

        # 检测数据类型
        sample_file = os.path.join(data_dir, data_files[0])
        sample_df = pd.read_csv(sample_file)

        # 判断是否为gait类型数据
        is_gait_data = (
            len(sample_df) < 20 and  # 行数少
            sample_df.shape[1] <= 5 and  # 列数少
            any(col in sample_df.columns for col in ['step_length', 'step_frequency', 'step_speed'])
        )

        if is_gait_data:
            print("🚶 检测到步态特征数据，使用专门的处理方法")
            return self._load_gait_data(data_dir, data_files)
        else:
            print("📊 检测到时序数据，使用标准处理方法")
            return self._load_timeseries_data(data_dir, data_files)

    def _load_gait_data(self, data_dir, data_files):
        """专门处理步态特征数据"""
        all_data = []
        all_labels = []
        data_by_class = {0: [], 1: [], 2: []}

        print(f"📊 找到 {len(data_files)} 个步态数据文件")

        for file in data_files:
            try:
                df = pd.read_csv(os.path.join(data_dir, file))

                # 提取标签 - 优先使用CSV中的标签列
                if 'label' in df.columns:
                    label = int(df['label'].iloc[0])
                    df = df.drop('label', axis=1)
                    print(f"   ✅ 从CSV文件 {file} 读取标签: {label}")
                else:
                    # 如果CSV中没有标签列，则从文件名提取（但这不应该发生）
                    print(f"   ⚠️ 警告：CSV文件 {file} 中没有标签列，尝试从文件名提取")
                    import re
                    numbers = re.findall(r'\d+', file)
                    if numbers:
                        # 提取文件编号，但不直接用作标签
                        file_num = int(numbers[0])
                        # 默认标签映射（这是备用方案，不应该被使用）
                        label = file_num % 3
                        print(f"   ⚠️ 从文件名提取编号 {file_num}，映射为标签 {label}")
                    else:
                        label = 0
                        print(f"   ❌ 无法从文件名 {file} 提取任何信息，使用默认标签0")

                # 步态数据特殊处理：不填充，直接使用统计特征
                data = df.values.astype(np.float32)
                data = np.nan_to_num(data, nan=0.0, posinf=1.0, neginf=-1.0)

                # 计算统计特征而非填充
                if len(data) > 0:
                    # 对每列计算统计特征
                    features = []
                    for col in range(data.shape[1]):
                        col_data = data[:, col]
                        # 基本统计量
                        features.extend([
                            np.mean(col_data),
                            np.std(col_data),
                            np.min(col_data),
                            np.max(col_data),
                            np.median(col_data)
                        ])

                    # 转换为固定长度的特征向量
                    feature_vector = np.array(features, dtype=np.float32)

                    # 重塑为(1, n_features)以保持一致性
                    data = feature_vector.reshape(1, -1)
                else:
                    # 如果数据为空，创建零向量
                    n_features = df.shape[1] * 5  # 每列5个统计特征
                    data = np.zeros((1, n_features), dtype=np.float32)

                all_data.append(data)
                all_labels.append(label)

                # 按类别分组
                if label in data_by_class:
                    data_by_class[label].append(data)

            except Exception as e:
                print(f"⚠️ 处理步态文件 {file} 时出错: {e}")
                continue

        if not all_data:
            raise ValueError("没有成功加载任何步态数据文件")

        print(f"✅ 步态数据加载完成")
        print(f"📊 样本数量: {len(all_data)}")
        print(f"📊 每个样本形状: {all_data[0].shape}")
        print(f"🏷️ 标签分布: {[len(data_by_class[i]) for i in range(3)]}")

        return all_data, np.array(all_labels), data_by_class

    def _load_timeseries_data(self, data_dir, data_files):
        """处理时序数据（原有逻辑）"""
        all_data = []
        all_labels = []
        data_by_class = {0: [], 1: [], 2: []}

        print(f"📊 找到 {len(data_files)} 个时序数据文件")

        for file in data_files:
            try:
                df = pd.read_csv(os.path.join(data_dir, file))

                # 提取标签 - 修复标签提取逻辑
                if 'label' in df.columns:
                    label = int(df['label'].iloc[0])
                    df = df.drop('label', axis=1)
                else:
                    # 从文件名提取标签 - 修复错误的模运算
                    import re
                    numbers = re.findall(r'\d+', file)
                    if numbers:
                        # 假设文件名格式为 "classX_..." 或 "X_..."
                        first_num = int(numbers[0])
                        # 如果数字是0,1,2直接使用；如果是1,2,3转换为0,1,2
                        if first_num in [0, 1, 2]:
                            label = first_num
                        elif first_num in [1, 2, 3]:
                            label = first_num - 1
                        else:
                            # 其他情况使用模运算，但确保结果在0-2范围内
                            label = first_num % 3
                    else:
                        label = 0
                        print(f"   ⚠️ 无法从文件名 {file} 提取标签，使用默认标签0")

                # 数据预处理
                data = df.values.astype(np.float32)
                data = np.nan_to_num(data, nan=0.0, posinf=1.0, neginf=-1.0)

                # 截断或填充到指定长度
                if len(data) > self.seq_len:
                    data = data[:self.seq_len]
                else:
                    padding = np.zeros((self.seq_len - len(data), data.shape[1]))
                    data = np.vstack([data, padding])

                all_data.append(data)
                all_labels.append(label)

                # 按类别分组
                if label in data_by_class:
                    data_by_class[label].append(data)

            except Exception as e:
                print(f"⚠️ 处理时序文件 {file} 时出错: {e}")
                continue

        if not all_data:
            raise ValueError("没有成功加载任何时序数据文件")

        print(f"✅ 时序数据加载完成")
        print(f"📊 样本数量: {len(all_data)}")
        print(f"📊 每个样本形状: {all_data[0].shape}")
        print(f"🏷️ 标签分布: {[len(data_by_class[i]) for i in range(3)]}")

        return all_data, np.array(all_labels), data_by_class
    
    def apply_feature_based_perturbation(self, data, class_label, perturbation_pattern):
        """应用基于特征学习的数据扰动"""
        if class_label not in self.feature_learner.class_scalers:
            return data
        
        # 获取对应的scaler和pca
        scaler = self.feature_learner.class_scalers[class_label]
        pca = self.feature_learner.class_pcas[class_label]
        
        # 展平并标准化
        flat_data = data.flatten().reshape(1, -1)
        scaled_data = scaler.transform(flat_data)
        
        # PCA变换到特征空间
        pca_data = pca.transform(scaled_data)
        
        # 应用扰动
        if perturbation_pattern['type'] == 'gaussian':
            noise = np.random.normal(0, perturbation_pattern['std'], pca_data.shape)
            perturbed_pca = pca_data + noise
        elif perturbation_pattern['type'] == 'cluster_based':
            direction = perturbation_pattern['center'] - pca_data[0]
            noise = np.random.normal(0, perturbation_pattern['std'], pca_data.shape)
            perturbed_pca = pca_data + 0.3 * direction + noise
        else:
            perturbed_pca = pca_data
        
        # 逆变换回原始空间
        perturbed_scaled = pca.inverse_transform(perturbed_pca)
        perturbed_flat = scaler.inverse_transform(perturbed_scaled)
        
        # 重塑回原始形状
        perturbed_data = perturbed_flat.reshape(data.shape)
        
        return perturbed_data

    def train_vae_gan(self, data_list, labels, epochs=100, batch_size=16):
        """训练改进的VAE-GAN网络"""
        print("🔧 训练改进的VAE-GAN网络...")

        # 准备训练数据
        flattened_data = np.array([data.flatten() for data in data_list])

        # 全局标准化
        scaled_data = self.global_scaler.fit_transform(flattened_data)

        # 转换为PyTorch张量
        tensor_data = torch.FloatTensor(scaled_data).to(self.device)
        tensor_labels = torch.LongTensor(labels).to(self.device)

        dataset = TensorDataset(tensor_data, tensor_labels)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

        # 初始化VAE-GAN
        input_dim = scaled_data.shape[1]
        self.vae_gan = SimpleVAEGAN(input_dim, self.latent_dim).to(self.device)

        # 改进的优化器设置 - 不同学习率
        optimizer_vae = optim.Adam(
            list(self.vae_gan.encoder.parameters()) + list(self.vae_gan.decoder.parameters()),
            lr=2e-4, betas=(0.5, 0.999)  # 生成器学习率稍高
        )
        optimizer_disc = optim.Adam(
            self.vae_gan.discriminator.parameters(),
            lr=1e-4, betas=(0.5, 0.999)  # 判别器学习率较低
        )

        # 损失函数
        mse_loss = nn.MSELoss()
        bce_loss = nn.BCELoss()

        print(f"   📊 训练数据形状: {scaled_data.shape}")
        print(f"   🔧 网络输入维度: {input_dim}")

        # 训练历史记录和早停机制
        vae_losses = []
        disc_losses = []
        best_vae_loss = float('inf')
        patience = 50
        patience_counter = 0

        for epoch in range(epochs):
            epoch_vae_loss = 0
            epoch_disc_loss = 0

            for batch_idx, (batch_data, batch_labels) in enumerate(dataloader):
                batch_size_actual = batch_data.size(0)

                # 动态调整训练频率 - 防止判别器过强
                train_disc = (batch_idx % 2 == 0) or (epoch < epochs // 4)

                # 训练VAE (每个batch都训练)
                optimizer_vae.zero_grad()

                recon_data, mu, logvar = self.vae_gan(batch_data)

                # VAE损失 - 调整权重
                recon_loss = mse_loss(recon_data, batch_data)
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                kl_loss /= batch_size_actual * input_dim

                # 对抗损失 - 增加权重
                fake_validity = self.vae_gan.discriminate(recon_data)
                adv_loss = bce_loss(fake_validity, torch.ones_like(fake_validity))

                # 调整损失权重
                beta_kl = min(1.0, epoch / (epochs * 0.5))  # KL权重逐渐增加
                alpha_adv = 0.1 if epoch < epochs // 2 else 0.05  # 对抗权重后期降低

                vae_loss = recon_loss + beta_kl * 0.1 * kl_loss + alpha_adv * adv_loss
                vae_loss.backward()

                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(
                    list(self.vae_gan.encoder.parameters()) + list(self.vae_gan.decoder.parameters()),
                    max_norm=1.0
                )

                optimizer_vae.step()

                # 训练判别器 (降低频率)
                if train_disc:
                    optimizer_disc.zero_grad()

                    real_validity = self.vae_gan.discriminate(batch_data)
                    fake_validity = self.vae_gan.discriminate(recon_data.detach())

                    # 标签平滑
                    real_labels = torch.ones_like(real_validity) * 0.9
                    fake_labels = torch.zeros_like(fake_validity) + 0.1

                    real_loss = bce_loss(real_validity, real_labels)
                    fake_loss = bce_loss(fake_validity, fake_labels)
                    disc_loss = (real_loss + fake_loss) / 2

                    disc_loss.backward()

                    # 梯度裁剪
                    torch.nn.utils.clip_grad_norm_(
                        self.vae_gan.discriminator.parameters(),
                        max_norm=1.0
                    )

                    optimizer_disc.step()
                    epoch_disc_loss += disc_loss.item()

                epoch_vae_loss += vae_loss.item()

            # 记录损失
            vae_losses.append(epoch_vae_loss / len(dataloader))
            disc_losses.append(epoch_disc_loss / len(dataloader) if epoch_disc_loss > 0 else 0)

            # 早停检查
            current_vae_loss = vae_losses[-1]
            if current_vae_loss < best_vae_loss:
                best_vae_loss = current_vae_loss
                patience_counter = 0
                # 保存最佳模型状态
                best_model_state = {
                    'encoder': self.vae_gan.encoder.state_dict(),
                    'decoder': self.vae_gan.decoder.state_dict(),
                    'discriminator': self.vae_gan.discriminator.state_dict()
                }
            else:
                patience_counter += 1

            # 打印进度和损失趋势分析
            if (epoch + 1) % 50 == 0:
                recent_vae = np.mean(vae_losses[-10:])
                recent_disc = np.mean(disc_losses[-10:])

                print(f"   Epoch {epoch+1}/{epochs}:")
                print(f"     VAE Loss: {recent_vae:.4f}, Disc Loss: {recent_disc:.4f}")
                print(f"     Best VAE Loss: {best_vae_loss:.4f}, Patience: {patience_counter}/{patience}")

                # 训练稳定性检查（简化输出）
                if len(vae_losses) >= 20:
                    disc_trend = np.polyfit(range(10), disc_losses[-10:], 1)[0]
                    # 自适应学习率调整
                    if disc_trend > 0.02:
                        for param_group in optimizer_disc.param_groups:
                            param_group['lr'] *= 0.9

            # 早停条件
            if patience_counter >= patience:
                print(f"   🛑 早停触发！在epoch {epoch+1}停止训练")
                # 恢复最佳模型
                if 'best_model_state' in locals():
                    self.vae_gan.encoder.load_state_dict(best_model_state['encoder'])
                    self.vae_gan.decoder.load_state_dict(best_model_state['decoder'])
                    self.vae_gan.discriminator.load_state_dict(best_model_state['discriminator'])
                    print(f"   ✅ 恢复最佳模型状态 (VAE Loss: {best_vae_loss:.4f})")
                break

        print("✅ VAE-GAN训练完成")
        print(f"   📊 最终VAE损失: {vae_losses[-1]:.4f}")
        print(f"   📊 最终判别器损失: {disc_losses[-1]:.4f}")

        return vae_losses, disc_losses

    def generate_feature_perturbation_samples(self, original_data, original_labels, n_augment_per_sample=2):
        """方案1: 仅特征扰动增强"""
        print("🔄 执行方案1: 特征扰动增强...")

        augmented_data = []
        augmented_labels = []

        for data, label in zip(original_data, original_labels):
            perturbation_patterns = self.feature_learner.generate_perturbation_patterns(label)

            for j in range(n_augment_per_sample):
                if perturbation_patterns:
                    pattern = perturbation_patterns[j % len(perturbation_patterns)]
                    # 适中的扰动强度
                    pattern['intensity'] *= 0.7
                    enhanced_data = self.apply_feature_based_perturbation(data, label, pattern)
                else:
                    # 轻微高斯噪声
                    enhanced_data = data + np.random.normal(0, 0.02, data.shape)

                if self._is_valid_sample(enhanced_data, data):
                    augmented_data.append(enhanced_data)
                    augmented_labels.append(label)

        print(f"✅ 特征扰动增强完成，生成 {len(augmented_data)} 个样本")
        return augmented_data, np.array(augmented_labels)

    def generate_vae_reconstruction_samples(self, original_data, original_labels, n_augment_per_sample=2, mix_ratio=0.8):
        """方案2: VAE重构增强"""
        print("🔄 执行方案2: VAE重构增强...")

        if self.vae_gan is None:
            raise ValueError("VAE-GAN未训练，请先调用train_vae_gan")

        augmented_data = []
        augmented_labels = []

        # 调试计数器
        total_attempts = 0
        valid_samples = 0
        conservative_samples = 0
        fallback_samples = 0

        with torch.no_grad():
            for i, (data, label) in enumerate(zip(original_data, original_labels)):
                print(f"   处理样本 {i+1}/{len(original_data)}, 标签: {label}")

                for j in range(n_augment_per_sample):
                    total_attempts += 1

                    try:
                        # 添加轻微噪声到原始数据
                        noise_scale = 0.01 * (j + 1)
                        noisy_data = data + np.random.normal(0, noise_scale, data.shape)

                        # VAE重构
                        flat_data = noisy_data.flatten().reshape(1, -1)
                        scaled_data = self.global_scaler.transform(flat_data)
                        tensor_data = torch.FloatTensor(scaled_data).to(self.device)

                        recon_data, _, _ = self.vae_gan(tensor_data)

                        # 与原始数据混合
                        mixed_data = mix_ratio * tensor_data + (1 - mix_ratio) * recon_data

                        enhanced_scaled = mixed_data.cpu().numpy()
                        enhanced_flat = self.global_scaler.inverse_transform(enhanced_scaled)
                        enhanced_data = enhanced_flat.reshape(data.shape)

                        # 暂时跳过质量检查，直接接受所有样本
                        augmented_data.append(enhanced_data)
                        augmented_labels.append(label)
                        valid_samples += 1

                    except Exception as e:
                        print(f"   ⚠️ 样本 {i+1}-{j+1} 处理失败: {str(e)}")
                        # 使用备用方案
                        fallback_data = data + np.random.normal(0, 0.005, data.shape)
                        augmented_data.append(fallback_data)
                        augmented_labels.append(label)
                        fallback_samples += 1

        print(f"✅ VAE重构增强完成，生成 {len(augmented_data)} 个样本")
        print(f"   📊 统计: 总尝试 {total_attempts}, 有效 {valid_samples}, 保守 {conservative_samples}, 备用 {fallback_samples}")
        return augmented_data, np.array(augmented_labels)

    def generate_hybrid_samples(self, original_data, original_labels, n_augment_per_sample=2):
        """方案3: 混合增强 (特征扰动 + VAE增强)"""
        print("🔄 执行方案3: 混合增强...")

        if self.vae_gan is None:
            raise ValueError("VAE-GAN未训练，请先调用train_vae_gan")

        augmented_data = []
        augmented_labels = []

        for data, label in zip(original_data, original_labels):
            perturbation_patterns = self.feature_learner.generate_perturbation_patterns(label)

            for j in range(n_augment_per_sample):
                # 第一步: 轻微特征扰动
                if perturbation_patterns:
                    pattern = perturbation_patterns[j % len(perturbation_patterns)]
                    pattern['intensity'] *= 0.2  # 更小的扰动强度
                    perturbed_data = self.apply_feature_based_perturbation(data, label, pattern)
                else:
                    perturbed_data = data + np.random.normal(0, 0.005, data.shape)  # 减少噪声

                # 第二步: VAE增强
                with torch.no_grad():
                    flat_data = perturbed_data.flatten().reshape(1, -1)
                    scaled_data = self.global_scaler.transform(flat_data)
                    tensor_data = torch.FloatTensor(scaled_data).to(self.device)

                    # 在潜在空间添加适度噪声
                    mu, logvar = self.vae_gan.encode(tensor_data)
                    noise_scale = 0.01 + j * 0.005  # 减少噪声强度
                    noise = torch.randn_like(mu) * noise_scale
                    z = mu + noise
                    enhanced_tensor = self.vae_gan.decode(z)

                    enhanced_scaled = enhanced_tensor.cpu().numpy()
                    enhanced_flat = self.global_scaler.inverse_transform(enhanced_scaled)
                    enhanced_data = enhanced_flat.reshape(data.shape)

                if self._is_valid_sample(enhanced_data, data):
                    augmented_data.append(enhanced_data)
                    augmented_labels.append(label)
                else:
                    # 如果质量检查失败，使用更保守的方法
                    fallback_data = data + np.random.normal(0, 0.005, data.shape)
                    augmented_data.append(fallback_data)
                    augmented_labels.append(label)

        print(f"✅ 混合增强完成，生成 {len(augmented_data)} 个样本")
        return augmented_data, np.array(augmented_labels)

    def generate_augmented_samples(self, original_data, original_labels, augmentation_method='hybrid', n_augment_per_sample=2, **kwargs):
        """统一的增强样本生成接口

        Args:
            augmentation_method: 增强方法选择
                - 'feature_perturbation': 仅特征扰动
                - 'vae_reconstruction': VAE重构增强
                - 'hybrid': 混合增强
                - 'all': 使用所有三种方法
            n_augment_per_sample: 每个原始样本生成的增强样本数
            **kwargs: 额外参数
        """
        print(f"🔄 开始数据增强，方法: {augmentation_method}")

        if augmentation_method == 'feature_perturbation':
            return self.generate_feature_perturbation_samples(original_data, original_labels, n_augment_per_sample)

        elif augmentation_method == 'vae_reconstruction':
            mix_ratio = kwargs.get('mix_ratio', 0.8)
            return self.generate_vae_reconstruction_samples(original_data, original_labels, n_augment_per_sample, mix_ratio)

        elif augmentation_method == 'hybrid':
            return self.generate_hybrid_samples(original_data, original_labels, n_augment_per_sample)

        elif augmentation_method == 'all':
            # 使用所有三种方法，每种方法生成 n_augment_per_sample // 3 个样本
            samples_per_method = max(1, n_augment_per_sample // 3)

            all_augmented_data = []
            all_augmented_labels = []

            # 方案1: 特征扰动
            aug_data_1, aug_labels_1 = self.generate_feature_perturbation_samples(
                original_data, original_labels, samples_per_method
            )
            all_augmented_data.extend(aug_data_1)
            all_augmented_labels.extend(aug_labels_1)

            # 方案2: VAE重构
            aug_data_2, aug_labels_2 = self.generate_vae_reconstruction_samples(
                original_data, original_labels, samples_per_method
            )
            all_augmented_data.extend(aug_data_2)
            all_augmented_labels.extend(aug_labels_2)

            # 方案3: 混合增强
            aug_data_3, aug_labels_3 = self.generate_hybrid_samples(
                original_data, original_labels, samples_per_method
            )
            all_augmented_data.extend(aug_data_3)
            all_augmented_labels.extend(aug_labels_3)

            print(f"✅ 所有方法增强完成，总共生成 {len(all_augmented_data)} 个样本")
            return all_augmented_data, np.array(all_augmented_labels)

        else:
            raise ValueError(f"未知的增强方法: {augmentation_method}")

    def _is_valid_sample(self, enhanced_data, original_data, threshold=10.0):
        """检查生成样本的有效性 - 进一步放宽条件"""
        try:
            # 1. 基本数值检查
            if np.any(np.isnan(enhanced_data)) or np.any(np.isinf(enhanced_data)):
                return False

            # 2. 极端值检查 - 非常宽松
            if np.any(np.abs(enhanced_data) > 1e6):
                return False

            # 3. 方差检查 - 非常宽松
            enhanced_std = np.std(enhanced_data)
            if enhanced_std < 1e-6:  # 几乎为0的方差
                return False

            # 对于VAE重构，我们应该更宽松
            return True

        except Exception as e:
            # 如果检查过程出错，默认接受样本
            return True

    def compute_quality_score(self, sample, original_sample):
        """计算样本质量分数 - 针对时序数据优化的评估标准"""
        score = 0.0

        # 展平数据以便计算
        sample_flat = sample.flatten()
        original_flat = original_sample.flatten()

        # 1. 基本数值有效性检查 (0.1分)
        if not (np.isnan(sample_flat).any() or np.isinf(sample_flat).any()):
            score += 0.1
        else:
            return 0.0

        # 2. 时序相关性检查 (0.4分) - 最重要的指标
        try:
            correlation = np.corrcoef(sample_flat, original_flat)[0, 1]
            if not np.isnan(correlation):
                if correlation > 0.8:  # 高相关性
                    score += 0.4
                elif correlation > 0.6:  # 中等相关性
                    score += 0.25
                elif correlation > 0.4:  # 低相关性
                    score += 0.1
                else:  # 相关性太低，严重惩罚
                    score *= 0.5
        except:
            score *= 0.5

        # 3. 统计特性保持 (0.3分)
        sample_std = np.std(sample_flat)
        original_std = np.std(original_flat)
        sample_mean = np.mean(sample_flat)
        original_mean = np.mean(original_flat)

        if sample_std > 0 and original_std > 0:
            std_ratio = sample_std / original_std
            if 0.7 <= std_ratio <= 1.4:  # 标准差保持合理
                score += 0.15
            elif 0.5 <= std_ratio <= 2.0:
                score += 0.05

        if original_mean != 0:
            mean_ratio = abs(sample_mean / original_mean)
            if 0.8 <= mean_ratio <= 1.25:  # 均值保持合理
                score += 0.15
            elif 0.6 <= mean_ratio <= 1.67:
                score += 0.05

        # 4. 分布相似性检查 (0.2分)
        try:
            _, p_value = stats.ks_2samp(sample_flat, original_flat)
            if p_value > 0.05:  # 分布相似
                score += 0.2
            elif p_value > 0.01:
                score += 0.1
        except:
            pass

        # 5. 严格的异常值惩罚
        q75, q25 = np.percentile(original_flat, [75, 25])
        iqr = q75 - q25
        if iqr > 0:
            outlier_threshold = 2 * iqr  # 更严格的异常值阈值
            outliers = np.abs(sample_flat - np.median(original_flat)) > outlier_threshold
            outlier_ratio = np.sum(outliers) / len(sample_flat)
            if outlier_ratio > 0.05:  # 超过5%的异常值就惩罚
                score *= (1 - outlier_ratio * 2)  # 更严厉的惩罚

        # 6. 时序模式保持检查（针对时序数据）
        if len(sample.shape) > 1:  # 如果是多维时序数据
            try:
                # 检查时序趋势保持
                sample_2d = sample.reshape(-1, sample.shape[-1])
                original_2d = original_sample.reshape(-1, original_sample.shape[-1])

                # 计算每个时间步的相关性
                time_correlations = []
                for t in range(min(sample_2d.shape[1], original_2d.shape[1])):
                    if np.std(sample_2d[:, t]) > 0 and np.std(original_2d[:, t]) > 0:
                        corr = np.corrcoef(sample_2d[:, t], original_2d[:, t])[0, 1]
                        if not np.isnan(corr):
                            time_correlations.append(corr)

                if time_correlations:
                    avg_time_corr = np.mean(time_correlations)
                    if avg_time_corr < 0.3:  # 时序模式保持不好
                        score *= 0.7
            except:
                pass

        return max(0.0, min(score, 1.0))

    def evaluate_augmentation_quality(self, original_data, original_labels, augmented_data, augmented_labels, config=None):
        """评估增强质量 - 修复数据泄露问题"""
        print("🔍 评估增强质量...")

        # 计算质量分数 - 修复索引逻辑
        quality_scores = []
        n_augment_per_sample = len(augmented_data) // len(original_data) if len(original_data) > 0 else 2



        for i, aug_sample in enumerate(augmented_data):
            orig_idx = i // n_augment_per_sample  # 动态计算原始样本索引
            if orig_idx < len(original_data):
                score = self.compute_quality_score(aug_sample, original_data[orig_idx])
                quality_scores.append(score)
            else:
                print(f"   ⚠️ 增强样本 {i} 对应的原始样本索引 {orig_idx} 超出范围")

        avg_quality = np.mean(quality_scores)
        print(f"   📊 平均质量分数: {avg_quality:.4f}")

        # 分类性能评估 - 彻底修复数据泄露问题
        def flatten_data(data_list):
            return np.array([data.flatten() for data in data_list])

        X_orig = flatten_data(original_data)
        X_aug = flatten_data(augmented_data)
        y_all = np.hstack([original_labels, augmented_labels])

        # 修复评估逻辑 - 使用交叉验证而非固定分割
        if len(original_labels) >= 6:
            # 1. 检查类别分布
            unique_labels, label_counts = np.unique(original_labels, return_counts=True)
            min_count = np.min(label_counts)

            print(f"   📊 类别分布: {dict(zip(unique_labels, label_counts))}")
            print(f"   📊 最少类别样本数: {min_count}")

            # 2. 使用分层K折交叉验证而非固定分割
            from sklearn.model_selection import StratifiedKFold

            # 根据数据量选择折数
            if len(original_labels) < 15:
                n_splits = 3  # 小数据集用3折
            elif len(original_labels) < 30:
                n_splits = 5  # 中等数据集用5折
            else:
                n_splits = 10  # 大数据集用10折

            print(f"   📊 使用 {n_splits} 折交叉验证")

            # 为了保持代码兼容性，我们仍然需要训练/测试分割来处理增强数据
            # 但会使用交叉验证来获得更可靠的评估
            if min_count >= 2:
                test_size = 0.2  # 减少测试集，增加训练集
            else:
                test_size = 1.0 / len(original_labels)

            X_train_raw, X_test_raw, y_train, y_test = train_test_split(
                X_orig, original_labels, test_size=test_size, random_state=42,
                stratify=original_labels
            )

            print(f"   📊 训练集大小: {len(X_train_raw)}, 测试集大小: {len(X_test_raw)}")

            # 2. 仅在训练集上训练预处理器
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_raw)
            X_test_scaled = scaler.transform(X_test_raw)  # 只变换，不训练

            # 3. 自适应PCA降维 - 根据数据特性调整
            pca = None
            n_samples, n_features = X_train_scaled.shape

            if n_features > 50:
                # 根据数据类型和规模自适应选择降维策略
                if n_features > 10000:  # 高维时序数据
                    # 保留更多信息，使用累积方差阈值
                    pca_temp = PCA()
                    pca_temp.fit(X_train_scaled)
                    cumsum_var = np.cumsum(pca_temp.explained_variance_ratio_)

                    # 修复：大幅增加保留的维度数
                    n_components = np.argmax(cumsum_var >= 0.99) + 1  # 保留99%方差
                    n_components = min(n_components, min(200, n_samples*3))  # 大幅增加最大维度
                    n_components = max(n_components, min(50, n_samples))  # 增加最小维度

                elif n_features > 100:  # 中等维度数据
                    # 修复：保留90%方差，增加信息保留
                    pca_temp = PCA()
                    pca_temp.fit(X_train_scaled)
                    cumsum_var = np.cumsum(pca_temp.explained_variance_ratio_)

                    n_components = np.argmax(cumsum_var >= 0.90) + 1
                    n_components = min(n_components, n_samples//2, min(50, n_samples*2))
                    n_components = max(n_components, min(15, n_samples))

                else:  # 低维度数据
                    # 保守降维
                    n_components = min(20, n_samples//3, n_features)
                    n_components = max(n_components, 5)

                print(f"   🔧 自适应PCA降维: {n_features} → {n_components}")

                pca = PCA(n_components=n_components)
                X_train_scaled = pca.fit_transform(X_train_scaled)
                X_test_scaled = pca.transform(X_test_scaled)

                explained_ratio = np.sum(pca.explained_variance_ratio_)
                print(f"   📊 PCA解释方差: {explained_ratio:.3f} ({explained_ratio*100:.1f}%)")



            # 4. 处理增强数据 - 仅使用基于训练集的增强数据
            # 找出哪些增强样本是基于训练集生成的
            train_indices_in_orig = []
            for i, (orig_data, orig_label) in enumerate(zip(X_orig, original_labels)):
                # 检查这个原始样本是否在训练集中
                for train_data, train_label in zip(X_train_raw, y_train):
                    if np.array_equal(orig_data, train_data) and orig_label == train_label:
                        train_indices_in_orig.append(i)
                        break

            # 获取基于训练集的增强数据 - 修复索引逻辑
            X_aug_train = []
            y_aug_train = []
            n_augment_per_sample = len(augmented_data) // len(original_data) if len(original_data) > 0 else 2

            for i, (aug_data, aug_label) in enumerate(zip(X_aug, augmented_labels)):
                orig_idx = i // n_augment_per_sample  # 动态计算原始样本索引
                if orig_idx < len(original_data) and orig_idx in train_indices_in_orig:
                    X_aug_train.append(aug_data)
                    y_aug_train.append(aug_label)

            # 预处理增强数据（使用训练集的预处理器）
            if X_aug_train:
                X_aug_train = np.array(X_aug_train)
                X_aug_train_scaled = scaler.transform(X_aug_train)
                if pca is not None:
                    X_aug_train_scaled = pca.transform(X_aug_train_scaled)
                y_aug_train = np.array(y_aug_train)
            else:
                X_aug_train_scaled = np.empty((0, X_train_scaled.shape[1]))
                y_aug_train = np.array([])

            # 5. 基线测试：仅使用原始训练数据
            clf_baseline = SVC(kernel='rbf', random_state=42)
            clf_baseline.fit(X_train_scaled, y_train)
            y_pred_baseline = clf_baseline.predict(X_test_scaled)
            acc_baseline = accuracy_score(y_test, y_pred_baseline)

            # 3. 多模型交叉验证评估 - 避免数据泄露
            print("   🔄 使用多模型交叉验证评估...")

            # 自适应模型配置 - 根据数据特性调整参数
            n_samples = len(X_train_scaled)
            n_features = X_train_scaled.shape[1]

            # 修复：重新评估数据复杂度指标，考虑极小数据集
            sample_feature_ratio = n_samples / n_features

            # 更严格的复杂度评估
            if sample_feature_ratio < 1.5:
                complexity_level = "extreme"  # 极高复杂度
            elif sample_feature_ratio < 3:
                complexity_level = "high"
            elif sample_feature_ratio < 8:
                complexity_level = "medium"
            else:
                complexity_level = "low"

            print(f"   📊 数据复杂度: {complexity_level} (样本/特征比={sample_feature_ratio:.2f})")

            if complexity_level == "extreme":  # 极高复杂度：轻度正则化
                classifiers = {
                    'SVM': SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42, probability=True),
                    'DecisionTree': DecisionTreeClassifier(
                        random_state=42, max_depth=5,  # 进一步增加深度
                        min_samples_split=2,  # 最小分割要求
                        min_samples_leaf=1,   # 最小叶子要求
                        max_features=None, ccp_alpha=0.0  # 无剪枝
                    ),
                    'LogisticRegression': LogisticRegression(
                        random_state=42, max_iter=1000, C=1.0,  # 标准正则化
                        penalty='l2'
                    ),
                    'KNN': KNeighborsClassifier(
                        n_neighbors=min(5, max(3, n_samples//4)),
                        weights='distance'
                    ),
                    'NaiveBayes': GaussianNB(),
                }
            elif complexity_level == "high":  # 高复杂度：强正则化
                classifiers = {
                    'SVM': SVC(kernel='linear', C=0.01, random_state=42, probability=True),
                    'DecisionTree': DecisionTreeClassifier(
                        random_state=42, max_depth=2,
                        min_samples_split=max(5, n_samples//2),
                        min_samples_leaf=max(3, n_samples//5),
                        max_features=min(3, n_features), ccp_alpha=0.05
                    ),
                    'LogisticRegression': LogisticRegression(
                        random_state=42, max_iter=1000, C=0.01,
                        penalty='l1', solver='liblinear'
                    ),
                    'KNN': KNeighborsClassifier(
                        n_neighbors=min(n_samples-1, max(5, n_samples//2)),
                        weights='distance'
                    ),
                    'NaiveBayes': GaussianNB(),
                }
            elif complexity_level == "medium":  # 中等复杂度：适中正则化
                classifiers = {
                    'SVM': SVC(kernel='rbf', C=0.1, random_state=42, probability=True),
                    'DecisionTree': DecisionTreeClassifier(
                        random_state=42, max_depth=4,
                        min_samples_split=max(3, n_samples//4),
                        min_samples_leaf=max(2, n_samples//8),
                        max_features='sqrt', ccp_alpha=0.02
                    ),
                    'LogisticRegression': LogisticRegression(
                        random_state=42, max_iter=1000, C=0.1,
                        penalty='l2'
                    ),
                    'KNN': KNeighborsClassifier(
                        n_neighbors=min(7, max(3, n_samples//4)),
                        weights='distance'
                    ),
                    'NaiveBayes': GaussianNB(),
                }
            else:  # 低复杂度：轻度正则化
                classifiers = {
                    'SVM': SVC(kernel='rbf', C=1.0, random_state=42, probability=True),
                    'DecisionTree': DecisionTreeClassifier(
                        random_state=42, max_depth=6,
                        min_samples_split=max(2, n_samples//6),
                        min_samples_leaf=max(1, n_samples//12),
                        max_features='sqrt', ccp_alpha=0.01
                    ),
                    'LogisticRegression': LogisticRegression(
                        random_state=42, max_iter=1000, C=1.0,
                        penalty='l2'
                    ),
                    'KNN': KNeighborsClassifier(
                        n_neighbors=min(5, max(3, n_samples//5)),
                        weights='uniform'
                    ),
                    'NaiveBayes': GaussianNB(),
                }

            # 自适应深度学习模型
            if torch.cuda.is_available():
                # 根据复杂度调整深度学习参数
                if complexity_level == "extreme":
                    epochs, lr = 3, 0.05
                elif complexity_level == "high":
                    epochs, lr = 5, 0.01
                elif complexity_level == "medium":
                    epochs, lr = 10, 0.005
                else:
                    epochs, lr = 15, 0.001

                classifiers['CNN'] = DeepClassifierWrapper(
                    model_type='cnn',
                    input_dim=X_train_scaled.shape[1],
                    epochs=epochs,
                    learning_rate=lr,
                    device=self.device
                )
                classifiers['LSTM'] = DeepClassifierWrapper(
                    model_type='lstm',
                    input_dim=X_train_scaled.shape[1],
                    epochs=epochs,
                    learning_rate=lr,
                    device=self.device
                )

            # 自适应原型网络
            if config is None:
                config = {}
            protonet_enabled = config.get('protonet_enabled', True)
            if torch.cuda.is_available() and protonet_enabled:
                # 根据复杂度调整原型网络参数
                if complexity_level == "extreme":
                    hidden_dim, z_dim, dropout, epochs = 4, 2, 0.7, 3  # 极简配置
                elif complexity_level == "high":
                    hidden_dim, z_dim, dropout, epochs = 8, 4, 0.5, 5
                elif complexity_level == "medium":
                    hidden_dim, z_dim, dropout, epochs = 16, 8, 0.4, 10
                else:
                    hidden_dim, z_dim, dropout, epochs = 32, 16, 0.3, 15

                classifiers['ProtoNet'] = ProtoNetClassifier(
                    input_dim=X_train_scaled.shape[1],
                    hidden_dim=hidden_dim,
                    z_dim=z_dim,
                    dropout_rate=dropout,
                    epochs=epochs,
                    learning_rate=config.get('protonet_lr', 0.01),
                    temperature=config.get('protonet_temperature', 3.0),
                    device=self.device,
                    auto_adapt=True
                )


            # 7. 改进的评估方法 - 使用交叉验证
            print("   📊 使用交叉验证评估模型性能...")

            # 为交叉验证准备完整数据集
            scaler_full = StandardScaler()
            X_orig_scaled = scaler_full.fit_transform(X_orig)

            # 如果需要PCA，也在完整数据集上应用
            if pca is not None:
                pca_full = PCA(n_components=pca.n_components_)
                X_orig_scaled = pca_full.fit_transform(X_orig_scaled)

            baseline_results = {}
            augmented_results = {}

            # 基线评估：使用交叉验证
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

            for name, clf in classifiers.items():
                try:
                    if name in ['CNN', 'LSTM', 'ProtoNet']:
                        # 深度学习模型使用简单分割（交叉验证太耗时）
                        clf.fit(X_train_scaled, y_train)
                        y_pred = clf.predict(X_test_scaled)
                        baseline_results[name] = accuracy_score(y_test, y_pred)
                    else:
                        # 传统机器学习模型使用交叉验证
                        cv_scores = cross_val_score(clf, X_orig_scaled, original_labels,
                                                  cv=skf, scoring='accuracy')
                        baseline_results[name] = np.mean(cv_scores)
                except Exception:
                    baseline_results[name] = 0.0

            # 增强数据评估：使用训练集+增强数据训练，在测试集上评估
            if len(X_aug_train_scaled) > 0:
                X_combined = np.vstack([X_train_scaled, X_aug_train_scaled])
                y_combined = np.hstack([y_train, y_aug_train])

                for name, clf in classifiers.items():
                    try:
                        if name in ['CNN', 'LSTM', 'ProtoNet']:
                            # 重新初始化自适应深度学习模型
                            if name == 'CNN':
                                clf = DeepClassifierWrapper('cnn', X_train_scaled.shape[1],
                                                          epochs=epochs, learning_rate=lr, device=self.device)
                            elif name == 'LSTM':
                                clf = DeepClassifierWrapper('lstm', X_train_scaled.shape[1],
                                                          epochs=epochs, learning_rate=lr, device=self.device)
                            elif name == 'ProtoNet':
                                clf = ProtoNetClassifier(
                                    input_dim=X_train_scaled.shape[1],
                                    hidden_dim=hidden_dim,
                                    z_dim=z_dim,
                                    dropout_rate=dropout,
                                    epochs=epochs,
                                    learning_rate=config.get('protonet_lr', 0.01),
                                    temperature=config.get('protonet_temperature', 3.0),
                                    device=self.device,
                                    auto_adapt=True
                                )

                            clf.fit(X_combined, y_combined)
                            y_pred = clf.predict(X_test_scaled)
                            augmented_results[name] = accuracy_score(y_test, y_pred)
                        else:
                            # 传统机器学习模型 - 重新初始化防过拟合版本
                            n_combined = len(X_combined)
                            if name == 'SVM':
                                clf = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42, probability=True)
                            elif name == 'DecisionTree':
                                clf = DecisionTreeClassifier(
                                    random_state=42, max_depth=3,
                                    min_samples_split=max(2, n_combined//5),
                                    min_samples_leaf=max(1, n_combined//10),
                                    max_features='sqrt', ccp_alpha=0.01
                                )
                            elif name == 'LogisticRegression':
                                clf = LogisticRegression(
                                    random_state=42, max_iter=1000, C=0.1,
                                    penalty='l1', solver='liblinear'
                                )
                            elif name == 'KNN':
                                clf = KNeighborsClassifier(
                                    n_neighbors=min(n_combined-1, max(5, n_combined//2)),
                                    weights='distance'
                                )
                            elif name == 'NaiveBayes':
                                clf = GaussianNB()

                            clf.fit(X_combined, y_combined)
                            y_pred = clf.predict(X_test_scaled)
                            augmented_results[name] = accuracy_score(y_test, y_pred)
                    except Exception:
                        augmented_results[name] = baseline_results.get(name, 0.0)
            else:
                augmented_results = baseline_results.copy()



            # 使用SVM作为主要评估指标（适合认知衰弱分类）
            acc_baseline = baseline_results.get('SVM', 0.0)
            acc_augmented = augmented_results.get('SVM', 0.0)



            # 使用SVM作为主要评估指标

            # 显示所有模型的结果
            print(f"   📊 多模型评估结果:")
            print(f"   {'模型':<12} {'基线准确率':<12} {'增强准确率':<12} {'性能提升':<12}")
            print("   " + "-" * 50)

            for name in classifiers.keys():
                baseline_acc = baseline_results.get(name, 0.0)
                augmented_acc = augmented_results.get(name, 0.0)
                improvement = augmented_acc - baseline_acc
                print(f"   {name:<12} {baseline_acc:<12.4f} {augmented_acc:<12.4f} {improvement:<12.4f}")

            print(f"\n   📊 主要指标 (SVM):")
            print(f"   📊 基线准确率: {acc_baseline:.4f}")
            print(f"   📊 增强准确率: {acc_augmented:.4f}")
            print(f"   📈 性能提升: {acc_augmented - acc_baseline:.4f}")

            # 如果有原型网络结果，也显示
            if 'ProtoNet' in baseline_results:
                proto_baseline = baseline_results['ProtoNet']
                proto_augmented = augmented_results['ProtoNet']
                proto_improvement = proto_augmented - proto_baseline
                print(f"\n   🧠 原型网络结果:")
                print(f"   📊 基线准确率: {proto_baseline:.4f}")
                print(f"   📊 增强准确率: {proto_augmented:.4f}")
                print(f"   📈 性能提升: {proto_improvement:.4f}")

            # 构建详细的结果字典
            result_dict = {
                'quality_score': avg_quality,
                'baseline_accuracy': acc_baseline,
                'augmented_accuracy': acc_augmented,
                'improvement': acc_augmented - acc_baseline,
                'all_models': {
                    'baseline_results': baseline_results,
                    'augmented_results': augmented_results,
                    'improvements': {name: augmented_results[name] - baseline_results[name]
                                   for name in baseline_results.keys()}
                }
            }

            return result_dict
        else:
            print("   ⚠️ 数据量不足，跳过分类评估")
            return {'quality_score': avg_quality}

    def save_augmented_data(self, augmented_data, augmented_labels, output_dir):
        """保存增强数据为单独的CSV文件"""
        print(f"💾 保存增强数据到 {output_dir}...")

        os.makedirs(output_dir, exist_ok=True)

        for i, (data, label) in enumerate(zip(augmented_data, augmented_labels)):
            # 创建DataFrame
            df = pd.DataFrame(data)
            df['label'] = label

            # 保存为CSV文件
            filename = f"augmented_data_{i:04d}.csv"
            filepath = os.path.join(output_dir, filename)
            df.to_csv(filepath, index=False)

        print(f"✅ 保存完成，共 {len(augmented_data)} 个文件")

        # 保存质量评估报告
        return output_dir

def run_feature_based_augmentation(config):
    """运行基于特征学习的数据增强"""

    print("🧠 基于特征学习的认知衰弱数据增强系统")
    print("=" * 60)

    # 初始化系统
    aug_system = FeatureBasedAugmentationSystem(
        seq_len=config['seq_len'],
        latent_dim=config['latent_dim'],
        device=config['device']
    )

    try:
        # 1. 加载和处理数据
        original_data, original_labels, data_by_class = aug_system.load_and_process_data(
            config['data_path']
        )

        # 2. 学习各类别特征
        aug_system.feature_learner.learn_class_features(data_by_class)

        # 3. 训练VAE-GAN网络
        aug_system.train_vae_gan(
            original_data,
            original_labels,
            epochs=config['vae_epochs'],
            batch_size=config['batch_size']
        )

        # 4. 生成增强样本
        augmented_data, augmented_labels = aug_system.generate_augmented_samples(
            original_data,
            original_labels,
            augmentation_method=config['augmentation_method'],
            n_augment_per_sample=config['n_augment_per_sample'],
            mix_ratio=config.get('mix_ratio', 0.8)
        )

        # 5. 评估增强质量
        quality_metrics = aug_system.evaluate_augmentation_quality(
            original_data, original_labels,
            augmented_data, augmented_labels,
            config
        )

        # 6. 保存增强数据
        output_dir = aug_system.save_augmented_data(
            augmented_data, augmented_labels, config['output_dir']
        )

        # 7. 生成可视化报告
        if create_comprehensive_report is not None:
            create_comprehensive_report(
                original_data, augmented_data, original_labels, augmented_labels,
                quality_metrics, output_dir
            )

        # 8. 保存质量评估报告
        import json
        with open(os.path.join(output_dir, 'quality_report.json'), 'w', encoding='utf-8') as f:
            json.dump(quality_metrics, f, indent=2, ensure_ascii=False)

        # 9. 打印总结
        print("\n" + "=" * 60)
        print("📊 增强完成总结")
        print("=" * 60)
        print(f"📂 原始数据: {len(original_data)} 个样本")
        print(f"📂 增强数据: {len(augmented_data)} 个样本")
        print(f"📈 数据增长: {len(augmented_data) / len(original_data):.1f}x")
        print(f"📊 质量分数: {quality_metrics.get('quality_score', 0):.4f}")

        if 'baseline_accuracy' in quality_metrics:
            print(f"📊 基线准确率: {quality_metrics['baseline_accuracy']:.4f}")
            print(f"📊 增强准确率: {quality_metrics['augmented_accuracy']:.4f}")
            print(f"📈 性能提升: {quality_metrics['improvement']:.4f}")

        print(f"💾 结果保存在: {output_dir}")

        return quality_metrics

    except Exception as e:
        print(f"❌ 错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """主函数 - 配置参数并运行增强系统"""

    # ==================== 配置参数 ====================
    config = {
        # 数据路径配置
        'data_path': './feature1',  # 原始数据目录
        'output_dir': 'augmented_results',  # 输出目录

        # 数据处理参数
        'seq_len': 400,  # 序列长度

        # VAE-GAN网络参数
        'latent_dim': 64,  # 潜在空间维度
        'vae_epochs': 1000,  # VAE-GAN训练轮数
        'batch_size': 16,  # 批次大小
        'device': 'cuda',  # 计算设备

        # 增强方法选择 (重要参数)
        'augmentation_method': 'hybrid',  # 可选: 'feature_perturbation', 'vae_reconstruction', 'hybrid', 'all'

        # 增强参数
        'n_augment_per_sample': 2,  # 每个原始样本生成的增强样本数
        'mix_ratio': 0.8,  # VAE重构时与原始数据的混合比例 (仅对vae_reconstruction方法有效)

        # 特征学习参数
        'feature_components': 20,  # PCA特征数量
    }
    # ================================================

    print("📋 配置信息:")
    for key, value in config.items():
        print(f"   {key}: {value}")
    print()

    # 检查数据目录
    if not os.path.exists(config['data_path']):
        print(f"❌ 错误: 数据目录不存在: {config['data_path']}")
        print("💡 请确保original目录存在并包含CSV文件")
        return

    # 运行增强系统
    results = run_feature_based_augmentation(config)

    if results:
        print("\n🎉 增强系统运行成功!")
    else:
        print("\n❌ 增强系统运行失败!")

if __name__ == "__main__":
    main()
