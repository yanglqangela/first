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
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
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
        """加载和处理原始数据"""
        print(f"📂 从 {data_dir} 加载数据...")
        
        if not os.path.exists(data_dir):
            raise ValueError(f"目录 '{data_dir}' 不存在")
        
        data_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
        if not data_files:
            raise ValueError(f"在 {data_dir} 中未找到CSV文件")
        
        all_data = []
        all_labels = []
        data_by_class = {0: [], 1: [], 2: []}
        
        print(f"📊 找到 {len(data_files)} 个数据文件")
        
        for file in data_files:
            try:
                df = pd.read_csv(os.path.join(data_dir, file))
                
                # 提取标签
                if 'label' in df.columns:
                    label = int(df['label'].iloc[0])
                    df = df.drop('label', axis=1)
                else:
                    # 从文件名提取标签
                    import re
                    numbers = re.findall(r'\d+', file)
                    label = int(numbers[0]) % 3 if numbers else 0
                
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
                print(f"⚠️ 处理文件 {file} 时出错: {e}")
                continue
        
        if not all_data:
            raise ValueError("没有成功加载任何数据文件")
        
        print(f"✅ 数据加载完成")
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

                # 训练稳定性检查
                if len(vae_losses) >= 20:
                    vae_trend = np.polyfit(range(10), vae_losses[-10:], 1)[0]
                    disc_trend = np.polyfit(range(10), disc_losses[-10:], 1)[0]

                    if vae_trend > 0.01:
                        print(f"     ⚠️ VAE损失上升趋势: {vae_trend:.6f}")
                    if disc_trend > 0.01:
                        print(f"     ⚠️ 判别器损失上升趋势: {disc_trend:.6f}")

                    # 自适应学习率调整
                    if disc_trend > 0.02:  # 判别器损失上升过快
                        for param_group in optimizer_disc.param_groups:
                            param_group['lr'] *= 0.9
                        print(f"     🔧 降低判别器学习率至: {optimizer_disc.param_groups[0]['lr']:.6f}")

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

    def generate_augmented_samples(self, original_data, original_labels, n_augment_per_sample=2):
        """生成改进的增强样本"""
        print("🔄 生成改进的增强样本...")

        if self.vae_gan is None:
            raise ValueError("VAE-GAN未训练，请先调用train_vae_gan")

        augmented_data = []
        augmented_labels = []

        # 计算每个类别的样本数量，用于平衡增强
        class_counts = np.bincount(original_labels)
        max_class_count = np.max(class_counts)

        for sample_idx, (data, label) in enumerate(zip(original_data, original_labels)):
            # 获取该类别的扰动模式
            perturbation_patterns = self.feature_learner.generate_perturbation_patterns(label)

            # 动态调整增强数量 - 少数类别生成更多样本
            class_ratio = class_counts[label] / max_class_count
            adaptive_n_augment = max(1, int(n_augment_per_sample / class_ratio))
            adaptive_n_augment = min(adaptive_n_augment, n_augment_per_sample * 2)  # 限制最大数量

            for j in range(adaptive_n_augment):
                # 选择增强策略
                augment_strategy = j % 3  # 三种策略轮换

                if augment_strategy == 0:
                    # 策略1: 仅特征扰动（保持原始数据特性）
                    if perturbation_patterns:
                        pattern = perturbation_patterns[j % len(perturbation_patterns)]
                        # 减少扰动强度
                        pattern['intensity'] *= 0.5
                        enhanced_data = self.apply_feature_based_perturbation(data, label, pattern)
                    else:
                        enhanced_data = data + np.random.normal(0, 0.01, data.shape)

                elif augment_strategy == 1:
                    # 策略2: VAE重构（轻微变换）
                    with torch.no_grad():
                        flat_data = data.flatten().reshape(1, -1)
                        scaled_data = self.global_scaler.transform(flat_data)
                        tensor_data = torch.FloatTensor(scaled_data).to(self.device)

                        # 仅使用VAE重构，不添加额外噪声
                        recon_data, _, _ = self.vae_gan(tensor_data)

                        # 与原始数据混合
                        mix_ratio = 0.7  # 70%原始数据，30%重构数据
                        mixed_data = mix_ratio * tensor_data + (1 - mix_ratio) * recon_data

                        enhanced_scaled = mixed_data.cpu().numpy()
                        enhanced_flat = self.global_scaler.inverse_transform(enhanced_scaled)
                        enhanced_data = enhanced_flat.reshape(data.shape)

                else:
                    # 策略3: 特征扰动 + 轻微VAE增强
                    if perturbation_patterns:
                        pattern = perturbation_patterns[j % len(perturbation_patterns)]
                        pattern['intensity'] *= 0.3  # 更小的扰动
                        perturbed_data = self.apply_feature_based_perturbation(data, label, pattern)
                    else:
                        perturbed_data = data

                    with torch.no_grad():
                        flat_data = perturbed_data.flatten().reshape(1, -1)
                        scaled_data = self.global_scaler.transform(flat_data)
                        tensor_data = torch.FloatTensor(scaled_data).to(self.device)

                        # 在潜在空间添加很小的噪声
                        mu, _ = self.vae_gan.encode(tensor_data)
                        noise = torch.randn_like(mu) * 0.05  # 减少噪声强度
                        z = mu + noise
                        enhanced_tensor = self.vae_gan.decode(z)

                        enhanced_scaled = enhanced_tensor.cpu().numpy()
                        enhanced_flat = self.global_scaler.inverse_transform(enhanced_scaled)
                        enhanced_data = enhanced_flat.reshape(data.shape)

                # 质量检查 - 过滤异常样本
                if self._is_valid_sample(enhanced_data, data):
                    augmented_data.append(enhanced_data)
                    augmented_labels.append(label)
                else:
                    # 如果生成的样本质量不好，使用轻微噪声版本
                    fallback_data = data + np.random.normal(0, 0.005, data.shape)
                    augmented_data.append(fallback_data)
                    augmented_labels.append(label)

        print(f"✅ 生成 {len(augmented_data)} 个增强样本")
        print(f"   📊 平均每个原始样本生成 {len(augmented_data)/len(original_data):.1f} 个增强样本")
        return augmented_data, np.array(augmented_labels)

    def _is_valid_sample(self, enhanced_data, original_data, threshold=3.0):
        """检查生成样本的有效性"""
        try:
            # 检查是否有异常值
            if np.any(np.isnan(enhanced_data)) or np.any(np.isinf(enhanced_data)):
                return False

            # 检查与原始数据的差异是否过大
            diff_ratio = np.abs(enhanced_data - original_data) / (np.abs(original_data) + 1e-8)
            if np.mean(diff_ratio) > threshold:
                return False

            # 检查方差是否合理
            if np.std(enhanced_data) < 0.001 or np.std(enhanced_data) > 100 * np.std(original_data):
                return False

            return True
        except:
            return False

    def compute_quality_score(self, sample, original_sample):
        """计算样本质量分数"""
        score = 0.0

        # 1. 统计特性检查
        if not (np.isnan(sample).any() or np.isinf(sample).any()):
            score += 0.2

        # 2. 方差检查
        if np.std(sample) > 0.01:
            score += 0.2

        # 3. 范围合理性检查
        if np.abs(sample.mean()) < 10 * np.abs(original_sample.mean()):
            score += 0.2

        # 4. 形状相似性检查
        try:
            correlation = np.corrcoef(sample.flatten(), original_sample.flatten())[0, 1]
            if not np.isnan(correlation) and correlation > 0.3:
                score += 0.2
        except:
            pass

        # 5. 分布相似性检查
        try:
            _, p_value = stats.ks_2samp(sample.flatten(), original_sample.flatten())
            if p_value > 0.01:  # 分布不显著不同
                score += 0.2
        except:
            pass

        return score

    def evaluate_augmentation_quality(self, original_data, original_labels, augmented_data, augmented_labels):
        """评估增强质量"""
        print("🔍 评估增强质量...")

        # 计算质量分数
        quality_scores = []
        for i, aug_sample in enumerate(augmented_data):
            orig_idx = i // 2  # 假设每个原始样本生成2个增强样本
            if orig_idx < len(original_data):
                score = self.compute_quality_score(aug_sample, original_data[orig_idx])
                quality_scores.append(score)

        avg_quality = np.mean(quality_scores)
        print(f"   📊 平均质量分数: {avg_quality:.4f}")

        # 分类性能评估
        def flatten_data(data_list):
            return np.array([data.flatten() for data in data_list])

        X_orig = flatten_data(original_data)
        X_aug = flatten_data(augmented_data)

        # 标准化
        scaler = StandardScaler()
        X_orig_scaled = scaler.fit_transform(X_orig)
        X_aug_scaled = scaler.transform(X_aug)

        # PCA降维
        if X_orig_scaled.shape[1] > 100:
            n_components = min(20, X_orig_scaled.shape[0]-1, X_orig_scaled.shape[1])
            pca = PCA(n_components=n_components)
            X_orig_scaled = pca.fit_transform(X_orig_scaled)
            X_aug_scaled = pca.transform(X_aug_scaled)

        # 分割数据
        if len(original_labels) >= 6:
            X_train, X_test, y_train, y_test = train_test_split(
                X_orig_scaled, original_labels, test_size=0.3, random_state=42,
                stratify=original_labels
            )

            # 基线测试
            clf_baseline = RandomForestClassifier(n_estimators=100, random_state=42)
            clf_baseline.fit(X_train, y_train)
            y_pred_baseline = clf_baseline.predict(X_test)
            acc_baseline = accuracy_score(y_test, y_pred_baseline)

            # 增强数据测试
            X_combined = np.vstack([X_train, X_aug_scaled])
            y_combined = np.hstack([y_train, augmented_labels])

            clf_augmented = RandomForestClassifier(n_estimators=100, random_state=42)
            clf_augmented.fit(X_combined, y_combined)
            y_pred_augmented = clf_augmented.predict(X_test)
            acc_augmented = accuracy_score(y_test, y_pred_augmented)

            print(f"   📊 基线准确率: {acc_baseline:.4f}")
            print(f"   📊 增强准确率: {acc_augmented:.4f}")
            print(f"   📈 性能提升: {acc_augmented - acc_baseline:.4f}")

            return {
                'quality_score': avg_quality,
                'baseline_accuracy': acc_baseline,
                'augmented_accuracy': acc_augmented,
                'improvement': acc_augmented - acc_baseline
            }
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
            n_augment_per_sample=config['n_augment_per_sample']
        )

        # 5. 评估增强质量
        quality_metrics = aug_system.evaluate_augmentation_quality(
            original_data, original_labels,
            augmented_data, augmented_labels
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
        'batch_size': 8,  # 批次大小
        'device': 'cuda',  # 计算设备

        # 增强参数
        'n_augment_per_sample': 2,  # 每个原始样本生成的增强样本数

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
