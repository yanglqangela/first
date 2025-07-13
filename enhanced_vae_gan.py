#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
增强版VAE-GAN模型 - 解决收敛问题和提高生成质量
- 改进损失函数设计
- 添加数据质量筛选机制
- 使用渐进式训练和稳定性技术
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils import spectral_norm
import warnings
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from scipy import stats
import os
warnings.filterwarnings('ignore')


class SelfAttention(nn.Module):
    """自注意力机制"""
    
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.in_dim = in_dim
        self.query_conv = nn.Conv1d(in_dim, in_dim // 8, 1)
        self.key_conv = nn.Conv1d(in_dim, in_dim // 8, 1)
        self.value_conv = nn.Conv1d(in_dim, in_dim, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        batch_size, C, length = x.size()
        proj_query = self.query_conv(x).view(batch_size, -1, length).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(batch_size, -1, length)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(batch_size, -1, length)
        
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, length)
        
        out = self.gamma * out + x
        return out


class EnhancedEncoder(nn.Module):
    """增强编码器 - 多尺度特征提取"""
    
    def __init__(self, input_dim, hidden_dim=128, latent_dim=64):
        super(EnhancedEncoder, self).__init__()
        self.latent_dim = latent_dim
        
        # 多尺度卷积特征提取
        self.conv1 = nn.Sequential(
            nn.Conv1d(input_dim, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.25)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv1d(64, hidden_dim, kernel_size=7, padding=3),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        
        # 自注意力
        self.attention = SelfAttention(hidden_dim)
        
        # 全局特征提取
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # 潜在空间映射
        self.fc_hidden = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.4)
        )
        
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # 初始化权重
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # x: [batch_size, seq_len, input_dim] -> [batch_size, input_dim, seq_len]
        x = x.transpose(1, 2)
        
        # 多尺度特征提取
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        
        # 自注意力
        x_att = self.attention(x3)
        
        # 全局池化
        x_pooled = self.global_pool(x_att).squeeze(-1)
        
        # 隐藏层
        h = self.fc_hidden(x_pooled)
        
        # 潜在参数
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        # 稳定logvar以避免过度正则化
        logvar = torch.clamp(logvar, min=-10, max=2)
        
        return mu, logvar


class EnhancedDecoder(nn.Module):
    """增强解码器 - 注意力引导重构"""
    
    def __init__(self, latent_dim, hidden_dim=128, output_dim=1, seq_len=100):
        super(EnhancedDecoder, self).__init__()
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        
        # 潜在向量投影
        self.fc_expand = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim * seq_len // 4)
        )
        
        # 上采样卷积
        self.upconv1 = nn.Sequential(
            nn.ConvTranspose1d(hidden_dim, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.25)
        )
        
        self.upconv2 = nn.Sequential(
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2)
        )
        
        # 自注意力
        self.attention = SelfAttention(32)
        
        # 输出层
        self.output_conv = nn.Sequential(
            nn.Conv1d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.2),
            nn.Conv1d(16, output_dim, kernel_size=3, padding=1),
            nn.Sigmoid()  # 输出到[0,1]范围
        )
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, z):
        batch_size = z.size(0)
        
        # 扩展潜在向量
        x = self.fc_expand(z)
        x = x.view(batch_size, self.hidden_dim, self.seq_len // 4)
        
        # 上采样
        x = self.upconv1(x)
        x = self.upconv2(x)
        
        # 调整到目标长度
        if x.size(-1) != self.seq_len:
            x = F.interpolate(x, size=self.seq_len, mode='linear', align_corners=False)
        
        # 自注意力
        x = self.attention(x)
        
        # 输出
        x = self.output_conv(x)
        
        # 转回时序格式
        x = x.transpose(1, 2)
        
        return x


class EnhancedVAE(nn.Module):
    """增强VAE"""
    
    def __init__(self, input_dim, seq_len, hidden_dim=128, latent_dim=64):
        super(EnhancedVAE, self).__init__()
        self.latent_dim = latent_dim
        
        self.encoder = EnhancedEncoder(input_dim, hidden_dim, latent_dim)
        self.decoder = EnhancedDecoder(latent_dim, hidden_dim, input_dim, seq_len)
        
    def reparameterize(self, mu, logvar):
        """改进的重参数化"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decoder(z)
        return recon_x, mu, logvar, z


class EnhancedGenerator(nn.Module):
    """增强生成器 - 渐进式生成"""
    
    def __init__(self, noise_dim, output_dim, seq_len, hidden_dim=128):
        super(EnhancedGenerator, self).__init__()
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        
        # 噪声投影
        self.noise_fc = nn.Sequential(
            nn.Linear(noise_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim * seq_len // 8)
        )
        
        # 渐进式上采样
        self.progressive_blocks = nn.ModuleList([
            # Block 1: seq_len//8 -> seq_len//4
            nn.Sequential(
                nn.ConvTranspose1d(hidden_dim, hidden_dim//2, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm1d(hidden_dim//2),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.25)
            ),
            # Block 2: seq_len//4 -> seq_len//2
            nn.Sequential(
                nn.ConvTranspose1d(hidden_dim//2, hidden_dim//4, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm1d(hidden_dim//4),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.2)
            ),
            # Block 3: seq_len//2 -> seq_len
            nn.Sequential(
                nn.ConvTranspose1d(hidden_dim//4, hidden_dim//8, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm1d(hidden_dim//8),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.15)
            )
        ])
        
        # 输出层
        self.output_layer = nn.Sequential(
            nn.Conv1d(hidden_dim//8, output_dim, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, z):
        batch_size = z.size(0)
        
        # 噪声投影
        x = self.noise_fc(z)
        x = x.view(batch_size, self.hidden_dim, self.seq_len // 8)
        
        # 渐进式上采样
        for block in self.progressive_blocks:
            x = block(x)
        
        # 调整到目标长度
        if x.size(-1) != self.seq_len:
            x = F.interpolate(x, size=self.seq_len, mode='linear', align_corners=False)
        
        # 输出
        x = self.output_layer(x)
        
        # 转回时序格式
        x = x.transpose(1, 2)
        
        return x


class EnhancedDiscriminator(nn.Module):
    """增强判别器 - 多尺度判别"""
    
    def __init__(self, input_dim, seq_len, hidden_dim=128):
        super(EnhancedDiscriminator, self).__init__()
        
        # 多尺度卷积特征提取
        self.feature_extractor = nn.Sequential(
            # 第一层
            spectral_norm(nn.Conv1d(input_dim, 32, kernel_size=3, padding=1)),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            
            # 第二层
            spectral_norm(nn.Conv1d(32, 64, kernel_size=5, padding=2, stride=2)),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.25),
            
            # 第三层
            spectral_norm(nn.Conv1d(64, 128, kernel_size=5, padding=2, stride=2)),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        
        # 自注意力
        self.attention = SelfAttention(128)
        
        # 全局特征
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # 分类层
        self.classifier = nn.Sequential(
            spectral_norm(nn.Linear(128, hidden_dim)),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.4),
            spectral_norm(nn.Linear(hidden_dim, hidden_dim//2)),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            spectral_norm(nn.Linear(hidden_dim//2, 1))
        )
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x, return_features=False):
        # x: [batch_size, seq_len, input_dim] -> [batch_size, input_dim, seq_len]
        x = x.transpose(1, 2)
        
        # 特征提取
        features = self.feature_extractor(x)
        
        # 自注意力
        features = self.attention(features)
        
        # 全局池化
        pooled = self.global_pool(features).squeeze(-1)
        
        if return_features:
            return pooled
        
        # 分类
        output = self.classifier(pooled)
        output = torch.sigmoid(output)
        
        return output


class DataQualityFilter:
    """数据质量筛选器"""
    
    def __init__(self, threshold_std=2.0, threshold_autocorr=0.1):
        self.threshold_std = threshold_std
        self.threshold_autocorr = threshold_autocorr
        
    def filter_samples(self, samples, original_data=None):
        """筛选高质量样本"""
        if len(samples.shape) == 3:
            samples_2d = samples.reshape(samples.shape[0], -1)
        else:
            samples_2d = samples
            
        quality_scores = []
        valid_indices = []
        
        for i, sample in enumerate(samples_2d):
            score = self.compute_quality_score(sample, original_data)
            quality_scores.append(score)
            
            # 质量阈值筛选
            if score > 0.6:  # 可调整阈值
                valid_indices.append(i)
        
        return np.array(valid_indices), np.array(quality_scores)
    
    def compute_quality_score(self, sample, original_data=None):
        """计算样本质量分数"""
        score = 0.0
        
        # 1. 统计特性检查
        if not (np.isnan(sample).any() or np.isinf(sample).any()):
            score += 0.2
        
        # 2. 方差检查（避免常数序列）
        if np.std(sample) > 0.01:
            score += 0.2
        
        # 3. 范围检查
        if 0 <= sample.min() and sample.max() <= 1:
            score += 0.2
        
        # 4. 平滑性检查（避免过度震荡）
        if len(sample) > 1:
            diff = np.diff(sample)
            if np.std(diff) < self.threshold_std * np.std(sample):
                score += 0.2
        
        # 5. 如果有原始数据，检查分布相似性
        if original_data is not None:
            try:
                # KS测试
                _, p_value = stats.ks_2samp(sample, original_data.flatten())
                if p_value > 0.05:  # 分布相似
                    score += 0.2
            except:
                pass
        else:
            score += 0.2  # 默认给分
        
        return score


class EnhancedVAEGAN:
    """增强版VAE-GAN模型"""
    
    def __init__(self, input_dim, seq_len, latent_dim=64, hidden_dim=128, 
                 lr_vae=5e-5, lr_gen=1e-4, lr_disc=1e-4, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.latent_dim = latent_dim
        
        # 初始化网络
        self.vae = EnhancedVAE(input_dim, seq_len, hidden_dim, latent_dim).to(self.device)
        self.generator = EnhancedGenerator(latent_dim, input_dim, seq_len, hidden_dim).to(self.device)
        self.discriminator = EnhancedDiscriminator(input_dim, seq_len, hidden_dim).to(self.device)
        
        # 数据质量筛选器
        self.quality_filter = DataQualityFilter()
        
        # 优化器 - 使用不同学习率
        self.optimizer_vae = optim.Adam(self.vae.parameters(), lr=lr_vae, betas=(0.5, 0.999), weight_decay=1e-5)
        self.optimizer_gen = optim.Adam(self.generator.parameters(), lr=lr_gen, betas=(0.5, 0.999))
        self.optimizer_disc = optim.Adam(self.discriminator.parameters(), lr=lr_disc, betas=(0.5, 0.999))
        
        # 学习率调度器
        self.scheduler_vae = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer_vae, T_0=50, T_mult=2, eta_min=1e-6)
        self.scheduler_gen = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer_gen, T_0=50, T_mult=2, eta_min=1e-6)
        self.scheduler_disc = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer_disc, T_0=50, T_mult=2, eta_min=1e-6)
        
        # 损失权重 - 动态调整
        self.lambda_recon = 1.0
        self.lambda_kl = 0.01  # 降低KL权重
        self.lambda_adv = 0.1   # 降低对抗权重
        self.lambda_feature = 0.1
        
        # 训练历史
        self.train_history = {
            'vae_loss': [],
            'recon_loss': [],
            'kl_loss': [],
            'gen_loss': [],
            'disc_loss': [],
            'quality_scores': []
        }
        
        # 训练阶段标志
        self.current_epoch = 0
        
    def improved_vae_loss(self, recon_x, x, mu, logvar):
        """改进的VAE损失函数"""
        # 1. 重构损失 - 使用多种损失的组合
        mse_loss = F.mse_loss(recon_x, x, reduction='mean')
        l1_loss = F.l1_loss(recon_x, x, reduction='mean')
        
        # 感知损失（基于差分）
        x_diff = torch.diff(x, dim=1)
        recon_diff = torch.diff(recon_x, dim=1)
        perceptual_loss = F.mse_loss(recon_diff, x_diff, reduction='mean')
        
        # 组合重构损失
        recon_loss = 0.7 * mse_loss + 0.2 * l1_loss + 0.1 * perceptual_loss
        
        # 2. 改进的KL损失 - 使用β-VAE策略
        beta = min(1.0, self.current_epoch / 100.0)  # 渐进式增加
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss = beta * kl_loss
        
        return recon_loss, kl_loss
    
    def wasserstein_loss(self, real_validity, fake_validity, real_data=True):
        """Wasserstein损失"""
        if real_data:
            return -torch.mean(real_validity)
        else:
            return torch.mean(fake_validity)
    
    def gradient_penalty(self, real_data, fake_data):
        """梯度惩罚"""
        batch_size = real_data.size(0)
        alpha = torch.rand(batch_size, 1, 1).to(self.device)
        
        interpolated = alpha * real_data + (1 - alpha) * fake_data
        interpolated.requires_grad_(True)
        
        d_interpolated = self.discriminator(interpolated)
        
        gradients = torch.autograd.grad(
            outputs=d_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(d_interpolated),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        gradients = gradients.reshape(batch_size, -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        
        return gradient_penalty
    
    def train_step(self, real_data):
        """改进的训练步骤"""
        batch_size = real_data.size(0)
        
        # 标签平滑
        real_labels = torch.ones(batch_size, 1).to(self.device) * 0.9
        fake_labels = torch.zeros(batch_size, 1).to(self.device) + 0.1
        
        # ================== 第一阶段：训练VAE ==================
        self.optimizer_vae.zero_grad()
        
        recon_data, mu, logvar, z_vae = self.vae(real_data)
        recon_loss, kl_loss = self.improved_vae_loss(recon_data, real_data, mu, logvar)
        
        vae_loss = self.lambda_recon * recon_loss + self.lambda_kl * kl_loss
        vae_loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.vae.parameters(), max_norm=0.5)
        self.optimizer_vae.step()
        
        # ================== 第二阶段：训练判别器 ==================
        for _ in range(2):  # 多次训练判别器
            self.optimizer_disc.zero_grad()
            
            # 真实数据
            real_validity = self.discriminator(real_data)
            real_loss = F.binary_cross_entropy(real_validity, real_labels)
            
            # 生成数据
            z_noise = torch.randn(batch_size, self.latent_dim).to(self.device)
            fake_data = self.generator(z_noise).detach()
            fake_validity = self.discriminator(fake_data)
            fake_loss = F.binary_cross_entropy(fake_validity, fake_labels)
            
            # 梯度惩罚
            gp = self.gradient_penalty(real_data, fake_data)
            
            disc_loss = (real_loss + fake_loss) / 2 + 0.1 * gp
            disc_loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=0.5)
            self.optimizer_disc.step()
        
        # ================== 第三阶段：训练生成器 ==================
        self.optimizer_gen.zero_grad()
        
        z_noise = torch.randn(batch_size, self.latent_dim).to(self.device)
        fake_data = self.generator(z_noise)
        fake_validity = self.discriminator(fake_data)
        
        # 对抗损失
        adv_loss = F.binary_cross_entropy(fake_validity, real_labels)
        
        # 特征匹配损失
        with torch.no_grad():
            real_features = self.discriminator(real_data, return_features=True)
        fake_features = self.discriminator(fake_data, return_features=True)
        feature_loss = F.mse_loss(fake_features, real_features)
        
        # 多样性损失（鼓励生成多样化样本）
        z_noise2 = torch.randn(batch_size, self.latent_dim).to(self.device)
        fake_data2 = self.generator(z_noise2)
        diversity_loss = -F.mse_loss(fake_data, fake_data2)
        
        gen_loss = (self.lambda_adv * adv_loss + 
                   self.lambda_feature * feature_loss + 
                   0.01 * diversity_loss)
        
        gen_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=0.5)
        self.optimizer_gen.step()
        
        # 动态调整损失权重
        if self.current_epoch > 50:
            # 如果重构损失太高，增加其权重
            if recon_loss.item() > 0.1:
                self.lambda_recon = min(2.0, self.lambda_recon * 1.01)
            # 如果KL损失过低，增加其权重
            if kl_loss.item() < 0.01:
                self.lambda_kl = min(0.5, self.lambda_kl * 1.01)
        
        # 记录损失
        self.train_history['vae_loss'].append(vae_loss.item())
        self.train_history['recon_loss'].append(recon_loss.item())
        self.train_history['kl_loss'].append(kl_loss.item())
        self.train_history['gen_loss'].append(gen_loss.item())
        self.train_history['disc_loss'].append(disc_loss.item())
        
        return {
            'vae_loss': vae_loss.item(),
            'recon_loss': recon_loss.item(),
            'kl_loss': kl_loss.item(),
            'gen_loss': gen_loss.item(),
            'disc_loss': disc_loss.item()
        }
    
    def train(self, dataloader, epochs=200, save_interval=20):
        """训练模型"""
        print(f"Starting enhanced training on {self.device}")
        print(f"Data shape: batch_size x {self.seq_len} x {self.input_dim}")
        
        for epoch in range(epochs):
            self.current_epoch = epoch
            epoch_losses = {
                'vae_loss': [],
                'recon_loss': [],
                'kl_loss': [],
                'gen_loss': [],
                'disc_loss': []
            }
            
            for batch_idx, (data, _) in enumerate(dataloader):
                data = data.to(self.device)
                
                if len(data.shape) == 2:
                    data = data.unsqueeze(-1)
                
                losses = self.train_step(data)
                
                for key, value in losses.items():
                    epoch_losses[key].append(value)
            
            # 计算平均损失
            avg_losses = {key: np.mean(values) for key, values in epoch_losses.items()}
            
            # 更新学习率
            self.scheduler_vae.step()
            self.scheduler_gen.step()
            self.scheduler_disc.step()
            
            # 打印进度
            if epoch % 10 == 0:
                print(f"Epoch [{epoch}/{epochs}]:")
                for key, value in avg_losses.items():
                    print(f"  {key}: {value:.6f}")
                print(f"  lambda_recon: {self.lambda_recon:.4f}")
                print(f"  lambda_kl: {self.lambda_kl:.4f}")
                print()
            
            # 保存模型
            if epoch % save_interval == 0 and epoch > 0:
                self.save_models(f"checkpoints/enhanced_epoch_{epoch}")
        
        print("Enhanced training completed!")
    
    def generate_samples(self, num_samples, temperature=1.0):
        """生成样本 - 兼容接口"""
        self.vae.eval()
        self.generator.eval()
        
        with torch.no_grad():
            # 使用生成器生成
            z = torch.randn(num_samples, self.latent_dim).to(self.device) * temperature
            generated = self.generator(z).cpu().numpy()
            
            # 使用VAE生成
            z_vae = torch.randn(num_samples, self.latent_dim).to(self.device) * temperature
            generated_vae = self.vae.decoder(z_vae).cpu().numpy()
        
        self.vae.train()
        self.generator.train()
        
        return generated, generated_vae

    def generate_high_quality_samples(self, num_samples, original_data=None, quality_threshold=0.7):
        """生成高质量样本"""
        self.vae.eval()
        self.generator.eval()
        
        high_quality_samples = []
        attempts = 0
        max_attempts = num_samples * 5  # 最多尝试5倍数量
        
        with torch.no_grad():
            while len(high_quality_samples) < num_samples and attempts < max_attempts:
                batch_size = min(50, num_samples - len(high_quality_samples))
                
                # 生成候选样本
                z = torch.randn(batch_size, self.latent_dim).to(self.device)
                candidates = self.generator(z).cpu().numpy()
                
                # 质量筛选
                valid_indices, quality_scores = self.quality_filter.filter_samples(
                    candidates, original_data)
                
                # 选择高质量样本
                high_quality_indices = valid_indices[quality_scores[valid_indices] >= quality_threshold]
                
                if len(high_quality_indices) > 0:
                    selected_samples = candidates[high_quality_indices]
                    high_quality_samples.extend(selected_samples)
                
                attempts += batch_size
        
        self.vae.train()
        self.generator.train()
        
        if len(high_quality_samples) < num_samples:
            print(f"⚠️ 只生成了 {len(high_quality_samples)} 个高质量样本（目标：{num_samples}）")
        
        return np.array(high_quality_samples[:num_samples])
    
    def save_models(self, path):
        """保存模型"""
        os.makedirs(path, exist_ok=True)
        
        torch.save({
            'vae_state_dict': self.vae.state_dict(),
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'optimizer_vae_state_dict': self.optimizer_vae.state_dict(),
            'optimizer_gen_state_dict': self.optimizer_gen.state_dict(),
            'optimizer_disc_state_dict': self.optimizer_disc.state_dict(),
            'train_history': self.train_history,
            'current_epoch': self.current_epoch,
            'lambda_recon': self.lambda_recon,
            'lambda_kl': self.lambda_kl,
        }, f"{path}/enhanced_model.pth")
        
        print(f"Enhanced models saved to {path}")
    
    def load_models(self, path):
        """加载模型"""
        checkpoint = torch.load(f"{path}/enhanced_model.pth")
        
        self.vae.load_state_dict(checkpoint['vae_state_dict'])
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        
        self.optimizer_vae.load_state_dict(checkpoint['optimizer_vae_state_dict'])
        self.optimizer_gen.load_state_dict(checkpoint['optimizer_gen_state_dict'])
        self.optimizer_disc.load_state_dict(checkpoint['optimizer_disc_state_dict'])
        
        self.train_history = checkpoint['train_history']
        self.current_epoch = checkpoint['current_epoch']
        self.lambda_recon = checkpoint['lambda_recon']
        self.lambda_kl = checkpoint['lambda_kl']
        
        print(f"Enhanced models loaded from {path}")
    
    def plot_enhanced_training_history(self, save_path=None):
        """绘制增强训练历史"""
        plt.figure(figsize=(20, 12))
        
        # VAE损失
        plt.subplot(2, 4, 1)
        plt.plot(self.train_history['vae_loss'])
        plt.title('VAE Loss')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.yscale('log')
        plt.grid(True)
        
        # 重构损失
        plt.subplot(2, 4, 2)
        plt.plot(self.train_history['recon_loss'])
        plt.title('Reconstruction Loss')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.yscale('log')
        plt.grid(True)
        
        # KL损失
        plt.subplot(2, 4, 3)
        plt.plot(self.train_history['kl_loss'])
        plt.title('KL Divergence Loss')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.yscale('log')
        plt.grid(True)
        
        # 生成器损失
        plt.subplot(2, 4, 4)
        plt.plot(self.train_history['gen_loss'])
        plt.title('Generator Loss')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.grid(True)
        
        # 判别器损失
        plt.subplot(2, 4, 5)
        plt.plot(self.train_history['disc_loss'])
        plt.title('Discriminator Loss')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.grid(True)
        
        # 损失平滑曲线
        plt.subplot(2, 4, 6)
        window = 100
        if len(self.train_history['vae_loss']) > window:
            smoothed_vae = np.convolve(self.train_history['vae_loss'], 
                                     np.ones(window)/window, mode='valid')
            smoothed_gen = np.convolve(self.train_history['gen_loss'], 
                                     np.ones(window)/window, mode='valid')
            smoothed_disc = np.convolve(self.train_history['disc_loss'], 
                                      np.ones(window)/window, mode='valid')
            
            plt.plot(smoothed_vae, label='VAE (smoothed)')
            plt.plot(smoothed_gen, label='Generator (smoothed)')
            plt.plot(smoothed_disc, label='Discriminator (smoothed)')
            plt.title('Smoothed Losses')
            plt.xlabel('Iteration')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
        
        # 收敛性分析
        plt.subplot(2, 4, 7)
        recent_losses = self.train_history['recon_loss'][-1000:]
        if len(recent_losses) > 0:
            plt.plot(recent_losses)
            plt.title(f'Recent Reconstruction Loss\n(Avg: {np.mean(recent_losses):.6f})')
            plt.xlabel('Recent Iterations')
            plt.ylabel('Loss')
            plt.grid(True)
        
        # 质量分数（如果有）
        plt.subplot(2, 4, 8)
        if self.train_history.get('quality_scores'):
            plt.plot(self.train_history['quality_scores'])
            plt.title('Sample Quality Scores')
            plt.xlabel('Iteration')
            plt.ylabel('Quality Score')
            plt.grid(True)
        else:
            plt.text(0.5, 0.5, 'No Quality Scores', 
                    horizontalalignment='center', verticalalignment='center')
            plt.title('Quality Scores (Not Available)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Enhanced training history plot saved to {save_path}")
        
        plt.show()


def enhanced_prepare_data(data_dir, seq_len=100, batch_size=32):
    """增强数据准备"""
    import pandas as pd
    
    data_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    all_data = []
    all_labels = []
    
    print(f"Processing {len(data_files)} files...")
    
    for file in data_files:
        df = pd.read_csv(os.path.join(data_dir, file))
        
        # 提取标签
        if 'label' in df.columns:
            label = df['label'].iloc[0]
            df = df.drop('label', axis=1)
        else:
            label = 0
        
        # 数据预处理
        data = df.values.astype(np.float32)
        
        # 处理异常值
        data = np.nan_to_num(data)
        
        # 截断或填充
        if len(data) > seq_len:
            data = data[:seq_len]
        else:
            padding = np.zeros((seq_len - len(data), data.shape[1]))
            data = np.vstack([data, padding])
        
        all_data.append(data)
        all_labels.append(label)
    
    all_data = np.array(all_data)
    all_labels = np.array(all_labels)
    
    # 健壮的归一化
    scaler = MinMaxScaler(feature_range=(0.01, 0.99))  # 避免边界值
    original_shape = all_data.shape
    all_data_flat = all_data.reshape(-1, all_data.shape[-1])
    all_data_scaled = scaler.fit_transform(all_data_flat)
    all_data_scaled = all_data_scaled.reshape(original_shape)
    
    # 数据验证
    print(f"Data range after scaling: [{all_data_scaled.min():.4f}, {all_data_scaled.max():.4f}]")
    print(f"Data shape: {all_data_scaled.shape}")
    
    dataset = TensorDataset(
        torch.FloatTensor(all_data_scaled),
        torch.LongTensor(all_labels)
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    return dataloader, scaler, all_data_scaled.shape, all_data_scaled


if __name__ == "__main__":
    print("Enhanced VAE-GAN for High-Quality Time Series Data Augmentation")
    
    try:
        # 准备数据
        dataloader, scaler, data_shape, original_data = enhanced_prepare_data(
            'original', seq_len=100, batch_size=16)
        print(f"Data loaded: {data_shape}")
        
        # 初始化增强模型
        model = EnhancedVAEGAN(
            input_dim=data_shape[2],
            seq_len=data_shape[1],
            latent_dim=64,
            hidden_dim=128,
            lr_vae=5e-5,      # 较低的VAE学习率
            lr_gen=1e-4,      # 适中的生成器学习率
            lr_disc=1e-4,     # 适中的判别器学习率
            device='cuda'
        )
        
        # 训练模型
        model.train(dataloader, epochs=300, save_interval=25)
        
        # 生成高质量样本
        print("Generating high-quality samples...")
        high_quality_samples = model.generate_high_quality_samples(
            num_samples=100, 
            original_data=original_data,
            quality_threshold=0.7
        )
        print(f"Generated {len(high_quality_samples)} high-quality samples")
        
        # 绘制训练历史
        model.plot_enhanced_training_history('enhanced_training_history.png')
        
        # 保存最终模型
        model.save_models('checkpoints/enhanced_final')
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()