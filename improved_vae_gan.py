#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
改进的VAE-GAN融合模型 - 专门为时序数据增强设计
解决收敛问题，提高生成质量
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
warnings.filterwarnings('ignore')

class TemporalAttention(nn.Module):
    """时序注意力机制 - 捕获时间依赖关系"""
    
    def __init__(self, input_dim, attention_dim=64):
        super(TemporalAttention, self).__init__()
        self.attention_dim = attention_dim
        self.W_q = nn.Linear(input_dim, attention_dim)
        self.W_k = nn.Linear(input_dim, attention_dim)
        self.W_v = nn.Linear(input_dim, input_dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        # x: [batch_size, seq_len, input_dim]
        batch_size, seq_len, input_dim = x.size()
        
        # 计算Query, Key, Value
        Q = self.W_q(x)  # [batch_size, seq_len, attention_dim]
        K = self.W_k(x)  # [batch_size, seq_len, attention_dim]
        V = self.W_v(x)  # [batch_size, seq_len, input_dim]
        
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.attention_dim)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 应用注意力
        attended = torch.matmul(attention_weights, V)
        
        return attended

class ImprovedEncoder(nn.Module):
    """改进的编码器 - 使用LSTM + 注意力机制"""
    
    def __init__(self, input_dim, hidden_dim=128, latent_dim=64, num_layers=2):
        super(ImprovedEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        # LSTM层用于捕获时序特征
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, 
                           batch_first=True, dropout=0.2, bidirectional=True)
        
        # 注意力机制
        self.attention = TemporalAttention(hidden_dim * 2, attention_dim=64)
        
        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        
        # 均值和方差层
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
    def forward(self, x):
        # x: [batch_size, seq_len, input_dim]
        batch_size, seq_len, input_dim = x.size()
        
        # LSTM编码
        lstm_out, _ = self.lstm(x)  # [batch_size, seq_len, hidden_dim*2]
        
        # 应用注意力机制
        attended = self.attention(lstm_out)  # [batch_size, seq_len, hidden_dim*2]
        
        # 全局平均池化
        pooled = torch.mean(attended, dim=1)  # [batch_size, hidden_dim*2]
        
        # 全连接层
        h = self.fc(pooled)  # [batch_size, hidden_dim]
        
        # 计算均值和方差
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        return mu, logvar

class ImprovedDecoder(nn.Module):
    """改进的解码器 - 使用反向LSTM"""
    
    def __init__(self, latent_dim, hidden_dim=128, output_dim=1, seq_len=100, num_layers=2):
        super(ImprovedDecoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.output_dim = output_dim
        
        # 潜在向量到初始隐状态
        self.fc_init = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        
        # LSTM解码器
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, 
                           batch_first=True, dropout=0.2)
        
        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 2, output_dim),
            nn.Tanh()  # 输出归一化到[-1, 1]
        )
        
    def forward(self, z):
        batch_size = z.size(0)
        
        # 初始化
        h = self.fc_init(z)  # [batch_size, hidden_dim]
        
        # 扩展到序列长度
        h_seq = h.unsqueeze(1).repeat(1, self.seq_len, 1)  # [batch_size, seq_len, hidden_dim]
        
        # LSTM解码
        lstm_out, _ = self.lstm(h_seq)  # [batch_size, seq_len, hidden_dim]
        
        # 重塑为2D以应用全连接层
        lstm_out_reshaped = lstm_out.contiguous().view(-1, self.hidden_dim)
        output_reshaped = self.output_layer(lstm_out_reshaped)
        
        # 重塑回序列形状
        output = output_reshaped.view(batch_size, self.seq_len, self.output_dim)
        
        return output

class ImprovedVAE(nn.Module):
    """改进的VAE模型"""
    
    def __init__(self, input_dim, seq_len, hidden_dim=128, latent_dim=64):
        super(ImprovedVAE, self).__init__()
        self.latent_dim = latent_dim
        
        self.encoder = ImprovedEncoder(input_dim, hidden_dim, latent_dim)
        self.decoder = ImprovedDecoder(latent_dim, hidden_dim, input_dim, seq_len)
        
    def reparameterize(self, mu, logvar):
        """重参数化技巧"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decoder(z)
        return recon_x, mu, logvar, z

class ImprovedGenerator(nn.Module):
    """改进的生成器 - 专注于时序数据生成"""
    
    def __init__(self, latent_dim, output_dim, seq_len, hidden_dim=128):
        super(ImprovedGenerator, self).__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        
        # 噪声投影层
        self.noise_projection = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        
        # GRU生成器
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers=2, 
                         batch_first=True, dropout=0.2)
        
        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 2, output_dim),
            nn.Tanh()
        )
        
    def forward(self, z):
        batch_size = z.size(0)
        
        # 投影噪声
        h = self.noise_projection(z)  # [batch_size, hidden_dim]
        
        # 扩展到序列
        h_seq = h.unsqueeze(1).repeat(1, self.seq_len, 1)
        
        # GRU生成
        gru_out, _ = self.gru(h_seq)
        
        # 输出层
        gru_out_reshaped = gru_out.contiguous().view(-1, self.hidden_dim)
        output_reshaped = self.output_layer(gru_out_reshaped)
        output = output_reshaped.view(batch_size, self.seq_len, self.output_dim)
        return output

class ImprovedDiscriminator(nn.Module):
    """改进的判别器 - 使用卷积和注意力机制"""
    
    def __init__(self, input_dim, seq_len, hidden_dim=128):
        super(ImprovedDiscriminator, self).__init__()
        
        # 1D卷积层
        self.conv_layers = nn.Sequential(
            nn.Conv1d(input_dim, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.25),
            
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        
        # 注意力机制
        self.attention = TemporalAttention(128, attention_dim=64)
        
        # 分类层
        # 分类层 - 增加隐藏层容量
        self.classifier = nn.Sequential(
            spectral_norm(nn.Linear(128, hidden_dim * 2)),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.4),
            spectral_norm(nn.Linear(hidden_dim * 2, hidden_dim)),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            spectral_norm(nn.Linear(hidden_dim, 1)),
            nn.Sigmoid()
        )
    def forward(self, x, return_features=False):
        # x: [batch_size, seq_len, input_dim]
        batch_size, seq_len, input_dim = x.size()
        
        # 转置为卷积格式: [batch_size, input_dim, seq_len]
        x_conv = x.transpose(1, 2)
        
        # 卷积特征提取
        conv_out = self.conv_layers(x_conv)  # [batch_size, 128, seq_len]
        
        # 转回时序格式
        conv_out = conv_out.transpose(1, 2)  # [batch_size, seq_len, 128]
        
        # 注意力机制
        attended = self.attention(conv_out)
        
        # 全局平均池化
        pooled = torch.mean(attended, dim=1)  # [batch_size, 128]
        
        if return_features:
            return pooled  # 返回中间特征用于 Feature Matching
        
        # 分类输出
        output = self.classifier(pooled)
        
        return output


class ImprovedVAEGAN:
    """改进的VAE-GAN融合模型"""
    
    def __init__(self, input_dim, seq_len, latent_dim=64, hidden_dim=128, 
                 lr_vae=1e-4, lr_gan=2e-4, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.latent_dim = latent_dim
        
        # 初始化网络
        self.vae = ImprovedVAE(input_dim, seq_len, hidden_dim, latent_dim).to(self.device)
        self.generator = ImprovedGenerator(latent_dim, input_dim, seq_len, hidden_dim).to(self.device)
        self.discriminator = ImprovedDiscriminator(input_dim, seq_len, hidden_dim).to(self.device)
        
        # 优化器
        self.optimizer_vae = optim.Adam(self.vae.parameters(), lr=lr_vae, betas=(0.5, 0.999))
        self.optimizer_gen = optim.Adam(self.generator.parameters(), lr=lr_gan, betas=(0.5, 0.999))
        self.optimizer_disc = optim.Adam(self.discriminator.parameters(), lr=lr_gan, betas=(0.5, 0.999))
        
        # 学习率调度器
        self.scheduler_vae = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer_vae, mode='min', factor=0.8, patience=10, verbose=True)
        self.scheduler_gen = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer_gen, mode='min', factor=0.8, patience=10, verbose=True)
        self.scheduler_disc = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer_disc, mode='min', factor=0.8, patience=10, verbose=True)
        
        # 损失权重
        self.lambda_recon = 1.0
        self.lambda_kl = 0.1
        self.lambda_adv = 0.5
        
        # 训练历史
        self.train_history = {
            'vae_loss': [],
            'recon_loss': [],
            'kl_loss': [],
            'gen_loss': [],
            'disc_loss': []
        }
    
    def vae_loss_function(self, recon_x, x, mu, logvar):
        """VAE损失函数"""
        # 重构损失 (MSE)
        recon_loss = F.mse_loss(recon_x, x, reduction='mean')
        
        # KL散度损失
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
        
        return recon_loss, kl_loss
    
    def adversarial_loss(self, output, target):
        """对抗损失 (二元交叉熵)"""
        return F.binary_cross_entropy(output, target)
    
    def train_step(self, real_data):
        """单步训练"""
        batch_size = real_data.size(0)
        z = torch.randn(batch_size, self.latent_dim).to(self.device)
        fake_data = self.generator(z)
        #if fake_data.min() < 0 or fake_data.max() > 1:
            #print(f"⚠️ fake_data out of range: min={fake_data.min().item()}, max={fake_data.max().item()}")
        real_labels = torch.ones(batch_size, 1).to(self.device)
        fake_labels = torch.zeros(batch_size, 1).to(self.device)
        #print("real_data:", real_data.min().item(), real_data.max().item())

        # ================== 训练VAE ==================
        self.optimizer_vae.zero_grad()
        
        recon_data, mu, logvar, z = self.vae(real_data)
        recon_loss, kl_loss = self.vae_loss_function(recon_data, real_data, mu, logvar)
        
        vae_loss = self.lambda_recon * recon_loss + self.lambda_kl * kl_loss
        vae_loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.vae.parameters(), max_norm=1.0)
        self.optimizer_vae.step()
        
        # ================== 训练判别器 ==================
        self.optimizer_disc.zero_grad()
        
        # 真实数据
        real_validity = self.discriminator(real_data)
        real_loss = self.adversarial_loss(real_validity, real_labels)
        
        # 生成数据
        z_noise = torch.randn(batch_size, self.latent_dim).to(self.device)
        fake_data = self.generator(z_noise)
        fake_validity = self.discriminator(fake_data.detach())
        fake_loss = self.adversarial_loss(fake_validity, fake_labels)
        
        disc_loss = (real_loss + fake_loss) / 2
        disc_loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=1.0)
        self.optimizer_disc.step()
        
        # ================== 训练生成器 ==================
        self.optimizer_gen.zero_grad()
        
        z_noise = torch.randn(batch_size, self.latent_dim).to(self.device)
        fake_data = self.generator(z_noise)
        fake_validity = self.discriminator(fake_data)
        
        gen_loss = self.adversarial_loss(fake_validity, real_labels)
        
        # 添加特征匹配损失
        with torch.no_grad():
            real_features = self.discriminator(real_data, return_features=True)
        fake_features = self.discriminator(fake_data, return_features=True)
        feature_loss = F.mse_loss(fake_features, real_features)


        
        total_gen_loss = gen_loss + 0.1 * feature_loss
        total_gen_loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=1.0)
        self.optimizer_gen.step()
        
        # 记录损失
        self.train_history['vae_loss'].append(vae_loss.item())
        self.train_history['recon_loss'].append(recon_loss.item())
        self.train_history['kl_loss'].append(kl_loss.item())
        self.train_history['gen_loss'].append(total_gen_loss.item())
        self.train_history['disc_loss'].append(disc_loss.item())
        
        return {
            'vae_loss': vae_loss.item(),
            'recon_loss': recon_loss.item(),
            'kl_loss': kl_loss.item(),
            'gen_loss': total_gen_loss.item(),
            'disc_loss': disc_loss.item()
        }
    
    def train(self, dataloader, epochs=100, save_interval=10):
        """训练模型"""
        print(f"Starting training on {self.device}")
        print(f"Data shape: batch_size x {self.seq_len} x {self.input_dim}")
        
        for epoch in range(epochs):
            epoch_losses = {
                'vae_loss': [],
                'recon_loss': [],
                'kl_loss': [],
                'gen_loss': [],
                'disc_loss': []
            }
            
            for batch_idx, (data, _) in enumerate(dataloader):
                data = data.to(self.device)
                
                # 如果数据是2D，需要添加特征维度
                if len(data.shape) == 2:
                    data = data.unsqueeze(-1)
                
                losses = self.train_step(data)
                
                for key, value in losses.items():
                    epoch_losses[key].append(value)
            
            # 计算平均损失
            avg_losses = {key: np.mean(values) for key, values in epoch_losses.items()}
            
            # 更新学习率
            self.scheduler_vae.step(avg_losses['vae_loss'])
            self.scheduler_gen.step(avg_losses['gen_loss'])
            self.scheduler_disc.step(avg_losses['disc_loss'])
            
            # 打印进度
            if epoch % 10 == 0:
                print(f"Epoch [{epoch}/{epochs}]:")
                for key, value in avg_losses.items():
                    print(f"  {key}: {value:.6f}")
                print()
            
            # 保存模型
            if epoch % save_interval == 0 and epoch > 0:
                self.save_models(f"models/epoch_{epoch}")
        
        print("Training completed!")
    
    def generate_samples(self, num_samples, temperature=1.0):
        """生成样本"""
        self.vae.eval()
        self.generator.eval()
        
        with torch.no_grad():
            # 使用生成器生成
            z = torch.randn(num_samples, self.latent_dim).to(self.device) * temperature
            generated = self.generator(z)
            
            # 也可以使用VAE生成
            z_vae = torch.randn(num_samples, self.latent_dim).to(self.device) * temperature
            generated_vae = self.vae.decoder(z_vae)
            
        self.vae.train()
        self.generator.train()
        
        return generated.cpu().numpy(), generated_vae.cpu().numpy()
    
    def save_models(self, path):
        """保存模型"""
        import os
        os.makedirs(path, exist_ok=True)
        
        torch.save(self.vae.state_dict(), f"{path}/vae.pth")
        torch.save(self.generator.state_dict(), f"{path}/generator.pth")
        torch.save(self.discriminator.state_dict(), f"{path}/discriminator.pth")
        
        # 保存训练历史
        np.save(f"{path}/train_history.npy", self.train_history)
        
        print(f"Models saved to {path}")
    
    def load_models(self, path):
        """加载模型"""
        self.vae.load_state_dict(torch.load(f"{path}/vae.pth"))
        self.generator.load_state_dict(torch.load(f"{path}/generator.pth"))
        self.discriminator.load_state_dict(torch.load(f"{path}/discriminator.pth"))
        
        # 加载训练历史
        try:
            self.train_history = np.load(f"{path}/train_history.npy", allow_pickle=True).item()
        except:
            pass
        
        print(f"Models loaded from {path}")
    
    def plot_training_history(self, save_path=None):
        """绘制训练历史"""
        plt.figure(figsize=(15, 10))
        
        # VAE相关损失
        plt.subplot(2, 3, 1)
        plt.plot(self.train_history['vae_loss'])
        plt.title('VAE Loss')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.grid(True)
        
        plt.subplot(2, 3, 2)
        plt.plot(self.train_history['recon_loss'])
        plt.title('Reconstruction Loss')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.grid(True)
        
        plt.subplot(2, 3, 3)
        plt.plot(self.train_history['kl_loss'])
        plt.title('KL Divergence Loss')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.grid(True)
        
        # GAN相关损失
        plt.subplot(2, 3, 4)
        plt.plot(self.train_history['gen_loss'])
        plt.title('Generator Loss')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.grid(True)
        
        plt.subplot(2, 3, 5)
        plt.plot(self.train_history['disc_loss'])
        plt.title('Discriminator Loss')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.grid(True)
        
        # 所有损失对比
        plt.subplot(2, 3, 6)
        plt.plot(self.train_history['vae_loss'], label='VAE Loss')
        plt.plot(self.train_history['gen_loss'], label='Generator Loss')
        plt.plot(self.train_history['disc_loss'], label='Discriminator Loss')
        plt.title('All Losses')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training history plot saved to {save_path}")
        
        plt.show()

def prepare_time_series_data(data_dir, seq_len=100, batch_size=32):
    """准备时序数据"""
    import os
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler
    
    # 加载数据
    data_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    all_data = []
    all_labels = []
    
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
        
        # 截断或填充到固定长度
        if len(data) > seq_len:
            data = data[:seq_len]
        else:
            padding = np.zeros((seq_len - len(data), data.shape[1]))
            data = np.vstack([data, padding])
        
        all_data.append(data)
        all_labels.append(label)
    
    # 转换为numpy数组
    all_data = np.array(all_data)
    all_labels = np.array(all_labels)
    
    # 检查NaN值
    if np.isnan(all_data).any():
        print("⚠️ 输入数据包含NaN值，将用0填充")
        all_data = np.nan_to_num(all_data)
    
    # 归一化到[0,1]范围
    scaler = MinMaxScaler(feature_range=(0, 1))
    original_shape = all_data.shape
    all_data_flat = all_data.reshape(-1, all_data.shape[-1])
    all_data_scaled = scaler.fit_transform(all_data_flat)
    all_data_scaled = all_data_scaled.reshape(original_shape)
    
    # 验证数据范围
    if (all_data_scaled.min() < 0 or all_data_scaled.max() > 1):
        print(f"⚠️ 归一化后数据超出范围: min={all_data_scaled.min()}, max={all_data_scaled.max()}")
        all_data_scaled = np.clip(all_data_scaled, 0, 1)
    
    # 创建数据加载器
    dataset = TensorDataset(
        torch.FloatTensor(all_data_scaled),
        torch.LongTensor(all_labels)
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return dataloader, scaler, all_data.shape

if __name__ == "__main__":
    # 示例使用
    print("Improved VAE-GAN for Time Series Data Augmentation")
    
    # 准备数据 (假设原始数据在original目录下)
    try:
        dataloader, scaler, data_shape = prepare_time_series_data('original', seq_len=100, batch_size=16)
        print(f"Data loaded: {data_shape}")
        
        # 初始化模型
        model = ImprovedVAEGAN(
            input_dim=data_shape[2],  # 特征维度
            seq_len=data_shape[1],    # 序列长度
            latent_dim=64,
            hidden_dim=128,
            lr_vae=1e-4,
            lr_gan=2e-4,
            device='cuda'
        )
        
        # 训练模型
        model.train(dataloader, epochs=200, save_interval=20)
        
        # 生成样本
        generated_samples, generated_vae_samples = model.generate_samples(num_samples=50)
        print(f"Generated samples shape: {generated_samples.shape}")
        
        # 绘制训练历史
        model.plot_training_history('training_history.png')
        
    except Exception as e:
        print(f"Error: {e}")
        print("Please make sure the 'original' directory exists and contains CSV files.")