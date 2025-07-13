#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
优化的VAE-GAN模型 - 解决现有问题
主要改进:
1. 修复KL散度塌陷问题 (beta-VAE + 渐进式KL权重)
2. 改进损失函数平衡 (自适应权重)
3. 增强时序建模 (LSTM + 自注意力)
4. 数据质量筛选机制
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings('ignore')

class OptimizedVAEGAN(nn.Module):
    """优化的VAE-GAN模型"""
    
    def __init__(self, input_dim, seq_len, latent_dim=64, hidden_dim=128, device='cpu'):
        super(OptimizedVAEGAN, self).__init__()
        
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.device = device
        
        # VAE编码器 (LSTM + 注意力)
        self.encoder_lstm = nn.LSTM(input_dim, hidden_dim, 2, batch_first=True, dropout=0.2)
        self.encoder_attention = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
        self.encoder_norm = nn.LayerNorm(hidden_dim)
        
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # VAE解码器 (LSTM + 残差连接)
        self.decoder_fc = nn.Linear(latent_dim, hidden_dim)
        self.decoder_lstm = nn.LSTM(hidden_dim, hidden_dim, 2, batch_first=True, dropout=0.2)
        self.decoder_out = nn.Linear(hidden_dim, input_dim)
        
        # GAN生成器 (改进架构)
        self.generator = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(True),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_dim * 2, hidden_dim * 4),
            nn.BatchNorm1d(hidden_dim * 4),
            nn.ReLU(True),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_dim * 4, seq_len * input_dim),
            nn.Tanh()
        )
        
        # 判别器 (谱归一化 + 梯度惩罚)
        self.discriminator = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(seq_len * input_dim, hidden_dim * 4)),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(0.3),
            
            nn.utils.spectral_norm(nn.Linear(hidden_dim * 4, hidden_dim * 2)),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(0.3),
            
            nn.utils.spectral_norm(nn.Linear(hidden_dim * 2, hidden_dim)),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        self.to(device)
        
        # 优化器 (不同学习率)
        self.vae_optimizer = optim.AdamW(
            list(self.encoder_lstm.parameters()) + 
            list(self.encoder_attention.parameters()) +
            list(self.fc_mu.parameters()) + 
            list(self.fc_logvar.parameters()) +
            list(self.decoder_fc.parameters()) +
            list(self.decoder_lstm.parameters()) +
            list(self.decoder_out.parameters()),
            lr=1e-4, weight_decay=1e-5
        )
        
        self.gen_optimizer = optim.AdamW(
            self.generator.parameters(), 
            lr=2e-4, weight_decay=1e-5
        )
        
        self.disc_optimizer = optim.AdamW(
            self.discriminator.parameters(), 
            lr=1e-4, weight_decay=1e-5
        )
        
        # 学习率调度器
        self.vae_scheduler = optim.lr_scheduler.CosineAnnealingLR(self.vae_optimizer, T_max=200)
        self.gen_scheduler = optim.lr_scheduler.CosineAnnealingLR(self.gen_optimizer, T_max=200)
        self.disc_scheduler = optim.lr_scheduler.CosineAnnealingLR(self.disc_optimizer, T_max=200)
        
        # 训练历史
        self.training_history = {
            'vae_loss': [], 'recon_loss': [], 'kl_loss': [],
            'gen_loss': [], 'disc_loss': [],
            'beta': []
        }
        
    def encode(self, x):
        """VAE编码"""
        # LSTM编码
        lstm_out, _ = self.encoder_lstm(x)
        
        # 自注意力
        attn_out, _ = self.encoder_attention(lstm_out, lstm_out, lstm_out)
        attn_out = self.encoder_norm(attn_out + lstm_out)
        
        # 取最后时间步
        h = attn_out[:, -1, :]
        
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """重参数化"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        """VAE解码"""
        h = F.relu(self.decoder_fc(z))
        h = h.unsqueeze(1).repeat(1, self.seq_len, 1)
        
        lstm_out, _ = self.decoder_lstm(h)
        output = self.decoder_out(lstm_out)
        return output
    
    def generate(self, z):
        """GAN生成"""
        output = self.generator(z)
        return output.view(-1, self.seq_len, self.input_dim)
    
    def discriminate(self, x):
        """判别"""
        x_flat = x.view(x.size(0), -1)
        return self.discriminator(x_flat)
    
    def compute_gradient_penalty(self, real_data, fake_data, lambda_gp=10):
        """梯度惩罚"""
        batch_size = real_data.size(0)
        alpha = torch.rand(batch_size, 1, 1).to(self.device)
        
        interpolates = alpha * real_data + (1 - alpha) * fake_data
        interpolates.requires_grad_(True)
        
        disc_interpolates = self.discriminate(interpolates)
        
        gradients = torch.autograd.grad(
            outputs=disc_interpolates, inputs=interpolates,
            grad_outputs=torch.ones_like(disc_interpolates),
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]
        
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return lambda_gp * gradient_penalty
    
    def train_step(self, real_data, epoch, total_epochs):
        """单步训练"""
        batch_size = real_data.size(0)
        
        # 动态beta (解决KL塌陷)
        beta = min(1.0, 0.001 + (epoch / total_epochs) * 0.999)
        
        # 自适应损失权重
        lambda_recon = 1.0 + 0.5 * np.sin(epoch / 50)  # 周期性调整
        lambda_adv = 0.1 + 0.05 * (epoch / total_epochs)
        
        # ========== 训练VAE ==========
        self.vae_optimizer.zero_grad()
        
        mu, logvar = self.encode(real_data)
        z = self.reparameterize(mu, logvar)
        recon_data = self.decode(z)
        
        # VAE损失
        recon_loss = F.mse_loss(recon_data, real_data, reduction='mean')
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        vae_loss = lambda_recon * recon_loss + beta * kl_loss
        
        vae_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.encoder_lstm.parameters()) + 
            list(self.encoder_attention.parameters()) +
            list(self.fc_mu.parameters()) + 
            list(self.fc_logvar.parameters()) +
            list(self.decoder_fc.parameters()) +
            list(self.decoder_lstm.parameters()) +
            list(self.decoder_out.parameters()), 
            max_norm=1.0
        )
        self.vae_optimizer.step()
        
        # ========== 训练判别器 ==========
        self.disc_optimizer.zero_grad()
        
        # 真实数据
        real_pred = self.discriminate(real_data)
        real_loss = F.binary_cross_entropy(real_pred, torch.ones_like(real_pred))
        
        # 生成数据 (从VAE和GAN)
        with torch.no_grad():
            z_vae = self.reparameterize(mu, logvar)
            fake_vae = self.decode(z_vae)
            
            z_gan = torch.randn(batch_size, self.latent_dim).to(self.device)
            fake_gan = self.generate(z_gan)
        
        fake_pred_vae = self.discriminate(fake_vae.detach())
        fake_pred_gan = self.discriminate(fake_gan.detach())
        
        fake_loss = (F.binary_cross_entropy(fake_pred_vae, torch.zeros_like(fake_pred_vae)) +
                    F.binary_cross_entropy(fake_pred_gan, torch.zeros_like(fake_pred_gan))) / 2
        
        # 梯度惩罚
        gp_vae = self.compute_gradient_penalty(real_data, fake_vae)
        gp_gan = self.compute_gradient_penalty(real_data, fake_gan)
        
        disc_loss = real_loss + fake_loss + (gp_vae + gp_gan) / 2
        disc_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=1.0)
        self.disc_optimizer.step()
        
        # ========== 训练生成器 ==========
        self.gen_optimizer.zero_grad()
        
        z_gan = torch.randn(batch_size, self.latent_dim).to(self.device)
        fake_gan = self.generate(z_gan)
        fake_pred = self.discriminate(fake_gan)
        
        gen_loss = F.binary_cross_entropy(fake_pred, torch.ones_like(fake_pred))
        
        # 特征匹配损失
        real_features = self.discriminator[:-2](real_data.view(batch_size, -1))
        fake_features = self.discriminator[:-2](fake_gan.view(batch_size, -1))
        feature_loss = F.mse_loss(fake_features, real_features.detach())
        
        total_gen_loss = gen_loss + 0.1 * feature_loss
        total_gen_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=1.0)
        self.gen_optimizer.step()
        
        # 记录历史
        self.training_history['vae_loss'].append(vae_loss.item())
        self.training_history['recon_loss'].append(recon_loss.item())
        self.training_history['kl_loss'].append(kl_loss.item())
        self.training_history['gen_loss'].append(total_gen_loss.item())
        self.training_history['disc_loss'].append(disc_loss.item())
        self.training_history['beta'].append(beta)
        
        return {
            'vae_loss': vae_loss.item(),
            'recon_loss': recon_loss.item(),
            'kl_loss': kl_loss.item(),
            'gen_loss': total_gen_loss.item(),
            'disc_loss': disc_loss.item(),
            'beta': beta
        }
    
    def train_model(self, dataloader, epochs=300, save_interval=50):
        """训练模型"""
        print(f"开始优化训练，设备: {self.device}")
        print(f"训练轮数: {epochs}")
        
        for epoch in range(epochs):
            epoch_losses = []
            
            for batch_idx, (data, _) in enumerate(dataloader):
                data = data.to(self.device)
                losses = self.train_step(data, epoch, epochs)
                epoch_losses.append(losses)
            
            # 计算平均损失
            avg_losses = {key: np.mean([l[key] for l in epoch_losses]) for key in epoch_losses[0].keys()}
            
            if epoch % 10 == 0:
                print(f"Epoch [{epoch}/{epochs}]:")
                print(f"  VAE Loss: {avg_losses['vae_loss']:.6f}")
                print(f"  Recon Loss: {avg_losses['recon_loss']:.6f}")
                print(f"  KL Loss: {avg_losses['kl_loss']:.6f}")
                print(f"  Gen Loss: {avg_losses['gen_loss']:.6f}")
                print(f"  Disc Loss: {avg_losses['disc_loss']:.6f}")
                print(f"  Beta: {avg_losses['beta']:.4f}")
            
            # 更新学习率
            self.vae_scheduler.step()
            self.gen_scheduler.step()
            self.disc_scheduler.step()
            
            # 保存检查点
            if (epoch + 1) % save_interval == 0:
                self.save_checkpoint(f'checkpoints/optimized_epoch_{epoch+1}')
        
        print("优化训练完成!")
    
    def generate_samples(self, num_samples=100, temperature=1.0):
        """生成样本"""
        self.eval()
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim).to(self.device) * temperature
            
            # VAE生成
            vae_samples = self.decode(z)
            
            # GAN生成
            gan_samples = self.generate(z)
            
            # 质量筛选 (选择判别器分数适中的样本)
            vae_scores = self.discriminate(vae_samples).squeeze()
            gan_scores = self.discriminate(gan_samples).squeeze()
            
            # 选择分数在0.3-0.7之间的样本 (既不太假也不太真)
            vae_mask = (vae_scores > 0.3) & (vae_scores < 0.7)
            gan_mask = (gan_scores > 0.3) & (gan_scores < 0.7)
            
            if vae_mask.sum() > 0:
                vae_samples = vae_samples[vae_mask]
            if gan_mask.sum() > 0:
                gan_samples = gan_samples[gan_mask]
        
        self.train()
        return vae_samples.cpu().numpy(), gan_samples.cpu().numpy()
    
    def save_checkpoint(self, path):
        """保存模型"""
        import os
        os.makedirs(path, exist_ok=True)
        
        torch.save({
            'encoder_lstm': self.encoder_lstm.state_dict(),
            'encoder_attention': self.encoder_attention.state_dict(),
            'fc_mu': self.fc_mu.state_dict(),
            'fc_logvar': self.fc_logvar.state_dict(),
            'decoder_fc': self.decoder_fc.state_dict(),
            'decoder_lstm': self.decoder_lstm.state_dict(),
            'decoder_out': self.decoder_out.state_dict(),
            'generator': self.generator.state_dict(),
            'discriminator': self.discriminator.state_dict(),
            'training_history': self.training_history
        }, os.path.join(path, 'optimized_model.pth'))
        
        print(f"模型保存到: {path}")
    
    def plot_training_history(self, save_path):
        """绘制训练历史"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # VAE损失
        axes[0, 0].plot(self.training_history['vae_loss'])
        axes[0, 0].set_title('VAE Loss')
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].grid(True)
        
        # 重构损失
        axes[0, 1].plot(self.training_history['recon_loss'])
        axes[0, 1].set_title('Reconstruction Loss')
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].grid(True)
        
        # KL损失
        axes[0, 2].plot(self.training_history['kl_loss'])
        axes[0, 2].set_title('KL Divergence Loss')
        axes[0, 2].set_xlabel('Iteration')
        axes[0, 2].grid(True)
        
        # 生成器损失
        axes[1, 0].plot(self.training_history['gen_loss'])
        axes[1, 0].set_title('Generator Loss')
        axes[1, 0].set_xlabel('Iteration')
        axes[1, 0].grid(True)
        
        # 判别器损失
        axes[1, 1].plot(self.training_history['disc_loss'])
        axes[1, 1].set_title('Discriminator Loss')
        axes[1, 1].set_xlabel('Iteration')
        axes[1, 1].grid(True)
        
        # Beta值
        axes[1, 2].plot(self.training_history['beta'])
        axes[1, 2].set_title('Beta (KL Weight)')
        axes[1, 2].set_xlabel('Iteration')
        axes[1, 2].grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"训练历史图保存到: {save_path}")