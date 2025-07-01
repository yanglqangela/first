import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
from tqdm import tqdm

class ClassEncoder(nn.Module):
    """类别条件编码器"""
    def __init__(self, input_dim, hidden_dims, latent_dim, num_classes, dropout_rate=0.1):
        super(ClassEncoder, self).__init__()
        
        # 类别嵌入
        self.class_embedding = nn.Embedding(num_classes, 32)
        
        # 构建编码器网络
        modules = []
        
        # 输入层
        modules.append(nn.Linear(input_dim + 32, hidden_dims[0]))
        modules.append(nn.LayerNorm(hidden_dims[0]))
        modules.append(nn.LeakyReLU())
        modules.append(nn.Dropout(dropout_rate))
        
        # 隐藏层
        for i in range(len(hidden_dims) - 1):
            modules.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            modules.append(nn.LayerNorm(hidden_dims[i + 1]))
            modules.append(nn.LeakyReLU())
            modules.append(nn.Dropout(dropout_rate))
        
        self.encoder = nn.Sequential(*modules)
        
        # 均值和对数方差
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)
        
    def forward(self, x, labels):
        # 获取类别嵌入
        class_embed = self.class_embedding(labels)
        
        # 连接输入和类别嵌入
        x_concat = torch.cat([x, class_embed], dim=1)
        
        # 编码
        hidden = self.encoder(x_concat)
        
        # 生成均值和对数方差
        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)
        
        return mu, logvar

class ClassDecoder(nn.Module):
    """类别条件解码器"""
    def __init__(self, latent_dim, hidden_dims, output_dim, num_classes, dropout_rate=0.1):
        super(ClassDecoder, self).__init__()
        
        # 类别嵌入
        self.class_embedding = nn.Embedding(num_classes, 32)
        
        # 构建解码器网络
        modules = []
        
        # 输入层
        modules.append(nn.Linear(latent_dim + 32, hidden_dims[-1]))
        modules.append(nn.LayerNorm(hidden_dims[-1]))
        modules.append(nn.LeakyReLU())
        modules.append(nn.Dropout(dropout_rate))
        
        # 隐藏层
        for i in range(len(hidden_dims) - 1, 0, -1):
            modules.append(nn.Linear(hidden_dims[i], hidden_dims[i - 1]))
            modules.append(nn.LayerNorm(hidden_dims[i - 1]))
            modules.append(nn.LeakyReLU())
            modules.append(nn.Dropout(dropout_rate))
        
        # 输出层
        modules.append(nn.Linear(hidden_dims[0], output_dim))
        
        self.decoder = nn.Sequential(*modules)
        
    def forward(self, z, labels):
        # 获取类别嵌入
        class_embed = self.class_embedding(labels)
        
        # 连接潜在向量和类别嵌入
        z_concat = torch.cat([z, class_embed], dim=1)
        
        # 解码
        x_hat = self.decoder(z_concat)
        
        return x_hat

class ClassConditionalVAE(nn.Module):
    """类别条件VAE"""
    def __init__(self, input_dim, hidden_dims, latent_dim, num_classes, dropout_rate=0.1):
        super(ClassConditionalVAE, self).__init__()
        
        self.latent_dim = latent_dim
        
        # 编码器和解码器
        self.encoder = ClassEncoder(input_dim, hidden_dims, latent_dim, num_classes, dropout_rate)
        self.decoder = ClassDecoder(latent_dim, hidden_dims, input_dim, num_classes, dropout_rate)
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def encode(self, x, labels):
        mu, logvar = self.encoder(x, labels)
        return mu, logvar
    
    def decode(self, z, labels):
        return self.decoder(z, labels)
    
    def forward(self, x, labels):
        mu, logvar = self.encode(x, labels)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z, labels)
        return x_hat, mu, logvar

class ClassGenerator(nn.Module):
    """类别条件生成器"""
    def __init__(self, latent_dim, hidden_dims, output_dim, num_classes, dropout_rate=0.1):
        super(ClassGenerator, self).__init__()
        
        # 类别嵌入
        self.class_embedding = nn.Embedding(num_classes, 32)
        
        # 构建生成器网络
        modules = []
        
        # 输入层
        modules.append(nn.Linear(latent_dim + 32, hidden_dims[0]))
        modules.append(nn.LayerNorm(hidden_dims[0]))
        modules.append(nn.LeakyReLU())
        modules.append(nn.Dropout(dropout_rate))
        
        # 隐藏层
        for i in range(len(hidden_dims) - 1):
            modules.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            modules.append(nn.LayerNorm(hidden_dims[i + 1]))
            modules.append(nn.LeakyReLU())
            modules.append(nn.Dropout(dropout_rate))
        
        # 输出层
        modules.append(nn.Linear(hidden_dims[-1], output_dim))
        
        self.model = nn.Sequential(*modules)
        
    def forward(self, z, labels):
        # 获取类别嵌入
        class_embed = self.class_embedding(labels)
        
        # 连接潜在向量和类别嵌入
        z_concat = torch.cat([z, class_embed], dim=1)
        
        # 生成数据
        output = self.model(z_concat)
        
        return output

class ClassDiscriminator(nn.Module):
    """类别条件判别器"""
    def __init__(self, input_dim, hidden_dims, num_classes, dropout_rate=0.1):
        super(ClassDiscriminator, self).__init__()
        
        # 类别嵌入
        self.class_embedding = nn.Embedding(num_classes, 32)
        
        # 构建判别器网络
        modules = []
        
        # 输入层
        modules.append(nn.Linear(input_dim + 32, hidden_dims[0]))
        modules.append(nn.LeakyReLU())
        modules.append(nn.Dropout(dropout_rate))
        
        # 隐藏层
        for i in range(len(hidden_dims) - 1):
            modules.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            modules.append(nn.LeakyReLU())
            modules.append(nn.Dropout(dropout_rate))
        
        self.main_model = nn.Sequential(*modules)
        
        # 输出层
        self.validity = nn.Linear(hidden_dims[-1], 1)
        self.auxiliary = nn.Linear(hidden_dims[-1], num_classes)
    
    def forward(self, x, labels=None):
        if labels is not None:
            class_embed = self.class_embedding(labels)
            x_concat = torch.cat([x, class_embed], dim=1)
        else:
            batch_size = x.size(0)
            class_embed = torch.zeros(batch_size, 32).to(x.device)
            x_concat = torch.cat([x, class_embed], dim=1)
        
        features = self.main_model(x_concat)
        validity = self.validity(features)
        cls_pred = self.auxiliary(features)
        
        return validity, cls_pred, features

class ClassConditionalVAEGAN:
    """类别条件VAE-GAN"""
    def __init__(self, input_dim, latent_dim, hidden_dims_vae, hidden_dims_gen, hidden_dims_disc, 
                 num_classes, dropout_rate=0.1, device='cuda'):
        # 设置设备
        self.device = device
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        # 创建模型
        self.vae = ClassConditionalVAE(input_dim, hidden_dims_vae, latent_dim, num_classes, dropout_rate).to(device)
        self.generator = ClassGenerator(latent_dim, hidden_dims_gen, input_dim, num_classes, dropout_rate).to(device)
        self.discriminator = ClassDiscriminator(input_dim, hidden_dims_disc, num_classes, dropout_rate).to(device)
        
        # 创建优化器
        self.optimizer_vae = optim.Adam(self.vae.parameters(), lr=0.0001, betas=(0.5, 0.9))
        self.optimizer_gen = optim.Adam(self.generator.parameters(), lr=0.0001, betas=(0.5, 0.9))
        self.optimizer_disc = optim.Adam(self.discriminator.parameters(), lr=0.0001, betas=(0.5, 0.9))
        
    def train(self, dataloader, epochs=100, lambda_kl=1.0, lambda_rec=10.0, 
              lambda_gp=10.0, lambda_cls=1.0, class_weights=None, save_dir=None, kl_anneal_epochs=0):
        """训练模型"""
        # 创建保存目录
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        # 设置类别权重，并确保其类型为Float
        if class_weights is None:
            class_weights = torch.ones(self.num_classes, dtype=torch.float32).to(self.device)
        else:
            class_weights = torch.tensor(class_weights, dtype=torch.float32).to(self.device)
        
        # 训练历史记录
        history = {
            'vae_loss': [], 'gen_loss': [], 'disc_loss': [],
            'rec_loss': [], 'kl_loss': [], 'cls_loss': []
        }
        
        # 训练循环
        for epoch in range(epochs):
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
            
            # KL退火：在前kl_anneal_epochs个周期内，KL权重从0线性增加到lambda_kl
            if kl_anneal_epochs > 0:
                current_lambda_kl = lambda_kl * min(1.0, (epoch + 1) / kl_anneal_epochs)
            else:
                current_lambda_kl = lambda_kl

            vae_losses, gen_losses, disc_losses, rec_losses, kl_losses, cls_losses = [], [], [], [], [], []

            for batch_idx, batch in enumerate(pbar):
                real_data, labels = batch
                real_data = real_data.to(self.device)
                labels = labels.to(self.device)
                batch_size = real_data.size(0)
                
                # --- 训练判别器 ---
                self.optimizer_disc.zero_grad()
                
                z = torch.randn(batch_size, self.latent_dim, device=self.device)
                with torch.no_grad():
                    fake_data = self.generator(z, labels).detach()
                
                real_validity, real_cls, _ = self.discriminator(real_data, labels)
                fake_validity, _, _ = self.discriminator(fake_data, labels)

                # WGAN-GP损失
                wasserstein_distance = fake_validity.mean() - real_validity.mean()
                
                # 计算梯度惩罚
                alpha = torch.rand(batch_size, 1, device=self.device)
                interpolates = (alpha * real_data.data + (1 - alpha) * fake_data.data).requires_grad_(True)
                d_interpolates, _, _ = self.discriminator(interpolates, labels)
                
                fake_grad_outputs = torch.ones_like(d_interpolates, device=self.device)
                gradients = torch.autograd.grad(
                    outputs=d_interpolates,
                    inputs=interpolates,
                    grad_outputs=fake_grad_outputs,
                    create_graph=True,
                    retain_graph=True,
                    only_inputs=True
                )[0]
                
                gradients = gradients.view(batch_size, -1)
                gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
                
                disc_cls_loss = F.cross_entropy(real_cls, labels, weight=class_weights)
                
                disc_loss = wasserstein_distance + lambda_gp * gradient_penalty + lambda_cls * disc_cls_loss

                if torch.isfinite(disc_loss):
                    disc_loss.backward()
                    self.optimizer_disc.step()
                    disc_losses.append(disc_loss.item())
                    cls_losses.append(disc_cls_loss.item())

                # --- 训练生成器和VAE (降低频率) ---
                if batch_idx % 5 == 0:
                    # --- 训练生成器 ---
                    self.optimizer_gen.zero_grad()
                    
                    z = torch.randn(batch_size, self.latent_dim, device=self.device)
                    fake_data_for_gen = self.generator(z, labels)
                    fake_validity_for_gen, _, _ = self.discriminator(fake_data_for_gen, labels)
                    
                    # 生成器的目标是最大化其生成样本的"真实性"得分
                    gen_loss = -torch.mean(fake_validity_for_gen)

                    if torch.isfinite(gen_loss):
                        gen_loss.backward()
                        self.optimizer_gen.step()
                        gen_losses.append(gen_loss.item())
                    
                    # --- 训练VAE ---
                    self.optimizer_vae.zero_grad()

                    recon_batch, mu, logvar = self.vae(real_data, labels)
                    
                    recon_loss = F.mse_loss(recon_batch, real_data)
                    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                    
                    # 使用退火后的KL权重
                    vae_loss = lambda_rec * recon_loss + current_lambda_kl * kl_loss

                    if torch.isfinite(vae_loss):
                        vae_loss.backward()
                        self.optimizer_vae.step()
                        vae_losses.append(vae_loss.item())
                        rec_losses.append(recon_loss.item())
                        kl_losses.append(kl_loss.item())

                pbar.set_postfix({
                    'D_loss': f'{np.mean(disc_losses):.4f}' if disc_losses else 'N/A',
                    'G_loss': f'{np.mean(gen_losses):.4f}' if gen_losses else 'N/A',
                    'VAE_loss': f'{np.mean(vae_losses):.4f}' if vae_losses else 'N/A',
                })

            history['vae_loss'].append(np.mean(vae_losses) if vae_losses else 0)
            history['gen_loss'].append(np.mean(gen_losses) if gen_losses else 0)
            history['disc_loss'].append(np.mean(disc_losses) if disc_losses else 0)
            history['rec_loss'].append(np.mean(rec_losses) if rec_losses else 0)
            history['kl_loss'].append(np.mean(kl_losses) if kl_losses else 0)
            history['cls_loss'].append(np.mean(cls_losses) if cls_losses else 0)
            
            print(f"Epoch {epoch+1}/{epochs} - "
                  f"VAE: {history['vae_loss'][-1]:.4f}, "
                  f"Gen: {history['gen_loss'][-1]:.4f}, "
                  f"Disc: {history['disc_loss'][-1]:.4f}, "
                  f"Cls: {history['cls_loss'][-1]:.4f}")
            
            # 保存检查点
            if save_dir and (epoch + 1) % 10 == 0:
                checkpoint = {
                    'vae_state_dict': self.vae.state_dict(),
                    'gen_state_dict': self.generator.state_dict(),
                    'disc_state_dict': self.discriminator.state_dict(),
                    'epoch': epoch
                }
                torch.save(checkpoint, os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pt'))
        
        # 打印最后一个epoch的损失
        print(f"Final Loss - D: {history['disc_loss'][-1]:.4f}, "
              f"G: {history['gen_loss'][-1]:.4f}, "
              f"VAE: {history['vae_loss'][-1]:.4f}, "
              f"REC: {history['rec_loss'][-1]:.4f}, "
              f"KL: {history['kl_loss'][-1]:.4f}, "
              f"CLS: {history['cls_loss'][-1]:.4f}")

        # 保存模型
        if save_dir:
            final_model_path = os.path.join(save_dir, 'final_model.pth')
            self.save(final_model_path)
            print(f"Final model saved to {final_model_path}")
        
        return history
    
    def generate(self, num_samples, labels=None, device=None):
        """生成指定数量的样本"""
        if device is None:
            device = self.device
            
        if labels is None:
            labels = torch.randint(0, self.num_classes, (num_samples,)).to(device)
        elif isinstance(labels, int):
            labels = torch.full((num_samples,), labels).to(device)
        
        # 生成随机噪声
        z = torch.randn(num_samples, self.latent_dim).to(device)
        
        # 生成样本
        with torch.no_grad():
            samples = self.generator(z, labels)
        
        return samples.cpu().numpy(), labels.cpu().numpy()
    
    def _compute_class_centroids(self, train_data, train_labels, unique_labels):
        
        class_centroids = {}
        with torch.no_grad():
            for label in unique_labels:
                class_mask = (train_labels == label)
                class_data = train_data[class_mask]

                if len(class_data) == 0:
                    continue
                
                # 传递与数据匹配的标签
                class_labels = train_labels[class_mask]
                mu, _ = self.vae.encode(class_data, class_labels)
                class_centroids[label] = mu.mean(0)
        return class_centroids

    def generate_from_class_centroids(self, train_data, train_labels, samples_per_class, noise_level=0.1):
        """从类别中心生成样本（此方法会管理模型的评估模式）"""
        self.vae.eval()
        self.generator.eval()
        try:
            # 获取数据中实际存在的唯一类别
            unique_labels_in_data = torch.unique(train_labels).cpu().numpy()
            
            # 计算类别质心
            centroids = self._compute_class_centroids(train_data, train_labels, unique_labels_in_data)
            
            generated_samples = []
            generated_labels = []

            # 确定每个类别要生成的样本数
            if isinstance(samples_per_class, int):
                samples_to_generate = {label: samples_per_class for label in unique_labels_in_data}
            else:
                samples_to_generate = samples_per_class
            
            with torch.no_grad():
                for label, num_samples in samples_to_generate.items():
                    if num_samples <= 0:
                        continue
                    
                    if label not in centroids:
                        print(f"警告: 类别 {label} 的中心点不存在，跳过生成。")
                        continue
                    
                    centroid_mu = centroids[label]
                    
                    # 从质心附近采样潜在向量
                    z = torch.randn(num_samples, self.latent_dim, device=self.device) * noise_level + centroid_mu
                    gen_labels = torch.full((num_samples,), label, dtype=torch.int64, device=self.device)
                    
                    # 使用生成器生成数据
                    samples = self.generator(z, gen_labels)
                    
                    generated_samples.append(samples)
                    generated_labels.append(gen_labels)

            # 如果没有生成任何样本，则返回空张量
            if not generated_samples:
                return torch.FloatTensor(), torch.LongTensor()

            # 合并所有生成的数据
            final_data = torch.cat(generated_samples, dim=0)
            final_labels = torch.cat(generated_labels, dim=0)
            
            return final_data, final_labels
        
        finally:
            # 确保模型在结束后恢复训练模式
            self.vae.train()
            self.generator.train()
    
    def save(self, path):
        """保存模型"""
        torch.save({
            'vae_state_dict': self.vae.state_dict(),
            'gen_state_dict': self.generator.state_dict(),
            'disc_state_dict': self.discriminator.state_dict(),
            'num_classes': self.num_classes,
            'latent_dim': self.latent_dim,
            'input_dim': self.input_dim
        }, path)
        
    def load(self, path):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.vae.load_state_dict(checkpoint['vae_state_dict'])
        self.generator.load_state_dict(checkpoint['gen_state_dict'])
        self.discriminator.load_state_dict(checkpoint['disc_state_dict'])
        
        if 'num_classes' in checkpoint:
            self.num_classes = checkpoint['num_classes']
        if 'latent_dim' in checkpoint:
            self.latent_dim = checkpoint['latent_dim']
        if 'input_dim' in checkpoint:
            self.input_dim = checkpoint['input_dim'] 