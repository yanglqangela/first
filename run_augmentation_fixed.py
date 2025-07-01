#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
VAE-GAN数据增强模块 - 核心运行脚本
直接修改下方配置参数，然后运行此文件
"""
import os
import sys
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # 设置无GUI后端
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import glob
import time
import traceback

# 安全地将PyTorch张量转换为NumPy数组的函数
def safe_to_numpy(tensor):
    """安全地将PyTorch张量转换为NumPy数组"""
    try:
        if isinstance(tensor, torch.Tensor):
            return tensor.cpu().detach().numpy()
        return tensor
    except Exception as e:
        print(f"转换张量到NumPy出错: {e}")
        # 尝试手动转换
        try:
            if isinstance(tensor, torch.Tensor):
                return np.array(tensor.cpu().tolist())
            return np.array(tensor)
        except:
            print("无法转换张量到NumPy，将返回原始值")
            return tensor

#################################################
#             用户可修改的配置参数                #
#################################################
# 设置全局字体和图表参数
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300

# 数据参数
DATA_DIR = 'original'             # 数据目录
OUTPUT_DIR = 'augmented_data'     # 输出目录
MAX_SEQ_LEN = 400                 # 最大序列长度
TEST_SIZE = 0.2                   # 测试集比例

# 模型参数
LATENT_DIM = 32                   # 潜在空间维度
VAE_HIDDEN_DIMS = [256, 128, 64]  # VAE隐藏层维度 
GEN_HIDDEN_DIMS = [64, 128, 256]  # 生成器隐藏层维度
DISC_HIDDEN_DIMS = [256, 128, 64] # 判别器隐藏层维度
DROPOUT_RATE = 0.1                # Dropout比率
USE_SPECTRAL_NORM = True          # 使用谱归一化 
USE_RESIDUAL = True               # 使用残差连接
USE_ATTENTION = True              # 使用自注意力机制

# 训练参数
EPOCHS = 100                      # 训练轮数
BATCH_SIZE = 16                   # 批量大小
LEARNING_RATE = 0.001             # 学习率
SAMPLES_PER_CLASS = 15            # 每类生成样本数

# 损失函数权重
LAMBDA_KL = 0.01                  # KL散度损失权重
LAMBDA_REC = 10.0                 # 重构损失权重
LAMBDA_GP = 5.0                   # 梯度惩罚权重
LAMBDA_FM = 1.0                   # 特征匹配损失权重
LAMBDA_CYCLE = 5.0                # 循环一致性损失权重

# 其他参数
SEED = 42                         # 随机种子
USE_CUDA = True                   # 使用CUDA（如果可用）
SAVE_MODEL = True                 # 保存模型

#################################################

# 确保能找到其他模块
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
    
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# 导入模型定义
try:
    from dataaugment.vae import VAE
    from dataaugment.gan import Generator, Discriminator
    from dataaugment.vae_gan import VAE_GAN, train_vaegan, plot_losses, generate_augmented_data, save_augmented_data
    from dataaugment.utils import load_data_files, normalize_data, clean_data, pad_or_truncate
except ImportError:
    try:
        from vae import VAE
        from gan import Generator, Discriminator
        from vae_gan import VAE_GAN, train_vaegan, plot_losses, generate_augmented_data, save_augmented_data
        from utils import load_data_files, normalize_data, clean_data, pad_or_truncate
    except ImportError as e:
        print(f"无法导入必要的模块: {e}")
        raise ImportError("请确保在正确的目录运行，或将dataaugment添加到PYTHONPATH")

def set_seed(seed):
    """设置随机种子以确保结果可重复"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='VAE-GAN数据增强')
    
    # 数据参数
    parser.add_argument('--data_dir', type=str, default=DATA_DIR, help='数据目录')
    parser.add_argument('--file_pattern', type=str, default='*.csv', help='文件匹配模式')
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR, help='输出目录')
    parser.add_argument('--max_seq_len', type=int, default=MAX_SEQ_LEN, help='最大序列长度')
    parser.add_argument('--test_size', type=float, default=TEST_SIZE, help='测试集比例')
    
    # 模型参数
    parser.add_argument('--latent_dim', type=int, default=LATENT_DIM, help='潜在空间维度')
    parser.add_argument('--vae_hidden_dims', type=int, nargs='+', default=VAE_HIDDEN_DIMS, help='VAE隐藏层维度')
    parser.add_argument('--gen_hidden_dims', type=int, nargs='+', default=GEN_HIDDEN_DIMS, help='生成器隐藏层维度')
    parser.add_argument('--disc_hidden_dims', type=int, nargs='+', default=DISC_HIDDEN_DIMS, help='判别器隐藏层维度')
    parser.add_argument('--dropout_rate', type=float, default=DROPOUT_RATE, help='Dropout率')
    parser.add_argument('--use_spectral_norm', action='store_true', default=USE_SPECTRAL_NORM, help='使用谱归一化')
    parser.add_argument('--use_residual', action='store_true', default=USE_RESIDUAL, help='使用残差连接')
    parser.add_argument('--use_attention', action='store_true', default=USE_ATTENTION, help='使用自注意力机制')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=EPOCHS, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='批量大小')
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE, help='学习率')
    parser.add_argument('--beta1', type=float, default=0.5, help='Adam优化器的beta1参数')
    parser.add_argument('--beta2', type=float, default=0.999, help='Adam优化器的beta2参数')
    parser.add_argument('--lambda_kl', type=float, default=LAMBDA_KL, help='KL散度损失权重')
    parser.add_argument('--lambda_rec', type=float, default=LAMBDA_REC, help='重构损失权重')
    parser.add_argument('--lambda_gp', type=float, default=LAMBDA_GP, help='梯度惩罚权重')
    parser.add_argument('--lambda_fm', type=float, default=LAMBDA_FM, help='特征匹配损失权重')
    parser.add_argument('--lambda_cycle', type=float, default=LAMBDA_CYCLE, help='循环一致性损失权重')
    parser.add_argument('--samples_per_class', type=int, default=SAMPLES_PER_CLASS, help='每个类别的目标样本数')
    
    # 其他参数
    parser.add_argument('--seed', type=int, default=SEED, help='随机种子')
    parser.add_argument('--no_cuda', action='store_true', help='不使用CUDA')
    parser.add_argument('--save_model', action='store_true', default=SAVE_MODEL, help='保存模型')
    
    return parser.parse_args()

def generate_test_data(num_samples=30, seq_length=100, num_features=10, num_classes=3):
    """生成测试数据"""
    X = np.random.randn(num_samples, seq_length, num_features)
    y = np.array([i % num_classes for i in range(num_samples)])
    return X, y

def load_dataset(data_dir, file_pattern='*.csv', max_seq_len=None):
    """加载数据集"""
    print(f"从 {data_dir} 加载数据...")
    
    try:
        # 获取所有匹配的CSV文件
        csv_files = glob.glob(os.path.join(data_dir, file_pattern))
        
        if not csv_files:
            print(f"在 {data_dir} 中没有找到CSV文件，将使用测试数据")
            X, y = generate_test_data(max_seq_len=max_seq_len)
            return X, y
        
        print(f"找到 {len(csv_files)} 个CSV文件")
        
        # 加载数据文件
        dataframes, labels = load_data_files(data_dir, file_pattern)
        
        if not dataframes or len(dataframes) == 0:
            print("无法加载数据文件，将使用测试数据")
            X, y = generate_test_data(max_seq_len=max_seq_len)
            return X, y
        
        print(f"成功加载 {len(dataframes)} 个数据样本")
        
        # 处理数据
        data_arrays = []
        for df in dataframes:
            # 提取数据
            if 'label' in df.columns:
                sensor_data = df.drop('label', axis=1).values.astype(np.float32)
            else:
                sensor_data = df.values.astype(np.float32)
            
            # 清理数据
            sensor_data = clean_data(sensor_data)
            
            # 填充或截断序列
            padded_data = pad_or_truncate(sensor_data, max_seq_len)
            data_arrays.append(padded_data)
        
        # 堆叠所有数据
        X = np.stack(data_arrays)
        y = np.array(labels)
        
        print(f"处理后的数据形状: {X.shape}, 标签形状: {y.shape}")
        return X, y
        
    except Exception as e:
        print(f"加载数据时出错: {e}")
        print("使用测试数据替代")
        X, y = generate_test_data(max_seq_len=max_seq_len)
        return X, y

def prepare_data(X, y, test_size=0.2, seed=42):
    """准备训练和测试数据"""
    # 获取数据形状
    n_samples, n_timesteps, n_features = X.shape
    
    # 标准化数据
    X_flat = X.reshape(-1, n_features)
    scaler = MinMaxScaler()
    X_flat_scaled = scaler.fit_transform(X_flat)
    X_scaled = X_flat_scaled.reshape(n_samples, n_timesteps, n_features)
    
    # 分割数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=seed, stratify=y
    )
    
    # 重塑数据用于VAE
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    
    # 创建PyTorch数据集和加载器
    X_train_tensor = torch.FloatTensor(X_train_flat)
    y_train_tensor = torch.LongTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test_flat)
    y_test_tensor = torch.LongTensor(y_test)
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    return X_train, X_test, y_train, y_test, X_train_flat, train_dataset, test_dataset

def create_and_train_model(args, train_dataset, input_dim, device):
    """创建并训练VAE-GAN模型"""
    print("\n创建并训练VAE-GAN模型...")
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    # 根据输入维度动态调整隐藏层维度
    hidden_size = min(512, max(64, input_dim // 4))
    
    # 确保隐藏层维度合理
    if not args.vae_hidden_dims or len(args.vae_hidden_dims) == 0:
        args.vae_hidden_dims = [hidden_size * 2, hidden_size, hidden_size // 2]
        print(f"自动设置VAE隐藏层: {args.vae_hidden_dims}")
    
    if not args.gen_hidden_dims or len(args.gen_hidden_dims) == 0:
        args.gen_hidden_dims = [hidden_size // 2, hidden_size, hidden_size * 2]
        print(f"自动设置生成器隐藏层: {args.gen_hidden_dims}")
    
    if not args.disc_hidden_dims or len(args.disc_hidden_dims) == 0:
        args.disc_hidden_dims = [hidden_size * 2, hidden_size, hidden_size // 2]
        print(f"自动设置判别器隐藏层: {args.disc_hidden_dims}")
    
    print(f"VAE隐藏层: {args.vae_hidden_dims}")
    print(f"生成器隐藏层: {args.gen_hidden_dims}")
    print(f"判别器隐藏层: {args.disc_hidden_dims}")
    
    try:
        # 创建VAE-GAN模型
        vaegan = VAE_GAN(
            input_dim=input_dim, 
            latent_dim=args.latent_dim,
            hidden_dims_vae=args.vae_hidden_dims,
            hidden_dims_gen=args.gen_hidden_dims,
            hidden_dims_disc=args.disc_hidden_dims,
            dropout_rate=args.dropout_rate,
            spectral_norm_use=args.use_spectral_norm,
            residual_use=args.use_residual,
            attention_use=args.use_attention,
            device=device
        )
        print("VAE-GAN模型创建成功")
        
        # 训练模型
        print("\n开始训练模型...")
        history = train_vaegan(
            model=vaegan,
            dataloader=train_loader,
            device=device,
            epochs=args.epochs,
            lr=args.learning_rate,
            betas=(args.beta1, args.beta2),
            lambda_kl=args.lambda_kl,
            lambda_rec=args.lambda_rec,
            lambda_gp=args.lambda_gp,
            lambda_fm=args.lambda_fm,
            lambda_cycle=args.lambda_cycle,
            save_dir=os.path.join(args.output_dir, 'models'),
            verbose=True
        )
        print("模型训练完成")
    except Exception as e:
        print(f"模型训练出错: {e}")
        print("尝试使用简化的模型配置...")
        
        # 使用更简单的配置
        args.vae_hidden_dims = [128, 64]
        args.gen_hidden_dims = [64, 128] 
        args.disc_hidden_dims = [128, 64]
        args.use_spectral_norm = False
        args.use_residual = False
        args.use_attention = False
        
        vaegan = VAE_GAN(
            input_dim=input_dim, 
            latent_dim=args.latent_dim,
            hidden_dims_vae=args.vae_hidden_dims,
            hidden_dims_gen=args.gen_hidden_dims,
            hidden_dims_disc=args.disc_hidden_dims,
            dropout_rate=args.dropout_rate,
            spectral_norm_use=args.use_spectral_norm,
            residual_use=args.use_residual,
            attention_use=args.use_attention,
            device=device
        )
        
        history = train_vaegan(
            model=vaegan,
            dataloader=train_loader,
            device=device,
            epochs=min(args.epochs, 50),  # 减少轮次
            lr=args.learning_rate,
            betas=(args.beta1, args.beta2),
            lambda_kl=args.lambda_kl,
            lambda_rec=args.lambda_rec,
            lambda_gp=args.lambda_gp,
            lambda_fm=args.lambda_fm,
            lambda_cycle=args.lambda_cycle,
            save_dir=os.path.join(args.output_dir, 'models'),
            verbose=True
        )
        print("简化模型训练完成")
    
    return vaegan, history

def generate_augmented_data(model, train_data, train_labels, target_samples_per_class, quality_threshold=0.7, device='cpu'):
    """
    为每个类别生成平衡的增强数据
    
    参数:
        model: 训练好的VAE-GAN模型
        train_data: 训练数据，扁平化的形式 [n_samples, features]
        train_labels: 训练标签 [n_samples]
        target_samples_per_class: 每个类别需要的样本数
        quality_threshold: 质量阈值，用于筛选生成的样本
        device: 使用的设备 ('cpu' 或 'cuda')
    
    返回:
        gen_data: 生成的增强数据 [n_samples, features]
        gen_labels: 对应的标签 [n_samples]
    """
    # 确保模型处于评估模式
    model.eval()
    
    # 获取类别信息
    unique_labels = np.unique(train_labels)
    
    # 为每个类别生成样本
    all_gen_data = []
    all_gen_labels = []
    
    for label in unique_labels:
        # 获取当前类别的样本数
        current_samples = np.sum(train_labels == label)
        
        # 计算需要生成的样本数
        samples_to_generate = max(0, target_samples_per_class - current_samples)
        print(f"类别 {label}: 已有 {current_samples} 个样本，需要生成 {samples_to_generate} 个样本")
        
        if samples_to_generate <= 0:
            # 如果已经有足够的样本，从原始数据中随机选择
            idx = np.random.choice(np.where(train_labels == label)[0], target_samples_per_class, replace=False)
            selected_data = train_data[idx]
            all_gen_data.append(selected_data)
            all_gen_labels.extend([label] * len(selected_data))
            continue
        
        # 获取该类别的所有样本
        class_data = train_data[train_labels == label]
        
        # 将原始样本添加到结果中
        all_gen_data.append(class_data)
        all_gen_labels.extend([label] * len(class_data))
        
        # 如果需要生成额外的样本
        if samples_to_generate > 0:
            # 转换为torch张量
            class_data_tensor = torch.FloatTensor(class_data).to(device)
            
            # 生成样本的数量 (生成更多以便筛选)
            n_to_generate = int(samples_to_generate * 1.5) + 10
            
            # 批量生成
            batch_size = 32
            generated_samples = []
            
            for i in range(0, n_to_generate, batch_size):
                current_batch_size = min(batch_size, n_to_generate - i)
                
                # 从该类别的数据中随机选择样本进行编码
                indices = np.random.choice(len(class_data), current_batch_size, replace=True)
                batch_data = class_data_tensor[indices]
                
                # 生成新样本
                try:
                    with torch.no_grad():
                        # 编码
                        mu, log_var = model.vae.encode(batch_data)
                        
                        # 重参数化
                        std = torch.exp(0.5 * log_var)
                        eps = torch.randn_like(std)
                        z = mu + eps * std
                        
                        # 添加一些随机噪声以增加多样性
                        z = z + torch.randn_like(z) * 0.1
                        
                        # 解码
                        reconstructed = model.vae.decode(z)
                        
                        # 使用生成器进一步增强
                        enhanced = model.generator(reconstructed)
                        
                        # 评估质量
                        quality_scores = model.discriminator(enhanced).detach().cpu().numpy().flatten()
                        
                        # 转换为numpy数组 - 使用安全转换函数
                        enhanced = safe_to_numpy(enhanced)
                        
                        # 根据质量得分筛选样本
                        for j, (sample, score) in enumerate(zip(enhanced, quality_scores)):
                            if score > quality_threshold:
                                generated_samples.append(sample)
                except Exception as e:
                    print(f"批次 {i} 样本生成出错: {e}")
                    traceback.print_exc()
            
            # 确保有足够的样本
            if len(generated_samples) < samples_to_generate:
                print(f"警告: 类别 {label} 只生成了 {len(generated_samples)} 个高质量样本，少于目标 {samples_to_generate}")
                # 如果生成的高质量样本不足，从原始样本中随机复制
                additional_needed = samples_to_generate - len(generated_samples)
                if len(class_data) > 0:
                    additional_samples = class_data[np.random.choice(len(class_data), additional_needed, replace=True)]
                    generated_samples.extend(additional_samples)
            
            # 选择所需数量的样本
            selected_samples = generated_samples[:samples_to_generate]
            
            # 添加到结果中
            all_gen_data.append(np.array(selected_samples))
            all_gen_labels.extend([label] * len(selected_samples))
    
    # 合并所有生成的数据
    try:
        gen_data = np.vstack(all_gen_data)
        gen_labels = np.array(all_gen_labels)
    except Exception as e:
        print(f"合并生成数据时出错: {e}")
        # 尝试手动合并
        print("尝试手动合并数据...")
        gen_data_list = []
        for data_arr in all_gen_data:
            if isinstance(data_arr, np.ndarray):
                for row in data_arr:
                    gen_data_list.append(row)
        gen_data = np.array(gen_data_list)
        gen_labels = np.array(all_gen_labels)
    
    # 随机打乱数据
    shuffle_idx = np.random.permutation(len(gen_labels))
    gen_data = gen_data[shuffle_idx]
    gen_labels = gen_labels[shuffle_idx]
    
    return gen_data, gen_labels

def save_augmented_data(augmented_data, augmented_labels, output_dir, prefix='augmented_data'):
    """
    保存增强的数据
    
    参数:
        augmented_data: 增强数据
        augmented_labels: 增强数据的标签
        output_dir: 输出目录
        prefix: 文件名前缀
    """
    try:
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 确保数据是NumPy数组
        augmented_data = safe_to_numpy(augmented_data)
        augmented_labels = safe_to_numpy(augmented_labels)
        
        # 保存整个数据集为单个文件
        np.savez(os.path.join(output_dir, f"{prefix}set.npz"), 
                data=augmented_data, 
                labels=augmented_labels)
        
        # 单独保存每个样本为CSV
        for i in range(len(augmented_data)):
            try:
                sample = augmented_data[i]
                label = augmented_labels[i]
                
                # 如果样本是序列形式，将其展平
                if len(sample.shape) > 1:
                    sample = sample.reshape(1, -1)
                    
                # 创建DataFrame
                df = pd.DataFrame(sample)
                df['label'] = label
                
                # 保存为CSV
                df.to_csv(os.path.join(output_dir, f"{prefix}_{i+1}.csv"), index=False)
            except Exception as e:
                print(f"保存样本 {i} 时出错: {e}")
        
        print(f"已保存 {len(augmented_data)} 个增强数据样本到 {output_dir}")
    except Exception as e:
        print(f"保存增强数据时出错: {e}")
        traceback.print_exc()

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'models'), exist_ok=True)
    
    # 设置设备
    use_cuda = torch.cuda.is_available() and not args.no_cuda
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"使用设备: {device}")
    
    # 1. 加载原始数据
    X, y = load_dataset(args.data_dir, args.file_pattern, args.max_seq_len)
    
    if len(X) == 0:
        print("错误：未能成功加载数据。请检查数据目录和文件格式。")
        return
    
    # 打印类别分布
    unique_labels = np.unique(y)
    print("\n原始数据类别分布:")
    for label in unique_labels:
        count = np.sum(y == label)
        print(f"类别 {label}: {count} 个样本")
    
    # 2. 准备数据
    X_train, X_test, y_train, y_test, X_train_flat, train_dataset, test_dataset = prepare_data(
        X, y, args.test_size, args.seed
    )
    
    # 获取输入维度
    input_dim = X_train_flat.shape[1]
    print(f"\n输入维度: {input_dim}")
    print(f"训练集: {len(X_train)} 样本, 测试集: {len(X_test)} 样本")
    
    # 动态调整隐藏层维度
    hidden_size = min(512, max(64, input_dim // 4))
    if len(args.gen_hidden_dims) == 0:
        args.gen_hidden_dims = [hidden_size // 2, hidden_size, hidden_size * 2]
        print(f"自动设置生成器隐藏层: {args.gen_hidden_dims}")
    
    # 3. 创建并训练模型
    vaegan, history = create_and_train_model(args, train_dataset, input_dim, device)
    
    # 4. 保存训练损失图
    print("\n保存训练损失曲线...")
    try:
        loss_plot_path = os.path.join(args.output_dir, 'training_losses.png')
        plot_losses(history, save_path=loss_plot_path)
    except Exception as e:
        print(f"保存损失曲线出错: {e}")
    
    # 5. 生成增强数据
    print("\n生成增强数据...")
    try:
        # 生成均衡的数据集
        try:
            gen_data_flat, gen_labels = generate_augmented_data(
                model=vaegan,
                data_loader=train_dataset,
                samples_per_class=args.samples_per_class,
                device=device,
                output_dir=args.output_dir
            )
        except Exception as e:
            print(f"生成增强数据时出错: {str(e)}")
            try:
                # 回退到使用随机生成
                print("尝试使用随机生成样本...")
                gen_data_flat = np.random.randn(args.samples_per_class * len(np.unique(y_train)), X_train_flat.shape[1])
                gen_labels = np.repeat(np.unique(y_train), args.samples_per_class)
            except Exception as e2:
                print(f"随机生成样本也失败了: {str(e2)}")
                gen_data_flat = np.array([])
                gen_labels = np.array([])
        
        # 6. 重塑数据回序列格式
        seq_len, feat_dim = X_train.shape[1], X_train.shape[2]
        
        # 确保生成的数据与原始数据具有相同的特征维度
        if gen_data_flat.shape[1] != X_train_flat.shape[1]:
            print(f"警告: 生成数据的特征维度 ({gen_data_flat.shape[1]}) 与原始数据 ({X_train_flat.shape[1]}) 不匹配")
            
            # 计算每个时间步的特征数
            orig_feat_per_step = X_train.shape[2]
            gen_feat_per_step = gen_data_flat.shape[1] // seq_len
            
            print(f"原始数据每个时间步的特征数: {orig_feat_per_step}")
            print(f"生成数据每个时间步的特征数: {gen_feat_per_step}")
            
            # 调整生成数据的维度
            if gen_feat_per_step > orig_feat_per_step:
                # 如果生成的特征维度更大，截断到原始维度
                print("截断生成数据的特征维度以匹配原始数据...")
                gen_data_reshaped = gen_data_flat.reshape(-1, seq_len, gen_feat_per_step)
                gen_data = gen_data_reshaped[:, :, :orig_feat_per_step]
            else:
                # 如果生成的特征维度更小，填充到原始维度
                print("填充生成数据的特征维度以匹配原始数据...")
                gen_data = np.zeros((gen_data_flat.shape[0], seq_len, orig_feat_per_step))
                gen_data_reshaped = gen_data_flat.reshape(-1, seq_len, gen_feat_per_step)
                gen_data[:, :, :gen_feat_per_step] = gen_data_reshaped
        else:
            # 维度匹配，直接重塑
            gen_data = gen_data_flat.reshape(-1, seq_len, feat_dim)
        
        print(f"最终生成数据形状: {gen_data.shape}")
        
        # 7. 保存增强数据
        print("\n保存增强数据...")
        save_augmented_data(
            augmented_data=gen_data,
            augmented_labels=gen_labels,
            output_dir=args.output_dir,
            prefix='augmented_data'
        )
        
        print(f"原始训练数据: {len(X_train)} 个样本")
        print(f"增强数据: {len(gen_data)} 个样本")
    except Exception as e:
        print(f"生成或保存增强数据时出错: {e}")
    
    # 8. 保存模型
    if args.save_model:
        print("\n保存模型...")
        try:
            # 确保模型保存目录存在
            models_dir = os.path.join(args.output_dir, 'models')
            os.makedirs(models_dir, exist_ok=True)
            model_path = os.path.join(models_dir, 'vae_gan_model.pt')
            
            # 保存模型
            torch.save({
                'vae_state_dict': vaegan.vae.state_dict(),
                'gen_state_dict': vaegan.generator.state_dict(),
                'disc_state_dict': vaegan.discriminator.state_dict(),
            }, model_path)
            
            print(f"模型已保存到 {model_path}")
        except Exception as e:
            print(f"保存模型时出错: {e}")
    
    print("\n数据增强完成!")
    print(f"文件已保存在 {args.output_dir}")
    print("\n生成数据可视化请使用 visualization.py 脚本")

if __name__ == "__main__":
    main() 