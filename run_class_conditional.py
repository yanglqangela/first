#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
类别条件VAE-GAN数据增强脚本
保留类别特征更好的数据增强方案
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
# 设置全局字体和图表参数
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300

# 添加当前目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
    
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# 导入自定义模块
try:
    from class_conditional_vae_gan import ClassConditionalVAEGAN
    from utils import load_data_files, normalize_data, clean_data, pad_or_truncate
except ImportError as e:
    print(f"导入模块失败: {e}")
    print("请确保类别条件VAE-GAN模块存在")
    sys.exit(1)

# 配置参数
DATA_DIR = 'original'               # 数据目录
OUTPUT_DIR = 'augmented_data_cc'    # 输出目录
MAX_SEQ_LEN = 400                   # 最大序列长度
TEST_SIZE = 0.2                    # 测试集比例
LATENT_DIM = 16                     # 潜在空间维度
HIDDEN_DIMS_VAE = [256, 128]        # VAE隐藏层维度
HIDDEN_DIMS_GEN = [256, 128]        # 生成器隐藏层维度
HIDDEN_DIMS_DISC = [256, 128]       # 判别器隐藏层维度
DROPOUT_RATE = 0.1                  # Dropout比率
EPOCHS = 500                        # 训练轮数
BATCH_SIZE = 16                   # 批量大小
LEARNING_RATE = 0.0005              # 学习率
TOTAL_PER_CLASS = 25                # 增强后每类的目标总数
SEED = 1234567890                   # 随机种子

# 损失函数权重
LAMBDA_KL = 1.0                    # KL散度权重
LAMBDA_REC = 10.0                   # 重构损失权重
LAMBDA_GP = 10.0                    # 梯度惩罚权重
LAMBDA_CLS = 1.0                    # 分类损失权重

def set_seed(seed):
    """设置随机种子以确保结果可重复"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_dataset(data_dir, file_pattern='*.csv', max_seq_len=None):
    """加载数据集"""
    print(f"从 {data_dir} 加载数据...")
    
    try:
        # 获取所有匹配的CSV文件
        csv_files = glob.glob(os.path.join(data_dir, file_pattern))
        
        if not csv_files:
            print(f"在 {data_dir} 中没有找到CSV文件")
            return None, None
        
        print(f"找到 {len(csv_files)} 个CSV文件")
        
        # 加载数据文件
        dataframes, labels = load_data_files(data_dir, file_pattern)
        
        if not dataframes or len(dataframes) == 0:
            print("无法加载数据文件")
            return None, None
        
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
        traceback.print_exc()
        return None, None

def prepare_data(X, y, test_size=0.2, seed=42):
    """准备训练和测试数据"""
    try:
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
    except Exception as e:
        print(f"准备数据时出错: {e}")
        traceback.print_exc()
        raise e

def compute_class_weights(labels):
    """计算类别权重以处理不平衡数据"""
    try:
        # 确保labels是一个NumPy数组
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()
            
        unique_labels = np.unique(labels)
        num_classes = len(unique_labels)
        
        # 统计每个类别的样本数
        counts = np.bincount(labels, minlength=num_classes)
        
        # 计算权重 - 反比于类别频率
        weights = 1.0 / counts
        
        # 归一化权重
        weights = weights / np.sum(weights) * num_classes
        
        return weights
    except Exception as e:
        print(f"计算类别权重时出错: {e}")
        traceback.print_exc()
        # 返回均匀权重
        return np.ones(len(np.unique(labels)))

def plot_losses(history, save_path):
    """绘制训练损失曲线"""
    try:
        plt.figure(figsize=(12, 8))
        
        # VAE和GAN损失
        plt.subplot(2, 2, 1)
        plt.plot(history['vae_loss'], label='VAE Loss')
        plt.plot(history['gen_loss'], label='Generator Loss')
        plt.plot(history['disc_loss'], label='Discriminator Loss')
        plt.title('训练损失')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # 重构和KL损失
        plt.subplot(2, 2, 2)
        plt.plot(history['rec_loss'], label='Reconstruction Loss')
        plt.plot(history['kl_loss'], label='KL Divergence')
        plt.title('VAE组件损失')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # 类别损失
        plt.subplot(2, 2, 3)
        plt.plot(history['cls_loss'], label='Classification Loss')
        plt.title('类别损失')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    except Exception as e:
        print(f"绘制损失曲线时出错: {e}")
        traceback.print_exc()

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

def save_augmented_data(augmented_data, augmented_labels, output_dir, seq_shape, prefix='augmented'):
    """保存增强数据（仅保存生成的数据）"""
    try:
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 确保数据是NumPy数组
        augmented_data = safe_to_numpy(augmented_data)
        augmented_labels = safe_to_numpy(augmented_labels)
        
        # 重塑数据为序列形式
        seq_len, feat_dim = seq_shape
        
        try:
            augmented_data_seq = augmented_data.reshape(-1, seq_len, feat_dim)
        except Exception as e:
            print(f"重塑数据出错: {e}. 尝试不同方法...")
            # 如果数据维度与期望不匹配，尝试不同的重塑方法
            total_features = seq_len * feat_dim
            if augmented_data.size % total_features == 0:
                n_samples = augmented_data.size // total_features
                augmented_data_seq = augmented_data.reshape(n_samples, seq_len, feat_dim)
            else:
                print(f"警告: 无法正确重塑数据。原始维度: {augmented_data.shape}, 目标维度: (-1, {seq_len}, {feat_dim})")
                # 使用截断或填充调整维度
                augmented_data_seq = np.zeros((len(augmented_labels), seq_len, feat_dim))
                for i, data in enumerate(augmented_data):
                    if len(data) >= total_features:
                        reshaped = data[:total_features].reshape(seq_len, feat_dim)
                    else:
                        reshaped = np.zeros((seq_len, feat_dim))
                        reshaped.flat[:len(data)] = data
                    augmented_data_seq[i] = reshaped
        
        # 保存整个数据集为NPZ文件
        np.savez(
            os.path.join(output_dir, 'augmented_dataset.npz'),
            data=augmented_data_seq,
            labels=augmented_labels
        )
        
        # 单独保存CSV文件
        for i, (data, label) in enumerate(zip(augmented_data_seq, augmented_labels)):
            # 创建DataFrame
            df = pd.DataFrame(data)
            
            # 添加标签列
            df['label'] = label
            
            # 保存为CSV
            output_path = os.path.join(output_dir, f'{prefix}_class{label}_sample{i}.csv')
            df.to_csv(output_path, index=False)
        
        print(f"已保存 {len(augmented_labels)} 个增强数据样本到 {output_dir}")
    except Exception as e:
        print(f"保存增强数据时出错: {e}")
        traceback.print_exc()

def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='类别条件VAE-GAN数据增强')
    
    parser.add_argument('--data_dir', type=str, default=DATA_DIR, help='数据目录')
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR, help='输出目录')
    parser.add_argument('--max_seq_len', type=int, default=MAX_SEQ_LEN, help='最大序列长度')
    parser.add_argument('--latent_dim', type=int, default=LATENT_DIM, help='潜在空间维度')
    parser.add_argument('--epochs', type=int, default=EPOCHS, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='批量大小')
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE, help='学习率')
    parser.add_argument('--total_per_class', type=int, default=TOTAL_PER_CLASS, help='增强后每类的目标总样本数')
    parser.add_argument('--seed', type=int, default=SEED, help='随机种子')
    
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    model_dir = os.path.join(args.output_dir, 'models')
    os.makedirs(model_dir, exist_ok=True)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 1. 加载原始数据
    X, y = load_dataset(args.data_dir, '*.csv', args.max_seq_len)
    
    if X is None or len(X) == 0:
        print("错误：无法加载数据。请检查数据目录和文件格式。")
        return
    
    # 打印类别分布
    unique_labels = np.unique(y)
    num_classes = len(unique_labels)
    print("\n原始数据类别分布:")
    for label in unique_labels:
        count = np.sum(y == label)
        print(f"类别 {label}: {count} 个样本")
    
    # 2. 准备数据
    X_train, X_test, y_train, y_test, X_train_flat, train_dataset, test_dataset = prepare_data(
        X, y, test_size=0.2, seed=args.seed
    )
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    
    # 获取输入维度
    input_dim = X_train_flat.shape[1]
    print(f"\n输入维度: {input_dim}")
    print(f"训练集: {len(X_train)} 样本, 测试集: {len(X_test)} 样本")
    
    # 计算类别权重
    class_weights = compute_class_weights(y_train)
    print(f"类别权重: {class_weights}")
    
    # 3. 创建并训练模型
    print("\n创建并训练类别条件VAE-GAN模型...")
    try:
        model = ClassConditionalVAEGAN(
            input_dim=input_dim,
            latent_dim=args.latent_dim,
            hidden_dims_vae=HIDDEN_DIMS_VAE,
            hidden_dims_gen=HIDDEN_DIMS_GEN,
            hidden_dims_disc=HIDDEN_DIMS_DISC,
            num_classes=num_classes,
            dropout_rate=DROPOUT_RATE,
            device=device
        )
        
        # 训练模型
        print("\n开始训练模型...")
        history = model.train(
            dataloader=train_loader,
            epochs=args.epochs,
            lambda_kl=LAMBDA_KL,
            lambda_rec=LAMBDA_REC,
            lambda_gp=LAMBDA_GP,
            lambda_cls=LAMBDA_CLS,
            class_weights=class_weights,
            save_dir=model_dir,
            kl_anneal_epochs=50  # 在前50个epoch进行KL退火
        )
        
        # 保存损失曲线
        loss_plot_path = os.path.join(args.output_dir, 'training_losses.png')
        plot_losses(history, loss_plot_path)
        
        # 4. 生成增强数据
        print("\n生成增强数据以平衡类别...")
        
        # 将数据移动到设备
        X_train_tensor = torch.FloatTensor(X_train_flat).to(device)
        y_train_tensor = torch.LongTensor(y_train).to(device)
        
        # 计算每个类别需要生成的样本数量
        unique_labels_train, counts_train = np.unique(y_train, return_counts=True)
        samples_to_generate = {
            label: max(0, args.total_per_class - count)
            for label, count in zip(unique_labels_train, counts_train)
        }
        print(f"\n目标每类总数: {args.total_per_class}")
        print("将生成以下数量的样本:")
        for label, num in samples_to_generate.items():
            original_count = dict(zip(unique_labels_train, counts_train)).get(label, 0)
            print(f"类别 {label}: 原始 {original_count}个, 生成 {num}个")

        # 从类别中心生成样本
        try:
            gen_data, gen_labels = model.generate_from_class_centroids(
                train_data=X_train_tensor,
                train_labels=y_train_tensor,
                samples_per_class=samples_to_generate,
                noise_level=0.2  # 适中的噪声水平
            )
            
            # 确保生成的数据是NumPy数组
            gen_data = safe_to_numpy(gen_data)
            gen_labels = safe_to_numpy(gen_labels)
            
            # 5. 保存生成的数据（仅保存生成的数据）
            print("\n保存增强数据...")
            seq_shape = (X_train.shape[1], X_train.shape[2])  # (seq_len, feat_dim)
            save_augmented_data(
                augmented_data=gen_data,
                augmented_labels=gen_labels,
                output_dir=args.output_dir,
                seq_shape=seq_shape,
                prefix='augmented'
            )
            
            # 保存最终模型
            model.save(os.path.join(model_dir, 'final_model.pt'))
            
            # 打印生成结果
            print("\n数据增强完成!")
            print(f"原始训练数据: {len(X_train)} 个样本")
            print(f"增强数据: {len(gen_data)} 个样本")
            
            # 打印增强后的类别分布
            print("\n增强数据类别分布:")
            for label in unique_labels:
                count = np.sum(gen_labels == label)
                print(f"类别 {label}: {count} 个样本")
                
        except Exception as e:
            print(f"生成或保存增强数据时出错: {e}")
            traceback.print_exc()
            
    except Exception as e:
        print(f"训练或生成数据时出错: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main() 