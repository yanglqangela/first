#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
简化版数据增强模块 - 使用PyTorch实现的时序数据增强
"""
import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
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
import gc

# 确保能找到其他模块
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
    
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# 导入模型定义
try:
    # 尝试作为包导入
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from dataaugment.vae_gan import VAE_GAN, plot_losses, generate_augmented_data, save_augmented_data as vae_gan_save_augmented_data
    from dataaugment.gan import Discriminator # Import Discriminator to ensure its extract_features is available
except ImportError as e:
    print(f"无法导入必要的模块: {e}")
    try:
        # 尝试作为本地模块导入
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from vae_gan import VAE_GAN, plot_losses, generate_augmented_data, save_augmented_data as vae_gan_save_augmented_data
        from gan import Discriminator # Import Discriminator to ensure its extract_features is available
    except ImportError as e:
        print(f"第二次尝试导入失败: {e}")
        raise ImportError("请确保在正确的目录运行，或将dataaugment添加到PYTHONPATH")

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

def set_seed(seed):
    """设置随机种子，确保结果可复现"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def setup_args():
    parser = argparse.ArgumentParser(description='时间序列数据增强')
    parser.add_argument('--data_dir', type=str, default='./original1', help='原始数据目录')
    parser.add_argument('--output_dir', type=str, default='./augmented_data', help='增强数据输出目录')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='模型检查点目录')
    parser.add_argument('--viz_dir', type=str, default='./augmented_data', help='可视化输出目录')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮次')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--latent_dim', type=int, default=32, help='潜在空间维度')
    parser.add_argument('--seq_length', type=int, default=400, help='序列长度')
    parser.add_argument('--target_samples', type=int, default=25, help='每个类别的目标样本数')
    parser.add_argument('--model', type=str, default='vae_gan', choices=['vae_gan'], help='使用的模型类型')
    parser.add_argument('--device', type=str, default='cpu', help='训练设备')
    return parser.parse_args()

def load_data(data_dir, seq_length=None):
    """加载并预处理数据"""
    print(f"从 {data_dir} 加载数据...")
    
    # 获取所有CSV文件
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    
    # 初始化数据和标签列表
    all_data = []
    all_labels = []
    
    # 加载所有CSV文件
    for file in tqdm(csv_files, desc="处理文件"):
        file_path = os.path.join(data_dir, file)
        try:
            # 读取CSV文件
            df = pd.read_csv(file_path)
            
            # 提取标签（假设最后一列是标签）
            if 'label' in df.columns:
                label = df['label'].iloc[0]
                df = df.drop('label', axis=1)
            else:
                # 尝试从最后一列获取标签
                label = df.iloc[0, -1]
                df = df.iloc[:, :-1]
            
            # 转换为numpy数组
            data = df.values
            
            # 添加到列表
            all_data.append(data)
            all_labels.append(label)
        except Exception as e:
            print(f"处理文件 {file} 时出错: {e}")
    
    # 确定最大序列长度
    if seq_length is None:
        seq_length = max([len(d) for d in all_data])
        print(f"所有序列将被处理到长度: {seq_length}")
    
    # 填充/截断序列到相同长度
    processed_data = []
    for data in tqdm(all_data, desc="填充/截断数据"):
        if len(data) > seq_length:
            # 截断
            processed_data.append(data[:seq_length])
        else:
            # 填充
            padding = np.zeros((seq_length - len(data), data.shape[1]))
            padded_data = np.vstack((data, padding))
            processed_data.append(padded_data)
    
    # 转换为numpy数组
    X = np.array(processed_data)
    y = np.array(all_labels)
    
    # 归一化数据
    print("归一化数据...")
    X_norm = np.zeros_like(X)
    for i in range(len(X)):
        for j in range(X.shape[2]):
            # 对每个特征进行归一化
            feature = X[i, :, j]
            min_val = np.min(feature)
            max_val = np.max(feature)
            if max_val > min_val:
                X_norm[i, :, j] = (feature - min_val) / (max_val - min_val)
            else:
                X_norm[i, :, j] = 0
    
    print(f"\n数据集类别分布:")
    unique_labels, counts = np.unique(y, return_counts=True)
    for label, count in zip(unique_labels, counts):
        print(f"类别 {label}: {count} 个样本")
    
    print(f"加载和预处理完成。数据形状: {X_norm.shape}, 标签形状: {y.shape}")
    
    return X_norm, y, seq_length

def prepare_datasets(X, y, batch_size, test_size=0.2):
    """准备训练和测试数据集"""
    print("准备训练/测试数据集...")
    
    # 分割训练和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)
    
    # 打印训练集类别分布
    print("\n训练集类别分布:")
    unique_labels, counts = np.unique(y_train, return_counts=True)
    for label, count in zip(unique_labels, counts):
        print(f"类别 {label}: {count} 个样本")
    
    # 打印测试集类别分布
    print("\n测试集类别分布:")
    unique_labels, counts = np.unique(y_test, return_counts=True)
    for label, count in zip(unique_labels, counts):
        print(f"类别 {label}: {count} 个样本")
    
    print(f"\n训练集: {len(X_train)} 样本, 测试集: {len(X_test)} 样本\n")
    
    # 创建PyTorch数据集
    # 重塑数据以适应VAE-GAN模型
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    
    # 转换为PyTorch张量
    X_train_tensor = torch.FloatTensor(X_train_flat)
    y_train_tensor = torch.LongTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test_flat)
    y_test_tensor = torch.LongTensor(y_test)
    
    # 创建数据加载器
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return X_train, y_train, train_loader, X_test, y_test, test_loader

# 在VAE-GAN模型的train方法中修改，移除中间断点保存
def train_vae_gan(X_train, y_train, train_loader, input_dim, latent_dim, epochs, device, viz_dir, checkpoint_dir):
    """训练VAE-GAN模型"""
    print("\n创建VAE-GAN模型...")
    print(f"输入维度: {input_dim} (序列长度: {X_train.shape[1]}, 特征数: {X_train.shape[2]})")
    
    # 创建VAE-GAN模型
    model = VAE_GAN(
            input_dim=input_dim,
            latent_dim=latent_dim,
        hidden_dims_vae=[256, 128],  # 减小隐藏层大小以节省内存
        hidden_dims_gen=[128, 256],
        hidden_dims_disc=[256, 128],
        dropout_rate=0.3,
        lr=2e-5,
            device=device
    ).to(device)
    
    # 训练模型
    print(f"\n训练VAE-GAN模型 (Epochs: {epochs})...")
    try:
        # 修改训练参数，移除中间可视化和断点保存
        history = model.train(
            train_loader=train_loader,
            epochs=epochs,
            kl_weight=0.005,  # 降低KL散度权重以减少正则化
            gan_weight=0.1,
            gp_weight=10.0,
            viz_interval=epochs,  # 设置为epochs值，这样只在最后保存可视化
            checkpoint_interval=epochs,  # 设置为epochs值，这样只在最后保存检查点
            viz_dir=viz_dir,
            checkpoint_dir=checkpoint_dir
        )
        
        # 保存最终模型
        model_save_path = os.path.join(checkpoint_dir, f"vae_gan_final.pt")
        torch.save({
            'vae_state_dict': model.vae.state_dict(),
            'gen_state_dict': model.generator.state_dict(),
            'disc_state_dict': model.discriminator.state_dict(),
        }, model_save_path)
        print(f"最终模型已保存到 {model_save_path}")
        
        # 只保存最终的损失图
        if viz_dir:
            os.makedirs(viz_dir, exist_ok=True)
            
            # 绘制所有损失
            plt.figure(figsize=(12, 8))
            for key in history:
                if key.endswith('loss'):
                    plt.plot(history[key], label=key)
            plt.title('训练损失曲线')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(viz_dir, "final_losses.png"))
            plt.close()
        
        return model
    
    except Exception as e:
        print(f"训练VAE-GAN模型时出错: {e}")
        traceback.print_exc()
        return None

def generate_augmented_data_vae_gan(model, X_train, y_train, target_samples_per_class, device):
    """使用VAE-GAN生成增强数据"""
    print("\n使用VAE-GAN生成增强数据...")
    
    # 获取唯一类别和每个类别的样本数
    unique_classes = np.unique(y_train)
    class_counts = {cls: np.sum(y_train == cls) for cls in unique_classes}
    
    # 打印每个类别的原始样本数
    print("原始样本数:")
    for cls, count in class_counts.items():
        print(f"类别 {cls}: {count} 样本")
    
    # 计算每个类别需要生成的样本数
    samples_to_generate = {
        cls: max(0, target_samples_per_class - count) 
        for cls, count in class_counts.items()
    }
    
    print("\n需要生成的样本数:")
    for cls, count in samples_to_generate.items():
        print(f"类别 {cls}: {count} 样本")
    
    # 初始化生成数据和标签列表
    generated_data = []
    generated_labels = []
    
    # 对每个类别生成数据
    for cls in unique_classes:
        if samples_to_generate[cls] <= 0:
            print(f"类别 {cls} 已有足够样本，跳过生成")
            continue
        
        # 获取该类别的训练数据
        class_indices = np.where(y_train == cls)[0]
        X_class = X_train[class_indices]
        
        # 重塑数据以适应VAE-GAN模型
        X_class_flat = X_class.reshape(X_class.shape[0], -1)
        X_class_tensor = torch.FloatTensor(X_class_flat).to(device)
        
        print(f"为类别 {cls} 生成 {samples_to_generate[cls]} 个样本...")
        
        # 生成样本
        gen_samples = []
        batch_size = min(32, samples_to_generate[cls])  # 批量生成以减少内存使用
        
        for i in range(0, samples_to_generate[cls], batch_size):
            n_to_generate = min(batch_size, samples_to_generate[cls] - i)
            
            # 随机选择原始样本作为参考
            random_indices = np.random.choice(len(X_class), n_to_generate)
            reference_samples = X_class_tensor[random_indices]
            
            # 生成样本
            with torch.no_grad():
                # 编码参考样本
                mu, log_var = model.vae.encode(reference_samples)
                
                # 添加随机噪声以增加多样性
                z = model.vae.reparameterize(mu, log_var)
                z = z + torch.randn_like(z) * 0.1  # 添加少量噪声
                
                # 解码生成新样本
                generated = model.vae.decode(z)
                
                # 转换为NumPy数组
                generated_np = safe_to_numpy(generated)
                
                # 重塑回原始形状
                generated_np = generated_np.reshape(-1, X_class.shape[1], X_class.shape[2])
                
                gen_samples.append(generated_np)
        
        # 合并所有生成的样本
        gen_samples = np.vstack(gen_samples) if len(gen_samples) > 0 else np.array([])
        
        # 添加到生成数据列表
        generated_data.append(gen_samples)
        generated_labels.append(np.full(len(gen_samples), cls))
    
    # 合并所有类别的生成数据
    if generated_data:
        generated_data = np.vstack(generated_data)
        generated_labels = np.concatenate(generated_labels)
        
        print(f"VAE-GAN生成的数据形状: {generated_data.shape}, 标签形状: {generated_labels.shape}")
        
        # 检查生成数据的类别分布
        print("\nVAE-GAN生成数据的类别分布:")
        unique_labels, counts = np.unique(generated_labels, return_counts=True)
        for label, count in zip(unique_labels, counts):
            print(f"类别 {label}: {count} 个样本")
    else:
        print("没有生成任何数据")
        generated_data = np.array([])
        generated_labels = np.array([])
    
    return generated_data, generated_labels

def combine_data(original_data, original_labels, gen_data_vae, gen_labels_vae, target_samples_per_class):
    """合并原始数据和生成的数据"""
    print("\n合并原始数据和生成的数据...")
    
    # 获取唯一类别
    unique_classes = np.unique(original_labels)
    
    # 初始化合并后的数据和标签列表
    combined_data = []
    combined_labels = []
    
    # 对每个类别处理数据
    for cls in unique_classes:
        # 获取原始数据中该类别的样本
        orig_indices = np.where(original_labels == cls)[0]
        orig_data_cls = original_data[orig_indices]
        
        # 添加所有原始样本
        combined_data.append(orig_data_cls)
        combined_labels.append(np.full(len(orig_data_cls), cls))
        
        # 计算已有样本数
        current_count = len(orig_data_cls)
        
        # 如果有VAE-GAN生成的数据，添加它们
        if len(gen_data_vae) > 0:
            vae_indices = np.where(gen_labels_vae == cls)[0]
            if len(vae_indices) > 0:
                vae_data_cls = gen_data_vae[vae_indices]
                
                # 计算还需要多少样本
                samples_needed = target_samples_per_class - current_count
                
                if samples_needed > 0:
                    # 如果VAE-GAN生成的样本不够，全部使用
                    if len(vae_data_cls) <= samples_needed:
                        combined_data.append(vae_data_cls)
                        combined_labels.append(np.full(len(vae_data_cls), cls))
                        current_count += len(vae_data_cls)
                    else:
                        # 否则只使用需要的数量
                        combined_data.append(vae_data_cls[:samples_needed])
                        combined_labels.append(np.full(samples_needed, cls))
                        current_count += samples_needed
        
        print(f"类别 {cls}: 合并后共有 {current_count} 个样本")
    
    # 合并所有类别的数据
    combined_data = np.vstack(combined_data) if combined_data else np.array([])
    combined_labels = np.concatenate(combined_labels) if combined_labels else np.array([])
    
    print(f"合并后的数据形状: {combined_data.shape}, 标签形状: {combined_labels.shape}")
    
    # 检查合并后的类别分布
    print("\n合并后的类别分布:")
    unique_labels, counts = np.unique(combined_labels, return_counts=True)
    for label, count in zip(unique_labels, counts):
        print(f"类别 {label}: {count} 个样本 ({count/len(combined_labels)*100:.1f}%)")
    
    return combined_data, combined_labels

def save_augmented_data(augmented_data, augmented_labels, output_dir):
    """保存增强后的数据到CSV文件"""
    print(f"\n保存增强数据到 {output_dir}...")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取唯一类别
    unique_classes = np.unique(augmented_labels)
    
    # 保存每个样本到单独的CSV文件
    for i in range(len(augmented_data)):
        # 获取样本和标签
        sample = augmented_data[i]
        label = augmented_labels[i]
            
        # 创建DataFrame
        df = pd.DataFrame(sample)
        
        # 添加标签列
        df['label'] = label
        
        # 保存到CSV文件
        file_path = os.path.join(output_dir, f"augmented_data_{i}.csv")
        df.to_csv(file_path, index=False)
    
    # 保存整个数据集为NPZ文件
    np.savez(
        os.path.join(output_dir, "augmented_dataset.npz"),
        data=augmented_data,
        labels=augmented_labels
    )
    
    # 绘制类别分布
    plt.figure(figsize=(10, 6))
    unique_labels, counts = np.unique(augmented_labels, return_counts=True)
    plt.bar(unique_labels, counts)
    plt.title('Class Distribution')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.xticks(unique_labels)
    plt.grid(True, axis='y')
    
    # 添加数量和百分比标签
    for i, (label, count) in enumerate(zip(unique_labels, counts)):
        percentage = count / len(augmented_labels) * 100
        plt.text(label, count + 0.1, f"{count}\n({percentage:.1f}%)", 
                ha='center', va='bottom')
    
    plt.savefig(os.path.join(output_dir, "class_distribution.png"))
    plt.close()
    
    print(f"已保存 {len(augmented_data)} 个样本到 {output_dir}")
    print(f"类别分布图已保存到 {os.path.join(output_dir, 'class_distribution.png')}")

def main():
    # 解析命令行参数
    args = setup_args()
    
    # 设置随机种子
    set_seed(42)
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() and args.device != 'cpu' else "cpu")
    print(f"使用设备: {device}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.viz_dir, exist_ok=True)
    
    # 加载数据
    X, y, seq_length = load_data(args.data_dir, args.seq_length)
    
    # 准备数据集
    X_train, y_train, train_loader, X_test, y_test, test_loader = prepare_datasets(X, y, args.batch_size)
    
    # 计算输入维度
    input_dim = X_train.shape[1] * X_train.shape[2]
    
    # 训练VAE-GAN模型
    vae_gan_model = None
    if args.model == 'vae_gan':
        vae_gan_model = train_vae_gan(
            X_train, y_train, train_loader, input_dim, args.latent_dim, 
            args.epochs, device, args.viz_dir, args.checkpoint_dir
        )
    
    # 生成增强数据
    gen_data_vae, gen_labels_vae = np.array([]), np.array([])
    
    if vae_gan_model is not None:
        gen_data_vae, gen_labels_vae = generate_augmented_data_vae_gan(
            vae_gan_model, X_train, y_train, args.target_samples, device
        )
    
    # 合并数据
    augmented_data, augmented_labels = combine_data(
        X_train, y_train, gen_data_vae, gen_labels_vae, args.target_samples
    )
    
    # 保存增强数据
    save_augmented_data(augmented_data, augmented_labels, args.output_dir)
    
    print("\n数据增强完成!")

if __name__ == "__main__":
    try:
        main() 
    except Exception as e:
        print(f"执行过程中出错: {e}")
        traceback.print_exc() 