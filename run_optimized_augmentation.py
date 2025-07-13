#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
运行优化的VAE-GAN数据增强
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from optimized_vae_gan import OptimizedVAEGAN
from enhanced_vae_gan import enhanced_prepare_data
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

def run_optimized_augmentation(original_dir='original', output_dir='augmented_data_optimized', 
                             samples_per_class=25, seq_len=400, epochs=300):
    """
    运行优化的VAE-GAN数据增强
    """
    
    print("=" * 60)
    print("优化VAE-GAN时序数据增强")
    print("=" * 60)
    
    # 检查原始数据目录
    if not os.path.exists(original_dir):
        print(f"错误: 原始数据目录 '{original_dir}' 不存在")
        return
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)
    
    try:
        # 1. 加载和预处理数据
        print("\n1. 加载和预处理数据...")
        dataloader, scaler, data_shape, original_data = enhanced_prepare_data(
            original_dir, seq_len=seq_len, batch_size=8  # 减小batch size
        )
        print(f"数据形状: {data_shape}")
        print(f"序列长度: {data_shape[1]}, 特征维度: {data_shape[2]}")
        
        # 2. 初始化优化的VAE-GAN模型
        print("\n2. 初始化优化的VAE-GAN模型...")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"使用设备: {device}")
        
        model = OptimizedVAEGAN(
            input_dim=data_shape[2],
            seq_len=data_shape[1],
            latent_dim=64,
            hidden_dim=128,
            device=device
        )
        
        # 3. 训练模型
        print(f"\n3. 训练优化模型 ({epochs} 轮)...")
        model.train_model(dataloader, epochs=epochs, save_interval=50)
        
        # 4. 绘制训练历史
        print("\n4. 保存训练历史...")
        model.plot_training_history(os.path.join(output_dir, 'optimized_training_history.png'))
        
        # 5. 生成增强数据
        print(f"\n5. 生成优化的增强数据 (每类 {samples_per_class} 个样本)...")
        
        # 获取原始数据的类别信息
        original_labels = get_original_labels(original_dir)
        unique_labels = np.unique(original_labels)
        print(f"发现 {len(unique_labels)} 个类别: {unique_labels}")
        
        all_generated_samples = []
        all_generated_labels = []
        
        for class_label in unique_labels:
            print(f"  生成类别 {class_label} 的样本...")
            
            # 生成多批次样本以确保质量
            generated_samples = []
            attempts = 0
            max_attempts = 10
            
            while len(generated_samples) < samples_per_class and attempts < max_attempts:
                # 使用不同温度生成样本
                temps = [0.8, 1.0, 1.2]
                for temp in temps:
                    vae_samples, gan_samples = model.generate_samples(
                        num_samples=samples_per_class, 
                        temperature=temp
                    )
                    
                    # 合并VAE和GAN样本
                    if len(vae_samples) > 0:
                        generated_samples.extend(vae_samples)
                    if len(gan_samples) > 0:
                        generated_samples.extend(gan_samples)
                    
                    if len(generated_samples) >= samples_per_class:
                        break
                
                attempts += 1
            
            # 选择最好的样本
            if len(generated_samples) >= samples_per_class:
                class_samples = np.array(generated_samples[:samples_per_class])
            else:
                print(f"    警告: 只生成了 {len(generated_samples)} 个样本 (目标: {samples_per_class})")
                class_samples = np.array(generated_samples)
            
            # 反归一化
            if len(class_samples) > 0:
                original_shape = class_samples.shape
                class_samples_flat = class_samples.reshape(-1, class_samples.shape[-1])
                class_samples_denorm = scaler.inverse_transform(class_samples_flat)
                class_samples_denorm = class_samples_denorm.reshape(original_shape)
                
                all_generated_samples.append(class_samples_denorm)
                all_generated_labels.extend([class_label] * len(class_samples_denorm))
        
        # 6. 保存增强数据
        print("\n6. 保存优化的增强数据...")
        
        sample_idx = 0
        for class_idx, class_label in enumerate(unique_labels):
            if class_idx < len(all_generated_samples):
                class_samples = all_generated_samples[class_idx]
                
                for i in range(len(class_samples)):
                    # 创建DataFrame
                    df = pd.DataFrame(class_samples[i])
                    df['label'] = class_label
                    
                    # 保存为CSV
                    filename = f"optimized_class{class_label}_sample{i}.csv"
                    filepath = os.path.join(output_dir, filename)
                    df.to_csv(filepath, index=False)
                    
                    sample_idx += 1
        
        print(f"成功生成并保存了 {sample_idx} 个优化样本到 {output_dir}")
        
        # 7. 生成对比可视化
        print("\n7. 生成可视化对比...")
        visualize_optimized_results(original_dir, output_dir, model, scaler, data_shape)
        
        # 8. 保存最终模型
        print("\n8. 保存最终优化模型...")
        model.save_checkpoint('checkpoints/final_optimized_vae_gan')
        
        print("\n" + "=" * 60)
        print("优化数据增强完成!")
        print(f"增强数据保存在: {output_dir}")
        print(f"模型保存在: checkpoints/final_optimized_vae_gan")
        print(f"训练历史图: {output_dir}/optimized_training_history.png")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()

def get_original_labels(data_dir):
    """获取原始数据的标签"""
    labels = []
    data_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    
    for file in data_files:
        try:
            df = pd.read_csv(os.path.join(data_dir, file))
            
            if 'label' in df.columns:
                label = df['label'].iloc[0]
            else:
                # 从文件名提取标签
                import re
                numbers = re.findall(r'\d+', file)
                label = int(numbers[0]) if numbers else 0
            
            labels.append(int(label))
        except:
            labels.append(0)
    
    return np.array(labels)

def visualize_optimized_results(original_dir, output_dir, model, scaler, data_shape):
    """可视化优化结果对比"""
    try:
        # 加载一些原始数据样本
        original_files = [f for f in os.listdir(original_dir) if f.endswith('.csv')][:3]
        original_samples = []
        
        for file in original_files:
            df = pd.read_csv(os.path.join(original_dir, file))
            if 'label' in df.columns:
                df = df.drop('label', axis=1)
            data = df.values.astype(np.float32)
            
            # 截断或填充
            seq_len = data_shape[1]
            if len(data) > seq_len:
                data = data[:seq_len]
            else:
                padding = np.zeros((seq_len - len(data), data.shape[1]))
                data = np.vstack([data, padding])
            
            original_samples.append(data)
        
        # 生成一些样本
        vae_samples, gan_samples = model.generate_samples(num_samples=3)
        
        if len(vae_samples) == 0:
            print("警告: 没有生成VAE样本")
            return
        
        # 反归一化生成的样本
        original_shape = vae_samples.shape
        vae_flat = vae_samples.reshape(-1, vae_samples.shape[-1])
        vae_denorm = scaler.inverse_transform(vae_flat)
        vae_denorm = vae_denorm.reshape(original_shape)
        
        # 绘制对比图
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        
        for i in range(3):
            # 原始数据
            axes[i, 0].plot(original_samples[i][:, 0])
            axes[i, 0].set_title(f'Original Sample {i+1}')
            axes[i, 0].set_xlabel('Time')
            axes[i, 0].set_ylabel('Value')
            axes[i, 0].grid(True)
            
            # VAE生成数据
            if i < len(vae_denorm):
                axes[i, 1].plot(vae_denorm[i][:, 0])
                axes[i, 1].set_title(f'VAE Generated Sample {i+1}')
                axes[i, 1].set_xlabel('Time')
                axes[i, 1].set_ylabel('Value')
                axes[i, 1].grid(True)
            
            # GAN生成数据 (如果有)
            if len(gan_samples) > i:
                gan_flat = gan_samples[:3].reshape(-1, gan_samples.shape[-1])
                gan_denorm = scaler.inverse_transform(gan_flat)
                gan_denorm = gan_denorm.reshape(gan_samples[:3].shape)
                
                axes[i, 2].plot(gan_denorm[i][:, 0])
                axes[i, 2].set_title(f'GAN Generated Sample {i+1}')
                axes[i, 2].set_xlabel('Time')
                axes[i, 2].set_ylabel('Value')
                axes[i, 2].grid(True)
            else:
                axes[i, 2].text(0.5, 0.5, 'No GAN Sample', ha='center', va='center')
                axes[i, 2].set_title(f'GAN Generated Sample {i+1}')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'optimized_sample_comparison.png'), dpi=300)
        plt.close()
        
        print(f"优化样本对比图保存到: {output_dir}/optimized_sample_comparison.png")
        
    except Exception as e:
        print(f"可视化过程中出现错误: {e}")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='优化VAE-GAN时序数据增强')
    parser.add_argument('--original_dir', type=str, default='original',
                       help='原始数据目录')
    parser.add_argument('--output_dir', type=str, default='augmented_data_optimized',
                       help='输出目录')
    parser.add_argument('--samples_per_class', type=int, default=25,
                       help='每个类别生成的样本数')
    parser.add_argument('--seq_len', type=int, default=400,
                       help='序列长度')
    parser.add_argument('--epochs', type=int, default=300,
                       help='训练轮数')
    
    args = parser.parse_args()
    
    run_optimized_augmentation(
        original_dir=args.original_dir,
        output_dir=args.output_dir,
        samples_per_class=args.samples_per_class,
        seq_len=args.seq_len,
        epochs=args.epochs
    )

if __name__ == "__main__":
    main()