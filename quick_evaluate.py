#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
快速数据质量评估脚本
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.stats import wasserstein_distance
import warnings
warnings.filterwarnings('ignore')

def load_csv_files(data_dir, seq_len=400):
    """加载CSV文件"""
    data_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    data_list = []
    labels = []
    
    for file in data_files:
        try:
            df = pd.read_csv(os.path.join(data_dir, file))
            
            # 提取标签
            if 'label' in df.columns:
                label = df['label'].iloc[0]
                df = df.drop('label', axis=1)
            else:
                import re
                numbers = re.findall(r'\d+', file)
                label = int(numbers[0]) if numbers else 0
            
            data = df.values.astype(np.float32)
            
            # 统一长度
            if len(data) > seq_len:
                data = data[:seq_len]
            elif len(data) < seq_len:
                padding = np.zeros((seq_len - len(data), data.shape[1]))
                data = np.vstack([data, padding])
            
            data_list.append(data)
            labels.append(label)
            
        except Exception as e:
            print(f"Error loading {file}: {e}")
            continue
    
    return np.array(data_list), np.array(labels)

def compute_statistics(data, labels):
    """计算基本统计信息"""
    print(f"数据形状: {data.shape}")
    print(f"标签分布: {np.bincount(labels)}")
    print(f"数据范围: [{np.min(data):.4f}, {np.max(data):.4f}]")
    print(f"数据均值: {np.mean(data):.4f}")
    print(f"数据标准差: {np.std(data):.4f}")

def compare_distributions(orig_data, aug_data, output_dir):
    """比较分布"""
    plt.figure(figsize=(15, 5))
    
    # 1. 时序图对比
    plt.subplot(1, 3, 1)
    for i in range(min(3, len(orig_data))):
        plt.plot(orig_data[i][:, 0], alpha=0.7, label=f'Original {i+1}')
    for i in range(min(3, len(aug_data))):
        plt.plot(aug_data[i][:, 0], '--', alpha=0.7, label=f'Generated {i+1}')
    plt.title('Time Series Comparison')
    plt.legend()
    
    # 2. 分布直方图
    plt.subplot(1, 3, 2)
    orig_flat = orig_data.flatten()
    aug_flat = aug_data.flatten()
    plt.hist(orig_flat, bins=50, alpha=0.7, label='Original', density=True)
    plt.hist(aug_flat, bins=50, alpha=0.7, label='Generated', density=True)
    plt.title('Value Distribution')
    plt.legend()
    
    # 3. Wasserstein距离
    wd = wasserstein_distance(orig_flat, aug_flat)
    plt.subplot(1, 3, 3)
    plt.text(0.5, 0.5, f'Wasserstein Distance: {wd:.4f}\n(Lower is better)', 
             ha='center', va='center', fontsize=12, bbox=dict(boxstyle="round", facecolor='lightblue'))
    plt.title('Distribution Similarity')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'quality_comparison.png'), dpi=300)
    plt.close()
    
    return wd

def main():
    """主函数"""
    original_dir = 'original'
    augmented_dir = 'augmented_data_optimized'  # 修改为优化后的目录
    output_dir = 'evaluation_results_optimized'
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*60)
    print("快速数据质量评估")
    print("="*60)
    
    # 加载数据
    print("\n1. 加载原始数据...")
    orig_data, orig_labels = load_csv_files(original_dir)
    compute_statistics(orig_data, orig_labels)
    
    print("\n2. 加载增强数据...")
    aug_data, aug_labels = load_csv_files(augmented_dir)
    compute_statistics(aug_data, aug_labels)
    
    # 截断到相同长度
    min_len = min(orig_data.shape[1], aug_data.shape[1])
    orig_data = orig_data[:, :min_len, :]
    aug_data = aug_data[:, :min_len, :]
    
    print(f"\n3. 数据对齐后形状: Original {orig_data.shape}, Augmented {aug_data.shape}")
    
    # 比较分布
    print("\n4. 比较数据分布...")
    wd = compare_distributions(orig_data, aug_data, output_dir)
    
    print(f"\n5. 评估结果:")
    print(f"   - Wasserstein距离: {wd:.4f} (越小越好)")
    print(f"   - 样本增加数量: {len(aug_data)} (原始: {len(orig_data)})")
    print(f"   - 类别平衡性: 增强数据各类别样本数 = {np.bincount(aug_labels)}")
    
    # 简单的数据质量评分
    if wd < 0.1:
        quality = "优秀"
    elif wd < 0.3:
        quality = "良好"
    elif wd < 0.5:
        quality = "一般"
    else:
        quality = "较差"
    
    print(f"   - 数据质量评估: {quality}")
    print(f"\n结果保存在: {output_dir}/quality_comparison.png")
    print("="*60)

if __name__ == "__main__":
    main()