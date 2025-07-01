import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用于显示中文
plt.rcParams['axes.unicode_minus'] = False    # 用于显示负号

def load_original_data(data_dir='original_processed'):
    """加载原始数据"""
    data_files = sorted([f for f in os.listdir(data_dir) if f.startswith('data') and f.endswith('.csv')])
    data_files = sorted(data_files, key=lambda x: int(x.replace('data', '').replace('.csv', '')))
    
    dataframes = []
    labels = []
    
    print(f"正在加载 {len(data_files)} 个原始数据文件...")
    
    for file in data_files:
        try:
            df = pd.read_csv(os.path.join(data_dir, file))
            if 'label' in df.columns:
                label = df['label'].iloc[0]
                labels.append(label)
                df = df.drop('label', axis=1)  # 删除标签列以获取特征数据
                dataframes.append(df)
            else:
                print(f"警告: {file} 中未找到标签")
        except Exception as e:
            print(f"加载 {file} 时出错: {e}")
    
    return dataframes, np.array(labels)

def load_augmented_data(data_dir='augmented_data_cc2'):
    """加载增强数据"""
    # 直接从npz文件加载
    npz_file = os.path.join(data_dir, 'augmented_dataset.npz')
    if os.path.exists(npz_file):
        try:
            data = np.load(npz_file)
            augmented_data = data['data']
            augmented_labels = data['labels']
            print(f"从NPZ文件加载了 {len(augmented_labels)} 个增强样本")
            return augmented_data, augmented_labels
        except Exception as e:
            print(f"加载NPZ文件出错: {e}")
    
    # 如果NPZ不存在，尝试加载CSV文件
    data_files = sorted([f for f in os.listdir(data_dir) if f.startswith('augmented_data_cc2') and f.endswith('.csv')])
    data_files = sorted(data_files, key=lambda x: int(x.replace('augmented_data_cc2', '').replace('.csv', '')))
    
    dataframes = []
    labels = []
    
    print(f"正在加载 {len(data_files)} 个增强数据文件...")
    
    for file in data_files:
        try:
            df = pd.read_csv(os.path.join(data_dir, file))
            # 确定标签位置
            if 'label' in df.columns:
                label = df['label'].iloc[0]
                df = df.drop('label', axis=1)  # 删除标签列以获取特征数据
            else:
                # 尝试从第一行获取标签值
                label = df.iloc[0, 0]
                df = df.iloc[1:] if len(df) > 1 else df
            
            labels.append(label)
            dataframes.append(df)
        except Exception as e:
            print(f"加载 {file} 时出错: {e}")
    
    return dataframes, np.array(labels)

def preprocess_data(dataframes_or_data, max_seq_len=100):
    """预处理数据：填充或截断序列"""
    # 如果输入已经是数组，直接返回
    if isinstance(dataframes_or_data, np.ndarray):
        return dataframes_or_data
    
    data_arrays = []
    
    for df in dataframes_or_data:
        # 提取传感器数据
        sensor_data = df.values.astype(np.float32)
        
        # 填充或截断序列
        seq_len = sensor_data.shape[0]
        feat_dim = sensor_data.shape[1]
        
        if seq_len > max_seq_len:
            # 截断
            processed_data = sensor_data[:max_seq_len, :]
        else:
            # 填充
            processed_data = np.zeros((max_seq_len, feat_dim))
            processed_data[:seq_len, :] = sensor_data
        
        data_arrays.append(processed_data)
    
    # 转换为numpy数组
    return np.array(data_arrays)

def plot_class_distribution(original_labels, augmented_labels, save_path='augmented_data/class_distribution.png'):
    """绘制类别分布对比图"""
    plt.figure(figsize=(12, 6))
    
    # 统计每个类别的样本数
    all_labels = np.unique(np.concatenate([original_labels, augmented_labels]))
    orig_class_counts = np.bincount(original_labels, minlength=len(all_labels))
    aug_class_counts = np.bincount(augmented_labels, minlength=len(all_labels))
    
    # 创建柱状图
    classes = np.arange(len(all_labels))
    width = 0.35
    
    plt.bar(classes - width/2, orig_class_counts, width, label='原始数据')
    plt.bar(classes + width/2, aug_class_counts, width, label='增强数据')
    
    plt.xlabel('类别')
    plt.ylabel('样本数量')
    plt.title('原始数据与增强数据的类别分布对比')
    plt.xticks(classes)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 在柱子上方显示具体数量
    for i, v in enumerate(orig_class_counts):
        plt.text(i - width/2, v + 0.5, str(v), ha='center')
    for i, v in enumerate(aug_class_counts):
        plt.text(i + width/2, v + 0.5, str(v), ha='center')
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"类别分布图已保存至 {save_path}")
    plt.close()

def plot_feature_distributions(original_data, augmented_data, original_labels, augmented_labels, save_dir='augmented_data_cc2'):
    """绘制特征分布图(PCA)"""
    # 确保输出目录存在
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 将3D序列数据转换为2D
    orig_flat = original_data.reshape(original_data.shape[0], -1)
    aug_flat = augmented_data.reshape(augmented_data.shape[0], -1)
    
    # 归一化数据
    scaler = MinMaxScaler()
    all_data = np.concatenate([orig_flat, aug_flat], axis=0)
    all_data_scaled = scaler.fit_transform(all_data)
    
    # 使用PCA降维
    pca = PCA(n_components=2)
    all_data_pca = pca.fit_transform(all_data_scaled)
    
    orig_pca = all_data_pca[:len(orig_flat)]
    aug_pca = all_data_pca[len(orig_flat):]
    
    # 绘制PCA分布图
    plt.figure(figsize=(16, 8))
    
    # 原始数据
    plt.subplot(1, 2, 1)
    for label in np.unique(original_labels):
        mask = original_labels == label
        plt.scatter(orig_pca[mask, 0], orig_pca[mask, 1], label=f'类别 {label}', alpha=0.8)
    plt.title('原始数据的PCA分布')
    plt.xlabel('主成分1')
    plt.ylabel('主成分2')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 增强数据
    plt.subplot(1, 2, 2)
    for label in np.unique(augmented_labels):
        mask = augmented_labels == label
        plt.scatter(aug_pca[mask, 0], aug_pca[mask, 1], label=f'类别 {label}', alpha=0.8)
    plt.title('增强数据的PCA分布')
    plt.xlabel('主成分1')
    plt.ylabel('主成分2')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'feature_distributions_pca.png'))
    print(f"特征分布图(PCA)已保存至 {os.path.join(save_dir, 'feature_distributions_pca.png')}")
    plt.close()

def plot_feature_densities(original_data, augmented_data, original_labels, augmented_labels, save_dir='augmented_data'):
    """绘制特征密度图"""
    # 确保输出目录存在
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 计算每个序列的平均值
    orig_means = np.mean(original_data, axis=(1, 2))
    aug_means = np.mean(augmented_data, axis=(1, 2))
    
    print(f"原始数据平均值形状: {orig_means.shape}")
    print(f"增强数据平均值形状: {aug_means.shape}")
    
    # 绘制总体密度图
    plt.figure(figsize=(12, 6))
    
    sns.kdeplot(orig_means, label='原始数据', shade=True)
    sns.kdeplot(aug_means, label='增强数据', shade=True)
    
    plt.title('原始数据与增强数据的特征平均值密度分布')
    plt.xlabel('特征平均值')
    plt.ylabel('密度')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'feature_density.png'))
    print(f"特征密度图已保存至 {os.path.join(save_dir, 'feature_density.png')}")
    plt.close()
    
    # 绘制按类别的密度图
    plt.figure(figsize=(15, 5*len(np.unique(np.concatenate([original_labels, augmented_labels])))))
    
    for i, label in enumerate(np.unique(np.concatenate([original_labels, augmented_labels]))):
        plt.subplot(len(np.unique(np.concatenate([original_labels, augmented_labels]))), 1, i+1)
        
        orig_mask = original_labels == label
        aug_mask = augmented_labels == label
        
        if np.sum(orig_mask) > 1:  # 需要至少2个样本才能绘制密度图
            sns.kdeplot(orig_means[orig_mask], label=f'原始数据-类别{label}', shade=True)
        elif np.sum(orig_mask) == 1:
            plt.axvline(x=orig_means[orig_mask][0], color='blue', linestyle='--', label=f'原始数据-类别{label}')
        else:
            print(f"类别 {label} 在原始数据中不存在")
            
        if np.sum(aug_mask) > 1:
            sns.kdeplot(aug_means[aug_mask], label=f'增强数据-类别{label}', shade=True)
        elif np.sum(aug_mask) == 1:
            plt.axvline(x=aug_means[aug_mask][0], color='orange', linestyle='--', label=f'增强数据-类别{label}')
        else:
            print(f"类别 {label} 在增强数据中不存在")
        
        plt.title(f'类别 {label} 的特征密度分布')
        plt.xlabel('特征平均值')
        plt.ylabel('密度')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'class_densities.png'))
    print(f"类别密度图已保存至 {os.path.join(save_dir, 'class_densities.png')}")
    plt.close()

def plot_temporal_patterns(original_data, augmented_data, original_labels, augmented_labels, save_dir='augmented_data'):
    """绘制时序模式图"""
    # 确保输出目录存在
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 为每个类别绘制一个包含3个子图的图形
    for label in np.unique(np.concatenate([original_labels, augmented_labels])):
        fig, axes = plt.subplots(3, 1, figsize=(12, 18), sharex=True)
        fig.suptitle(f'类别 {label} 的时序模式对比', fontsize=16)

        # 1. 原始数据
        ax1 = axes[0]
        orig_mask = original_labels == label
        if np.any(orig_mask):
            orig_class_data = original_data[orig_mask]
            orig_mean = np.mean(orig_class_data, axis=(0, 2))
            orig_std = np.std(orig_class_data, axis=(0, 2))
            x = np.arange(len(orig_mean))
            ax1.plot(x, orig_mean, label='原始数据平均值', color='blue')
            ax1.fill_between(x, orig_mean - orig_std, orig_mean + orig_std, alpha=0.2, color='blue')
            ax1.set_title('原始数据时序模式')
            ax1.set_ylabel('特征平均值')
            ax1.legend()
            ax1.grid(True, linestyle='--', alpha=0.7)
        else:
            ax1.text(0.5, 0.5, '无原始数据', ha='center', va='center', transform=ax1.transAxes)

        # 2. 增强数据
        ax2 = axes[1]
        aug_mask = augmented_labels == label
        if np.any(aug_mask):
            aug_class_data = augmented_data[aug_mask]
            aug_mean = np.mean(aug_class_data, axis=(0, 2))
            aug_std = np.std(aug_class_data, axis=(0, 2))
            x = np.arange(len(aug_mean))
            ax2.plot(x, aug_mean, label='增强数据平均值', color='red')
            ax2.fill_between(x, aug_mean - aug_std, aug_mean + aug_std, alpha=0.2, color='red')
            ax2.set_title('增强数据时序模式')
            ax2.set_ylabel('特征平均值')
            ax2.legend()
            ax2.grid(True, linestyle='--', alpha=0.7)
        else:
            ax2.text(0.5, 0.5, '无增强数据', ha='center', va='center', transform=ax2.transAxes)

        # 3. 原始数据 vs 增强数据
        ax3 = axes[2]
        if np.any(orig_mask):
            ax3.plot(x, orig_mean, label='原始数据平均值', color='blue')
            ax3.fill_between(x, orig_mean - orig_std, orig_mean + orig_std, alpha=0.2, color='blue')
        if np.any(aug_mask):
            ax3.plot(x, aug_mean, label='增强数据平均值', color='red')
            ax3.fill_between(x, aug_mean - aug_std, aug_mean + aug_std, alpha=0.2, color='red')
        
        ax3.set_title('原始数据与增强数据对比')
        ax3.set_xlabel('时间点')
        ax3.set_ylabel('特征平均值')
        ax3.legend()
        ax3.grid(True, linestyle='--', alpha=0.7)

        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        save_path = os.path.join(save_dir, f'temporal_class_{label}_comparison.png')
        plt.savefig(save_path)
        print(f"类别 {label} 的时序模式对比图已保存至 {save_path}")
        plt.close()

def plot_class_comparisons(original_data, augmented_data, original_labels, augmented_labels, save_dir='augmented_data_viz2'):
    """绘制类别间对比图"""
    # 确保输出目录存在
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 将3D序列数据转换为2D
    orig_flat = original_data.reshape(original_data.shape[0], -1)
    aug_flat = augmented_data.reshape(augmented_data.shape[0], -1)
    
    print(f"类别对比-扁平化后的原始数据形状: {orig_flat.shape}")
    print(f"类别对比-扁平化后的增强数据形状: {aug_flat.shape}")
    
    # 检查维度是否匹配
    if orig_flat.shape[1] != aug_flat.shape[1]:
        print(f"警告: 原始数据和增强数据的特征维度不匹配 ({orig_flat.shape[1]} vs {aug_flat.shape[1]})")
        print("将分别对原始数据和增强数据进行降维处理...")
        
        # 单独对原始数据和增强数据进行降维
        # 使用PCA降维
        pca_orig = PCA(n_components=min(2, orig_flat.shape[0], orig_flat.shape[1]))
        orig_pca = pca_orig.fit_transform(orig_flat)
        
        pca_aug = PCA(n_components=min(2, aug_flat.shape[0], aug_flat.shape[1]))
        aug_pca = pca_aug.fit_transform(aug_flat)
        
        print(f"PCA降维后的原始数据形状: {orig_pca.shape}")
        print(f"PCA降维后的增强数据形状: {aug_pca.shape}")
        
        # 使用t-SNE降维
        try:
            tsne_orig = TSNE(n_components=2, random_state=42)
            orig_tsne = tsne_orig.fit_transform(orig_flat)
            
            tsne_aug = TSNE(n_components=2, random_state=42)
            aug_tsne = tsne_aug.fit_transform(aug_flat)
            
            print(f"t-SNE降维后的原始数据形状: {orig_tsne.shape}")
            print(f"t-SNE降维后的增强数据形状: {aug_tsne.shape}")
        except Exception as e:
            print(f"t-SNE降维失败: {e}")
            # 如果t-SNE失败，使用PCA结果
            orig_tsne = orig_pca
            aug_tsne = aug_pca
    else:
        # 维度匹配，可以合并处理
        # 归一化数据
        scaler = MinMaxScaler()
        all_data = np.concatenate([orig_flat, aug_flat], axis=0)
        all_data_scaled = scaler.fit_transform(all_data)
        
        orig_scaled = all_data_scaled[:len(orig_flat)]
        aug_scaled = all_data_scaled[len(orig_flat):]
        
        # 使用PCA降维
        pca = PCA(n_components=2)
        all_data_pca = pca.fit_transform(all_data_scaled)
        
        orig_pca = all_data_pca[:len(orig_flat)]
        aug_pca = all_data_pca[len(orig_flat):]
        
        # 使用t-SNE降维
        try:
            tsne = TSNE(n_components=2, random_state=42)
            all_data_tsne = tsne.fit_transform(all_data_scaled)
            
            orig_tsne = all_data_tsne[:len(orig_flat)]
            aug_tsne = all_data_tsne[len(orig_flat):]
        except Exception as e:
            print(f"t-SNE降维失败: {e}")
            # 如果t-SNE失败，使用PCA结果
            orig_tsne = orig_pca
            aug_tsne = aug_pca
    
    # 1. 原始数据三类对比
    plt.figure(figsize=(12, 10))
    plt.subplot(2, 1, 1)
    for label in np.unique(original_labels):
        mask = original_labels == label
        plt.scatter(orig_pca[mask, 0], orig_pca[mask, 1], label=f'原始-类别{label}')
    plt.title('原始数据三类对比 (PCA)')
    plt.xlabel('主成分1')
    plt.ylabel('主成分2')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.subplot(2, 1, 2)
    for label in np.unique(original_labels):
        mask = original_labels == label
        plt.scatter(orig_tsne[mask, 0], orig_tsne[mask, 1], label=f'原始-类别{label}')
    plt.title('原始数据三类对比 (t-SNE)')
    plt.xlabel('维度1')
    plt.ylabel('维度2')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'original_classes_comparison.png'))
    print(f"原始数据类别对比图已保存至 {os.path.join(save_dir, 'original_classes_comparison.png')}")
    plt.close()
    
    # 2. 增强数据三类对比
    plt.figure(figsize=(12, 10))
    plt.subplot(2, 1, 1)
    for label in np.unique(augmented_labels):
        mask = augmented_labels == label
        plt.scatter(aug_pca[mask, 0], aug_pca[mask, 1], label=f'增强-类别{label}')
    plt.title('增强数据三类对比 (PCA)')
    plt.xlabel('主成分1')
    plt.ylabel('主成分2')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.subplot(2, 1, 2)
    for label in np.unique(augmented_labels):
        mask = augmented_labels == label
        plt.scatter(aug_tsne[mask, 0], aug_tsne[mask, 1], label=f'增强-类别{label}')
    plt.title('增强数据三类对比 (t-SNE)')
    plt.xlabel('维度1')
    plt.ylabel('维度2')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'augmented_classes_comparison.png'))
    print(f"增强数据类别对比图已保存至 {os.path.join(save_dir, 'augmented_classes_comparison.png')}")
    plt.close()
    
    # 3. 所有六类数据对比（3类原始 + 3类增强）
    # 注意：由于原始和增强数据是分别降维的，它们不在同一个坐标系中
    # 我们需要重新进行降维以获得一个统一的坐标系
    
    # 先对所有数据进行归一化
    scaler_orig = MinMaxScaler()
    orig_flat_scaled = scaler_orig.fit_transform(orig_flat)
    
    scaler_aug = MinMaxScaler()
    aug_flat_scaled = scaler_aug.fit_transform(aug_flat)
    
    # 然后分别降维到2维
    pca_combined = PCA(n_components=2)
    # 使用已经降维的数据进行联合PCA
    combined_data = np.vstack([orig_pca, aug_pca])
    combined_pca = pca_combined.fit_transform(combined_data)
    
    orig_pca_combined = combined_pca[:len(orig_pca)]
    aug_pca_combined = combined_pca[len(orig_pca):]
    
    # 对t-SNE也做类似处理
    try:
        tsne_combined = TSNE(n_components=2, random_state=42, perplexity=min(30, len(orig_tsne) + len(aug_tsne) - 1))
        combined_tsne = tsne_combined.fit_transform(np.vstack([orig_tsne, aug_tsne]))
        
        orig_tsne_combined = combined_tsne[:len(orig_tsne)]
        aug_tsne_combined = combined_tsne[len(orig_tsne):]
    except Exception as e:
        print(f"联合t-SNE降维失败: {e}")
        # 如果联合t-SNE失败，使用原始的单独降维结果
        orig_tsne_combined = orig_tsne
        aug_tsne_combined = aug_tsne
    
    plt.figure(figsize=(15, 12))
    
    # PCA对比
    plt.subplot(2, 1, 1)
    markers = ['o', 's', '^', 'D', 'p', '*']
    colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow']
    
    # 原始数据
    for i, label in enumerate(np.unique(original_labels)):
        mask = original_labels == label
        plt.scatter(orig_pca_combined[mask, 0], orig_pca_combined[mask, 1], 
                   label=f'原始-类别{label}',
                   marker=markers[i], 
                   color=colors[i],
                   edgecolors='black',
                   alpha=0.7)
    
    # 增强数据
    for i, label in enumerate(np.unique(augmented_labels)):
        mask = augmented_labels == label
        plt.scatter(aug_pca_combined[mask, 0], aug_pca_combined[mask, 1], 
                   label=f'增强-类别{label}',
                   marker=markers[i+3], 
                   color=colors[i+3],
                   edgecolors='black',
                   alpha=0.7)
    
    plt.title('原始数据与增强数据六类对比 (PCA)')
    plt.xlabel('主成分1')
    plt.ylabel('主成分2')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # t-SNE对比
    plt.subplot(2, 1, 2)
    
    # 原始数据
    for i, label in enumerate(np.unique(original_labels)):
        mask = original_labels == label
        plt.scatter(orig_tsne_combined[mask, 0], orig_tsne_combined[mask, 1], 
                   label=f'原始-类别{label}',
                   marker=markers[i], 
                   color=colors[i],
                   edgecolors='black',
                   alpha=0.7)
    
    # 增强数据
    for i, label in enumerate(np.unique(augmented_labels)):
        mask = augmented_labels == label
        plt.scatter(aug_tsne_combined[mask, 0], aug_tsne_combined[mask, 1], 
                   label=f'增强-类别{label}',
                   marker=markers[i+3], 
                   color=colors[i+3],
                   edgecolors='black',
                   alpha=0.7)
    
    plt.title('原始数据与增强数据六类对比 (t-SNE)')
    plt.xlabel('维度1')
    plt.ylabel('维度2')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'all_six_classes_comparison.png'))
    print(f"六类数据对比图已保存至 {os.path.join(save_dir, 'all_six_classes_comparison.png')}")
    plt.close()
    
    # 4. 创建一个更清晰的网格布局可视化
    plt.figure(figsize=(20, 12))
    
    # 设置一个3x2的网格布局
    gs = plt.GridSpec(2, 3)
    
    # 1. 原始数据PCA
    ax1 = plt.subplot(gs[0, 0])
    for label in np.unique(original_labels):
        mask = original_labels == label
        ax1.scatter(orig_pca[mask, 0], orig_pca[mask, 1], 
                   label=f'类别{label}',
                   s=80,
                   alpha=0.8)
    ax1.set_title('原始数据 (PCA)', fontsize=14)
    ax1.set_xlabel('主成分1')
    ax1.set_ylabel('主成分2')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # 2. 增强数据PCA
    ax2 = plt.subplot(gs[0, 1])
    for label in np.unique(augmented_labels):
        mask = augmented_labels == label
        ax2.scatter(aug_pca[mask, 0], aug_pca[mask, 1], 
                   label=f'类别{label}',
                   s=80,
                   alpha=0.8)
    ax2.set_title('增强数据 (PCA)', fontsize=14)
    ax2.set_xlabel('主成分1')
    ax2.set_ylabel('主成分2')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # 3. 六类数据PCA对比
    ax3 = plt.subplot(gs[0, 2])
    # 原始数据
    for i, label in enumerate(np.unique(original_labels)):
        mask = original_labels == label
        ax3.scatter(orig_pca_combined[mask, 0], orig_pca_combined[mask, 1], 
                   label=f'原始-类别{label}',
                   marker=markers[i], 
                   color=colors[i],
                   edgecolors='black',
                   s=80,
                   alpha=0.8)
    
    # 增强数据
    for i, label in enumerate(np.unique(augmented_labels)):
        mask = augmented_labels == label
        ax3.scatter(aug_pca_combined[mask, 0], aug_pca_combined[mask, 1], 
                   label=f'增强-类别{label}',
                   marker=markers[i+3], 
                   color=colors[i+3],
                   edgecolors='black',
                   s=80,
                   alpha=0.8)
    ax3.set_title('原始与增强数据对比 (PCA)', fontsize=14)
    ax3.set_xlabel('主成分1')
    ax3.set_ylabel('主成分2')
    ax3.legend()
    ax3.grid(True, linestyle='--', alpha=0.7)
    
    # 4. 原始数据t-SNE
    ax4 = plt.subplot(gs[1, 0])
    for label in np.unique(original_labels):
        mask = original_labels == label
        ax4.scatter(orig_tsne[mask, 0], orig_tsne[mask, 1], 
                   label=f'类别{label}',
                   s=80,
                   alpha=0.8)
    ax4.set_title('原始数据 (t-SNE)', fontsize=14)
    ax4.set_xlabel('维度1')
    ax4.set_ylabel('维度2')
    ax4.legend()
    ax4.grid(True, linestyle='--', alpha=0.7)
    
    # 5. 增强数据t-SNE
    ax5 = plt.subplot(gs[1, 1])
    for label in np.unique(augmented_labels):
        mask = augmented_labels == label
        ax5.scatter(aug_tsne[mask, 0], aug_tsne[mask, 1], 
                   label=f'类别{label}',
                   s=80,
                   alpha=0.8)
    ax5.set_title('增强数据 (t-SNE)', fontsize=14)
    ax5.set_xlabel('维度1')
    ax5.set_ylabel('维度2')
    ax5.legend()
    ax5.grid(True, linestyle='--', alpha=0.7)
    
    # 6. 六类数据t-SNE对比
    ax6 = plt.subplot(gs[1, 2])
    # 原始数据
    for i, label in enumerate(np.unique(original_labels)):
        mask = original_labels == label
        ax6.scatter(orig_tsne_combined[mask, 0], orig_tsne_combined[mask, 1], 
                   label=f'原始-类别{label}',
                   marker=markers[i], 
                   color=colors[i],
                   edgecolors='black',
                   s=80,
                   alpha=0.8)
    
    # 增强数据
    for i, label in enumerate(np.unique(augmented_labels)):
        mask = augmented_labels == label
        ax6.scatter(aug_tsne_combined[mask, 0], aug_tsne_combined[mask, 1], 
                   label=f'增强-类别{label}',
                   marker=markers[i+3], 
                   color=colors[i+3],
                   edgecolors='black',
                   s=80,
                   alpha=0.8)
    ax6.set_title('原始与增强数据对比 (t-SNE)', fontsize=14)
    ax6.set_xlabel('维度1')
    ax6.set_ylabel('维度2')
    ax6.legend()
    ax6.grid(True, linestyle='--', alpha=0.7)
    
    plt.suptitle('原始数据与增强数据的类别分布对比', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'class_distribution_grid.png'), dpi=300)
    print(f"类别分布网格图已保存至 {os.path.join(save_dir, 'class_distribution_grid.png')}")
    plt.close()

def main():
    # 创建输出目录
    output_dir = 'augmented_data_viz2'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print("开始加载和处理数据...")
    
    try:
        # 加载原始数据和增强数据
        print("加载原始数据...")
        original_dataframes, original_labels = load_original_data(data_dir='original_processed')
        
        print("加载增强数据...")
        augmented_data_or_dfs, augmented_labels = load_augmented_data(data_dir='augmented_data_cc2')
        
        # 预处理数据
        print("预处理原始数据...")
        original_data = preprocess_data(original_dataframes, max_seq_len=400)
        
        print("预处理增强数据...")
        augmented_data = preprocess_data(augmented_data_or_dfs, max_seq_len=400)
        
        print(f"原始数据形状: {original_data.shape}, 标签: {original_labels.shape}")
        print(f"增强数据形状: {augmented_data.shape}, 标签: {augmented_labels.shape}")
        
        # 绘制类别分布图
        print("绘制类别分布图...")
        plot_class_distribution(original_labels, augmented_labels, save_path=os.path.join(output_dir, 'class_distribution.png'))
        
        # 绘制特征分布图 (PCA)
        print("绘制特征分布图 (PCA)...")
        plot_feature_distributions(original_data, augmented_data, original_labels, augmented_labels, save_dir=output_dir)
        
        # 绘制特征密度图
        print("绘制特征密度图...")
        plot_feature_densities(original_data, augmented_data, original_labels, augmented_labels, save_dir=output_dir)
        
        # 绘制时序模式图
        print("绘制时序模式图...")
        plot_temporal_patterns(original_data, augmented_data, original_labels, augmented_labels, save_dir=output_dir)
        
        # 绘制类别对比图 (新增)
        print("绘制类别对比图...")
        plot_class_comparisons(original_data, augmented_data, original_labels, augmented_labels, save_dir=output_dir)
        
        print("所有可视化图表已生成!")
    except Exception as e:
        import traceback
        print(f"处理数据时出错: {e}")
        traceback.print_exc()

if __name__ == '__main__':
    main() 