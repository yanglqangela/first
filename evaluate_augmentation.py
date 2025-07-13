#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
数据增强评估框架
基于以下评估维度:
- 分类性能: 原始 vs 增强数据 (Accuracy, F1, Recall)
- 判别能力: 判别器训练 (AUC, 0.5越好)
- 分布可视化: t-SNE, UMAP (overlap 越好)
- 分布相似: FID, 均值协方差 (越低越好)
- 时间结构: ACF, DTW, FFT (越接近越好)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import jensenshannon
from scipy.fft import fft, fftfreq
from scipy import signal
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
from sklearn.decomposition import PCA
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance
import numpy as np

def compute_fid_with_pca(original_data, augmented_data):
    """使用PCA降维后计算FID"""
    orig_flat = original_data.reshape(original_data.shape[0], -1)
    aug_flat = augmented_data.reshape(augmented_data.shape[0], -1)
    orig_pca, aug_pca = reduce_dimensions(orig_flat, aug_flat)

    mu1, sigma1 = np.mean(orig_pca, axis=0), np.cov(orig_pca, rowvar=False)
    mu2, sigma2 = np.mean(aug_pca, axis=0), np.cov(aug_pca, rowvar=False)

    return compute_frechet_distance(mu1, sigma1, mu2, sigma2)

def compute_statistical_distances_with_clipping(original_data, augmented_data):
    """计算 Wasserstein 距离和 JS 散度，先裁剪维度避免内存爆炸"""
    orig_flat = original_data.reshape(original_data.shape[0], -1)
    aug_flat = augmented_data.reshape(augmented_data.shape[0], -1)
    min_features = min(orig_flat.shape[1], aug_flat.shape[1])
    orig_flat = orig_flat[:, :min_features]
    aug_flat = aug_flat[:, :min_features]

    distances = {}

    # Wasserstein
    wasserstein_dists = [
        wasserstein_distance(orig_flat[:, i], aug_flat[:, i])
        for i in range(min_features)
    ]
    distances['wasserstein'] = np.mean(wasserstein_dists)

    # Jensen-Shannon
    hist_orig, _ = np.histogram(orig_flat.flatten(), bins=50, density=True)
    hist_aug, _ = np.histogram(aug_flat.flatten(), bins=50, density=True)
    hist_orig += 1e-10
    hist_aug += 1e-10
    distances['jensen_shannon'] = jensenshannon(hist_orig, hist_aug)

    return distances

def reduce_dimensions(orig_flat, aug_flat, max_dim=64):
    """对原始和增强数据做统一的PCA降维"""
    min_features = min(orig_flat.shape[1], aug_flat.shape[1])
    orig_flat = orig_flat[:, :min_features]
    aug_flat = aug_flat[:, :min_features]
    pca_dim = min(max_dim, min(orig_flat.shape[0], min_features))
    pca = PCA(n_components=pca_dim)
    orig_pca = pca.fit_transform(orig_flat)
    aug_pca = pca.transform(aug_flat)
    return orig_pca, aug_pca

def compute_frechet_distance(mu1, sigma1, mu2, sigma2):
    """计算Fréchet距离（FID核心公式）"""
    diff = mu1 - mu2
    covmean = np.sqrt(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fid

class AugmentationEvaluator:
    """数据增强评估器"""
    
    def __init__(self, original_dir='original', augmented_dir='augmented_data_improved', output_dir='evaluation_results'):
        self.original_dir = original_dir
        self.augmented_dir = augmented_dir
        self.output_dir = output_dir
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 存储评估结果
        self.results = {}
        
    def load_data(self):
        """加载原始数据和增强数据"""
        print("Loading data...")
        
        # 加载原始数据
        orig_data, orig_labels = self._load_csv_files(self.original_dir)
        print(f"Loaded {len(orig_data)} original samples")
        
        # 加载增强数据
        aug_data, aug_labels = self._load_csv_files(self.augmented_dir)
        print(f"Loaded {len(aug_data)} augmented samples")
        
        # 预处理数据
        self.original_data, self.original_labels = self._preprocess_data(orig_data, orig_labels)
        self.augmented_data, self.augmented_labels = self._preprocess_data(aug_data, aug_labels)
        
        print(f"Original data shape: {self.original_data.shape}")
        print(f"Augmented data shape: {self.augmented_data.shape}")
        
    def _load_csv_files(self, data_dir):
        """加载CSV文件"""
        data_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
        data_files = sorted(data_files)
        
        dataframes = []
        labels = []
        
        for file in data_files:
            try:
                df = pd.read_csv(os.path.join(data_dir, file))
                
                # 提取标签
                if 'label' in df.columns:
                    label = df['label'].iloc[0]
                    df = df.drop('label', axis=1)
                else:
                    # 从文件名提取标签
                    import re
                    numbers = re.findall(r'\d+', file)
                    label = int(numbers[0]) if numbers else 0
                
                dataframes.append(df.values.astype(np.float32))
                labels.append(int(label))
                
            except Exception as e:
                print(f"Error loading {file}: {e}")
                
        return dataframes, np.array(labels)
    
    def _preprocess_data(self, dataframes, labels, max_seq_len=None):
        """预处理数据"""
        if not dataframes:
            return np.array([]), np.array([])
            
        # 确定最大序列长度
        if max_seq_len is None:
            max_seq_len = max(df.shape[0] for df in dataframes)
        
        # 填充或截断序列
        processed_data = []
        for df in dataframes:
            if len(df.shape) == 1:
                df = df.reshape(-1, 1)
                
            seq_len, feat_dim = df.shape
            
            if seq_len > max_seq_len:
                # 截断
                processed = df[:max_seq_len, :]
            else:
                # 填充
                processed = np.zeros((max_seq_len, feat_dim))
                processed[:seq_len, :] = df
                
            processed_data.append(processed)
        
        return np.array(processed_data), labels
    
    def evaluate_classification_performance(self):
        
        print("\n1. Evaluating Classification Performance...")

        # 展平数据
        X_orig_flat = self.original_data.reshape(self.original_data.shape[0], -1)
        X_aug_flat = self.augmented_data.reshape(self.augmented_data.shape[0], -1)

# 对齐维度（截断到相同的特征长度）
        min_features = min(X_orig_flat.shape[1], X_aug_flat.shape[1])
        if X_orig_flat.shape[1] != X_aug_flat.shape[1]:
            print(f"[Warning] Feature size mismatch: orig={X_orig_flat.shape[1]}, aug={X_aug_flat.shape[1]}. Trimming to {min_features}.")
            X_orig_flat = X_orig_flat[:, :min_features]
            X_aug_flat = X_aug_flat[:, :min_features]

        # 评估原始数据
        orig_metrics = self._evaluate_classifier(X_orig_flat, self.original_labels, "Original Data")

        # 评估增强数据
        aug_metrics = self._evaluate_classifier(X_aug_flat, self.augmented_labels, "Augmented Data")

        # 评估混合数据
        X_combined = np.vstack([X_orig_flat, X_aug_flat])
        y_combined = np.hstack([self.original_labels, self.augmented_labels])
        combined_metrics = self._evaluate_classifier(X_combined, y_combined, "Combined Data")

        # 使用增强数据训练，原始数据测试
        cross_metrics = self._cross_evaluate(X_aug_flat, self.augmented_labels,
                                                X_orig_flat, self.original_labels)

        self.results['classification'] = {
            'original': orig_metrics,
            'augmented': aug_metrics,
            'combined': combined_metrics,
            'cross_validation': cross_metrics
        }

    # 可视化
        self._plot_classification_results()

        
    def _evaluate_classifier(self, X, y, data_name):
        """评估分类器性能"""
        if len(np.unique(y)) < 2:
            return {'accuracy': 0, 'f1': 0, 'recall': 0, 'precision': 0}
            
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # 训练随机森林分类器
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)
        
        # 预测
        y_pred = clf.predict(X_test)
        
        # 计算指标
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        precision = precision_score(y_test, y_pred, average='weighted')
        
        print(f"{data_name} - Accuracy: {accuracy:.3f}, F1: {f1:.3f}, Recall: {recall:.3f}, Precision: {precision:.3f}")
        
        return {
            'accuracy': accuracy,
            'f1': f1,
            'recall': recall,
            'precision': precision
        }
    
    def _cross_evaluate(self, X_train, y_train, X_test, y_test):
        """交叉评估: 增强数据训练，原始数据测试"""
        if len(np.unique(y_train)) < 2:
            return {'accuracy': 0, 'f1': 0, 'recall': 0, 'precision': 0}
            
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        precision = precision_score(y_test, y_pred, average='weighted')
        
        print(f"Cross Validation (Aug->Orig) - Accuracy: {accuracy:.3f}, F1: {f1:.3f}")
        
        return {
            'accuracy': accuracy,
            'f1': f1,
            'recall': recall,
            'precision': precision
        }
    def evaluate_discriminator_ability(self):
        """评估判别能力"""
        print("\n2. Evaluating Discriminator Ability...")

        # 展平数据
        X_orig_flat = self.original_data.reshape(self.original_data.shape[0], -1)
        X_aug_flat = self.augmented_data.reshape(self.augmented_data.shape[0], -1)

        # 对齐维度
        min_features = min(X_orig_flat.shape[1], X_aug_flat.shape[1])
        if X_orig_flat.shape[1] != X_aug_flat.shape[1]:
            print(f"[Warning] Feature size mismatch in Discriminator: orig={X_orig_flat.shape[1]}, aug={X_aug_flat.shape[1]}. Trimming to {min_features}.")
            X_orig_flat = X_orig_flat[:, :min_features]
            X_aug_flat = X_aug_flat[:, :min_features]

        # 标签: 0=真实, 1=生成
        X_combined = np.vstack([X_orig_flat, X_aug_flat])
        y_discriminator = np.hstack([
            np.zeros(len(X_orig_flat)),
            np.ones(len(X_aug_flat))
        ])

        # 训练判别器
        X_train, X_test, y_train, y_test = train_test_split(
            X_combined, y_discriminator, test_size=0.3, random_state=42
        )

        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)

        # 预测概率
        y_pred_proba = clf.predict_proba(X_test)[:, 1]

        # 计算AUC (越接近0.5越好，说明难以区分)
        auc = roc_auc_score(y_test, y_pred_proba)
        discriminator_quality = abs(auc - 0.5)  # 距离0.5的偏差

        print(f"Discriminator AUC: {auc:.3f} (closer to 0.5 is better)")
        print(f"Discriminator Quality Score: {discriminator_quality:.3f} (lower is better)")

        self.results['discriminator'] = {
            'auc': auc,
            'quality_score': discriminator_quality
        }
      
        
    def evaluate_distribution_similarity(self):
        print("\n3. Evaluating Distribution Similarity...")

        # 使用新函数
        fid_score = compute_fid_with_pca(self.original_data, self.augmented_data)
        stat_distances = compute_statistical_distances_with_clipping(
            self.original_data, self.augmented_data
        )

        self.results['distribution'] = {
            'fid': fid_score,
            'statistical_distances': stat_distances
        }

        print(f"FID Score: {fid_score:.3f} (lower is better)")
        for metric, value in stat_distances.items():
            print(f"{metric}: {value:.3f}")



    def _compute_fid(self):
        """计算FID分数"""
        def calculate_frechet_distance(mu1, sigma1, mu2, sigma2):
            """计算Fréchet距离"""
            diff = mu1 - mu2
            covmean = np.sqrt(sigma1.dot(sigma2))
            if np.iscomplexobj(covmean):
                covmean = covmean.real
            fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
            return fid
        
        # 展平数据
        orig_flat = self.original_data.reshape(self.original_data.shape[0], -1)
        aug_flat = self.augmented_data.reshape(self.augmented_data.shape[0], -1)
        
        # 计算统计量
        mu1, sigma1 = np.mean(orig_flat, axis=0), np.cov(orig_flat, rowvar=False)
        mu2, sigma2 = np.mean(aug_flat, axis=0), np.cov(aug_flat, rowvar=False)
        
        return calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
    

    
    def evaluate_temporal_structure(self):
        """评估时间结构相似性"""
        print("\n4. Evaluating Temporal Structure...")
        
        temporal_metrics = {}
        
        # 自相关函数 (ACF)
        acf_similarity = self._compute_acf_similarity()
        temporal_metrics['acf_similarity'] = acf_similarity
        
        # 频域分析 (FFT)
        fft_similarity = self._compute_fft_similarity()
        temporal_metrics['fft_similarity'] = fft_similarity
        
        # DTW距离
        dtw_distance = self._compute_dtw_distance()
        temporal_metrics['dtw_distance'] = dtw_distance
        
        self.results['temporal'] = temporal_metrics
        
        print(f"ACF Similarity: {acf_similarity:.3f} (higher is better)")
        print(f"FFT Similarity: {fft_similarity:.3f} (higher is better)")
        print(f"DTW Distance: {dtw_distance:.3f} (lower is better)")
    
    def _compute_acf_similarity(self):
        """计算自相关函数相似性"""
        def autocorrelation(x, max_lag=50):
            """计算自相关函数"""
            n = len(x)
            x = x - np.mean(x)
            autocorr = np.correlate(x, x, mode='full')
            autocorr = autocorr[n-1:n+max_lag]
            return autocorr / autocorr[0]
        
        # 对每个样本的第一个特征计算ACF
        orig_acfs = []
        aug_acfs = []
        
        for i in range(min(len(self.original_data), len(self.augmented_data))):
            orig_acf = autocorrelation(self.original_data[i, :, 0])
            aug_acf = autocorrelation(self.augmented_data[i, :, 0])
            
            orig_acfs.append(orig_acf)
            aug_acfs.append(aug_acf)
        
        # 计算平均相关性
        correlations = []
        for orig_acf, aug_acf in zip(orig_acfs, aug_acfs):
            corr = np.corrcoef(orig_acf, aug_acf)[0, 1]
            if not np.isnan(corr):
                correlations.append(corr)
        
        return np.mean(correlations) if correlations else 0
    
    def _compute_fft_similarity(self):
        """计算FFT谱相似性"""
        def compute_power_spectrum(x, target_length=None):
            """计算功率谱，并统一长度"""
            if target_length is not None:
                x = x[:target_length]
            fft_vals = fft(x)
            power_spectrum = np.abs(fft_vals) ** 2
            return power_spectrum[:len(power_spectrum) // 2]

        orig_spectra = []
        aug_spectra = []

        # 确定截断长度
        min_length = min(self.original_data.shape[1], self.augmented_data.shape[1])

        for i in range(min(len(self.original_data), len(self.augmented_data))):
            orig_spec = compute_power_spectrum(self.original_data[i, :, 0], target_length=min_length)
            aug_spec = compute_power_spectrum(self.augmented_data[i, :, 0], target_length=min_length)

            min_spec_len = min(len(orig_spec), len(aug_spec))
            orig_spectra.append(orig_spec[:min_spec_len])
            aug_spectra.append(aug_spec[:min_spec_len])

        # 计算平均相关性
        correlations = []
        for orig_spec, aug_spec in zip(orig_spectra, aug_spectra):
            if len(orig_spec) != len(aug_spec):
                continue  # 跳过不匹配的
            corr = np.corrcoef(orig_spec, aug_spec)[0, 1]
            if not np.isnan(corr):
                correlations.append(corr)

        return np.mean(correlations) if correlations else 0

    
    def _compute_dtw_distance(self):
        """计算DTW距离"""
        def dtw_distance(x, y):
            """简单的DTW实现"""
            n, m = len(x), len(y)
            dtw_matrix = np.inf * np.ones((n + 1, m + 1))
            dtw_matrix[0, 0] = 0
            
            for i in range(1, n + 1):
                for j in range(1, m + 1):
                    cost = abs(x[i-1] - y[j-1])
                    dtw_matrix[i, j] = cost + min(
                        dtw_matrix[i-1, j],      # insertion
                        dtw_matrix[i, j-1],      # deletion
                        dtw_matrix[i-1, j-1]     # match
                    )
            
            return dtw_matrix[n, m]
        
        distances = []
        n_samples = min(10, len(self.original_data), len(self.augmented_data))
        
        for i in range(n_samples):
            dist = dtw_distance(
                self.original_data[i, :, 0],
                self.augmented_data[i, :, 0]
            )
            distances.append(dist)
        
        return np.mean(distances)
    
    def visualize_distributions(self):
        """可视化分布"""
        print("\n5. Creating Distribution Visualizations...")
        
        # t-SNE可视化
        self._plot_tsne()
        
        # PCA可视化
        self._plot_pca()
        
        # 特征分布对比
        self._plot_feature_distributions()
    
    def _plot_tsne(self):
        """t-SNE可视化"""
        # 准备数据
        X_orig_flat = self.original_data.reshape(self.original_data.shape[0], -1)
        X_aug_flat = self.augmented_data.reshape(self.augmented_data.shape[0], -1)
        
        # 确保维度匹配
        min_features = min(X_orig_flat.shape[1], X_aug_flat.shape[1])
        X_orig_flat = X_orig_flat[:, :min_features]
        X_aug_flat = X_aug_flat[:, :min_features]
        
        X_combined = np.vstack([X_orig_flat, X_aug_flat])
        
        # 标准化
        scaler = MinMaxScaler()
        X_combined_scaled = scaler.fit_transform(X_combined)
        
        # t-SNE降维
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X_combined)-1))
        X_tsne = tsne.fit_transform(X_combined_scaled)
        
        # 分离结果
        X_orig_tsne = X_tsne[:len(X_orig_flat)]
        X_aug_tsne = X_tsne[len(X_orig_flat):]
        
        # 绘图
        plt.figure(figsize=(15, 5))
        
        # 按数据类型分组
        plt.subplot(1, 3, 1)
        plt.scatter(X_orig_tsne[:, 0], X_orig_tsne[:, 1], 
                   c='blue', alpha=0.6, label='Original', s=50)
        plt.scatter(X_aug_tsne[:, 0], X_aug_tsne[:, 1], 
                   c='red', alpha=0.6, label='Augmented', s=50)
        plt.title('t-SNE: Original vs Augmented')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 按类别分组 - 原始数据
        plt.subplot(1, 3, 2)
        for label in np.unique(self.original_labels):
            mask = self.original_labels == label
            plt.scatter(X_orig_tsne[mask, 0], X_orig_tsne[mask, 1], 
                       label=f'Original Class {label}', alpha=0.7, s=50)
        plt.title('t-SNE: Original Data by Class')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 按类别分组 - 增强数据
        plt.subplot(1, 3, 3)
        for label in np.unique(self.augmented_labels):
            mask = self.augmented_labels == label
            plt.scatter(X_aug_tsne[mask, 0], X_aug_tsne[mask, 1], 
                       label=f'Augmented Class {label}', alpha=0.7, s=50)
        plt.title('t-SNE: Augmented Data by Class')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'tsne_visualization.png'), dpi=300)
        plt.close()
    
    def _plot_pca(self):
        """PCA可视化"""
        # 准备数据
        X_orig_flat = self.original_data.reshape(self.original_data.shape[0], -1)
        X_aug_flat = self.augmented_data.reshape(self.augmented_data.shape[0], -1)
        
        # 确保维度匹配
        min_features = min(X_orig_flat.shape[1], X_aug_flat.shape[1])
        X_orig_flat = X_orig_flat[:, :min_features]
        X_aug_flat = X_aug_flat[:, :min_features]
        
        X_combined = np.vstack([X_orig_flat, X_aug_flat])
        
        # 标准化
        scaler = MinMaxScaler()
        X_combined_scaled = scaler.fit_transform(X_combined)
        
        # PCA降维
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_combined_scaled)
        
        # 分离结果
        X_orig_pca = X_pca[:len(X_orig_flat)]
        X_aug_pca = X_pca[len(X_orig_flat):]
        
        # 绘图
        plt.figure(figsize=(15, 5))
        
        # 按数据类型分组
        plt.subplot(1, 3, 1)
        plt.scatter(X_orig_pca[:, 0], X_orig_pca[:, 1], 
                   c='blue', alpha=0.6, label='Original', s=50)
        plt.scatter(X_aug_pca[:, 0], X_aug_pca[:, 1], 
                   c='red', alpha=0.6, label='Augmented', s=50)
        plt.title(f'PCA: Original vs Augmented\n(Explained Variance: {pca.explained_variance_ratio_.sum():.3f})')
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.3f})')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.3f})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 按类别分组 - 原始数据
        plt.subplot(1, 3, 2)
        for label in np.unique(self.original_labels):
            mask = self.original_labels == label
            plt.scatter(X_orig_pca[mask, 0], X_orig_pca[mask, 1], 
                       label=f'Original Class {label}', alpha=0.7, s=50)
        plt.title('PCA: Original Data by Class')
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.3f})')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.3f})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 按类别分组 - 增强数据
        plt.subplot(1, 3, 3)
        for label in np.unique(self.augmented_labels):
            mask = self.augmented_labels == label
            plt.scatter(X_aug_pca[mask, 0], X_aug_pca[mask, 1], 
                       label=f'Augmented Class {label}', alpha=0.7, s=50)
        plt.title('PCA: Augmented Data by Class')
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.3f})')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.3f})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'pca_visualization.png'), dpi=300)
        plt.close()
    
    def _plot_feature_distributions(self):
        """特征分布对比"""
        # 计算每个序列的统计特征
        orig_means = np.mean(self.original_data, axis=1)  # [samples, features]
        aug_means = np.mean(self.augmented_data, axis=1)
        
        orig_stds = np.std(self.original_data, axis=1)
        aug_stds = np.std(self.augmented_data, axis=1)
        
        # 绘制前3个特征的分布
        n_features = min(3, orig_means.shape[1])
        
        plt.figure(figsize=(15, 10))
        
        for i in range(n_features):
            # 均值分布
            plt.subplot(2, n_features, i + 1)
            plt.hist(orig_means[:, i], bins=20, alpha=0.7, label='Original', density=True)
            plt.hist(aug_means[:, i], bins=20, alpha=0.7, label='Augmented', density=True)
            plt.title(f'Feature {i+1} - Mean Distribution')
            plt.xlabel('Mean Value')
            plt.ylabel('Density')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # 标准差分布
            plt.subplot(2, n_features, i + 1 + n_features)
            plt.hist(orig_stds[:, i], bins=20, alpha=0.7, label='Original', density=True)
            plt.hist(aug_stds[:, i], bins=20, alpha=0.7, label='Augmented', density=True)
            plt.title(f'Feature {i+1} - Std Distribution')
            plt.xlabel('Std Value')
            plt.ylabel('Density')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'feature_distributions.png'), dpi=300)
        plt.close()
    
    def _plot_classification_results(self):
        """可视化分类结果"""
        if 'classification' not in self.results:
            return
            
        metrics = ['accuracy', 'f1', 'recall', 'precision']
        data_types = ['original', 'augmented', 'combined', 'cross_validation']
        
        # 准备数据
        plot_data = {metric: [] for metric in metrics}
        labels = []
        
        for data_type in data_types:
            if data_type in self.results['classification']:
                labels.append(data_type.replace('_', ' ').title())
                for metric in metrics:
                    plot_data[metric].append(self.results['classification'][data_type][metric])
        
        # 绘图
        x = np.arange(len(labels))
        width = 0.2
        
        plt.figure(figsize=(12, 6))
        for i, metric in enumerate(metrics):
            plt.bar(x + i * width, plot_data[metric], width, label=metric.title())
        
        plt.xlabel('Data Type')
        plt.ylabel('Score')
        plt.title('Classification Performance Comparison')
        plt.xticks(x + width * 1.5, labels)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)
        
        # 添加数值标签
        for i, metric in enumerate(metrics):
            for j, value in enumerate(plot_data[metric]):
                plt.text(j + i * width, value + 0.01, f'{value:.3f}', 
                        ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'classification_comparison.png'), dpi=300)
        plt.close()
    
    def generate_report(self):
        """生成评估报告"""
        print("\n6. Generating Evaluation Report...")
        
        report_path = os.path.join(self.output_dir, 'evaluation_report.txt')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("数据增强质量评估报告\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"原始数据: {self.original_data.shape}\n")
            f.write(f"增强数据: {self.augmented_data.shape}\n\n")
            
            # 分类性能
            if 'classification' in self.results:
                f.write("1. 分类性能评估\n")
                f.write("-" * 30 + "\n")
                for data_type, metrics in self.results['classification'].items():
                    f.write(f"{data_type.title()}:\n")
                    for metric, value in metrics.items():
                        f.write(f"  {metric}: {value:.4f}\n")
                    f.write("\n")
            
            # 判别能力
            if 'discriminator' in self.results:
                f.write("2. 判别能力评估\n")
                f.write("-" * 30 + "\n")
                f.write(f"AUC: {self.results['discriminator']['auc']:.4f}\n")
                f.write(f"质量分数: {self.results['discriminator']['quality_score']:.4f}\n")
                f.write("注: AUC越接近0.5，质量分数越低，说明生成数据越真实\n\n")
            
            # 分布相似性
            if 'distribution' in self.results:
                f.write("3. 分布相似性评估\n")
                f.write("-" * 30 + "\n")
                f.write(f"FID分数: {self.results['distribution']['fid']:.4f}\n")
                for metric, value in self.results['distribution']['statistical_distances'].items():
                    f.write(f"{metric}: {value:.4f}\n")
                f.write("注: 分数越低说明分布越相似\n\n")
            
            # 时间结构
            if 'temporal' in self.results:
                f.write("4. 时间结构评估\n")
                f.write("-" * 30 + "\n")
                for metric, value in self.results['temporal'].items():
                    f.write(f"{metric}: {value:.4f}\n")
                f.write("注: ACF和FFT相似性越高越好，DTW距离越低越好\n\n")
            
            f.write("=" * 60 + "\n")
            f.write("评估完成！详细可视化结果请查看生成的图片文件。\n")
        
        print(f"Evaluation report saved to {report_path}")
    
    def run_full_evaluation(self):
        """运行完整评估"""
        print("Starting comprehensive data augmentation evaluation...")
        
        # 加载数据
        self.load_data()
        
        if len(self.original_data) == 0 or len(self.augmented_data) == 0:
            print("No data found! Please check your data directories.")
            return
        
        # 运行各项评估
        self.evaluate_classification_performance()
        self.evaluate_discriminator_ability()
        self.evaluate_distribution_similarity()
        self.evaluate_temporal_structure()
        self.visualize_distributions()
        
        # 生成报告
        self.generate_report()
        
        print(f"\nEvaluation completed! Results saved to {self.output_dir}")

def main():
    """主函数"""
    evaluator = AugmentationEvaluator(
        original_dir='original',
        augmented_dir='augmented_data',
        output_dir='evaluation_results'
    )
    
    evaluator.run_full_evaluation()

if __name__ == '__main__':
    main()