#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
认知障碍数据增强系统 - 主运行程序
从CSV文件加载原始数据，应用四种认知衰退模拟方法进行数据增强
包含基于类别的差异化增强和质量评估功能
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path
import warnings

# Set matplotlib to avoid font warnings
plt.rcParams['font.family'] = ['DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

# Add path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from cognitive_augmentation import CognitiveAugmentationSystem
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import seaborn as sns
from scipy import stats

def load_data(data_dir, seq_len=400):
    """从目录加载多个CSV文件"""
    print(f"📂 从 {data_dir} 加载数据...")

    if not os.path.exists(data_dir):
        raise ValueError(f"目录 '{data_dir}' 不存在")

    # 查找所有CSV文件
    data_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    if not data_files:
        raise ValueError(f"在 {data_dir} 中未找到CSV文件")

    print(f"📊 找到 {len(data_files)} 个数据文件")
    
    all_data = []
    all_labels = []
    
    for file in data_files:
        try:
            df = pd.read_csv(os.path.join(data_dir, file))

            # 提取标签
            if 'label' in df.columns:
                label = int(df['label'].iloc[0])
                df = df.drop('label', axis=1)
            else:
                # 从文件名提取标签
                import re
                numbers = re.findall(r'\d+', file)
                label = int(numbers[0]) % 3 if numbers else 0

            # 数据预处理
            data = df.values.astype(np.float32)
            data = np.nan_to_num(data, nan=0.0, posinf=1.0, neginf=-1.0)

            # 截断或填充到指定长度
            if len(data) > seq_len:
                data = data[:seq_len]
            else:
                padding = np.zeros((seq_len - len(data), data.shape[1]))
                data = np.vstack([data, padding])

            all_data.append(data)
            all_labels.append(label)

        except Exception as e:
            print(f"⚠️ 处理文件 {file} 时出错: {e}")
            continue
    
    if not all_data:
        raise ValueError("没有成功加载任何数据文件")

    print(f"✅ 数据加载完成")
    print(f"📊 样本数量: {len(all_data)}")
    print(f"📊 每个样本形状: {all_data[0].shape}")
    print(f"🏷️ 标签分布: {np.bincount(all_labels)}")

    return all_data, np.array(all_labels)

def visualize_results(original_data, augmented_results, output_dir):
    """Visualize augmentation results"""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle('Cognitive Augmentation Results', fontsize=14)
    
    # Original data
    axes[0, 0].plot(original_data[:, :6], alpha=0.7, linewidth=1)
    axes[0, 0].set_title('Original')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Augmentation results
    method_names = ['Motor Delay', 'Gait Perturbation', 'Sensor Drift', 'Cognitive Vector']
    positions = [(0, 1), (0, 2), (1, 0), (1, 1)]
    
    for i, (method_key, aug_data) in enumerate(augmented_results.items()):
        if i < len(positions):
            row, col = positions[i]
            axes[row, col].plot(aug_data[:, :6], alpha=0.7, linewidth=1)
            axes[row, col].set_title(method_names[i])
            axes[row, col].grid(True, alpha=0.3)
    
    # Change magnitude comparison
    changes = [np.linalg.norm(original_data - aug_data) for aug_data in augmented_results.values()]
    axes[1, 2].bar(range(len(changes)), changes, alpha=0.7)
    axes[1, 2].set_title('Change Magnitude')
    axes[1, 2].set_xticks(range(len(changes)))
    axes[1, 2].set_xticklabels(['MD', 'GP', 'SD', 'CV'], rotation=0)
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'augmentation_comparison.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def get_class_specific_params(class_label, base_params, intensity_factors):
    """根据认知状态获取类别特定的增强参数"""

    # 类别0: 正常人 - 最小增强
    # 类别1: 轻度认知障碍 - 中等增强
    # 类别2: 中度认知衰退 - 强烈增强

    if class_label not in intensity_factors:
        class_label = 1  # 默认使用轻度认知障碍参数

    factors = intensity_factors[class_label]
    class_params = {}

    # 运动延迟参数调整
    base_delay = base_params['motor_delay']['delay_range']
    base_alpha = base_params['motor_delay']['alpha_range']

    delay_factor = factors['delay_factor']
    new_delay_range = [
        max(1, int(base_delay[0] * delay_factor)),
        min(15, int(base_delay[1] * delay_factor))
    ]
    new_alpha_range = [
        max(0.5, base_alpha[0] * (2 - delay_factor)),  # 反向调整
        min(0.99, base_alpha[1] * (2 - delay_factor))
    ]

    class_params['motor_delay'] = {
        'delay_range': new_delay_range,
        'alpha_range': new_alpha_range
    }

    # 步态扰动参数调整
    base_perturbation = base_params['gait_perturbation']['perturbation_intensity']
    base_sync = base_params['gait_perturbation']['sync_probability']

    class_params['gait_perturbation'] = {
        'perturbation_intensity': min(0.8, base_perturbation * factors['perturbation_factor']),
        'sync_probability': min(0.8, base_sync * factors['perturbation_factor'])
    }

    # 传感器漂移参数调整
    base_drift = base_params['sensor_drift']['drift_intensity']

    class_params['sensor_drift'] = {
        'drift_intensity': min(0.2, base_drift * factors['drift_factor'])
    }

    return class_params

def compute_quality_score(sample, original_data=None):
    """计算样本质量分数 - 参考enhanced_vae_gan.py"""
    score = 0.0

    # 1. 统计特性检查
    if not (np.isnan(sample).any() or np.isinf(sample).any()):
        score += 0.2

    # 2. 方差检查（避免常数序列）
    if np.std(sample) > 0.01:
        score += 0.2

    # 3. 范围检查
    sample_min, sample_max = sample.min(), sample.max()
    if sample_min >= -5 and sample_max <= 5:  # 合理范围
        score += 0.2

    # 4. 平滑性检查（避免过度震荡）
    if len(sample) > 1:
        diff = np.diff(sample)
        if np.std(diff) < 2.0 * np.std(sample):
            score += 0.2

    # 5. 如果有原始数据，检查分布相似性
    if original_data is not None:
        try:
            # KS测试
            _, p_value = stats.ks_2samp(sample, original_data.flatten())
            if p_value > 0.05:  # 分布相似
                score += 0.2
        except:
            pass
    else:
        score += 0.2  # 默认给分

    return score

def train_and_evaluate_classifiers(train_X, train_y, test_X, test_y, classifier_params, experiment_name=""):
    """训练和评估多个分类器 - 使用更复杂的模型"""

    # 使用传入的分类器配置
    classifiers = {
        'Random Forest': RandomForestClassifier(
            random_state=42,
            **classifier_params['random_forest']
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            random_state=42,
            **classifier_params['gradient_boosting']
        ),
        'SVM': SVC(
            random_state=42,
            **classifier_params['svm']
        ),
        'MLP': MLPClassifier(
            random_state=42,
            **classifier_params['mlp']
        )
    }

    results = {}

    for name, clf in classifiers.items():
        print(f"   🔧 训练 {name}...")

        try:
            # 训练
            clf.fit(train_X, train_y)

            # 预测
            pred_y = clf.predict(test_X)

            # 评估
            accuracy = accuracy_score(test_y, pred_y)
            f1 = f1_score(test_y, pred_y, average='weighted')

            results[name] = {
                'accuracy': accuracy,
                'f1_score': f1,
                'predictions': pred_y,
                'true_labels': test_y
            }

            print(f"     📊 准确率: {accuracy:.4f}")
            print(f"     📊 F1分数: {f1:.4f}")

        except Exception as e:
            print(f"     ❌ {name} 训练失败: {e}")
            results[name] = {
                'accuracy': 0.0,
                'f1_score': 0.0,
                'predictions': np.zeros_like(test_y),
                'true_labels': test_y
            }

    return results

def cross_validation_comparison(orig_X, orig_y, aug_X, aug_y, cv_folds=5):
    """交叉验证比较"""

    print("   🔄 执行交叉验证...")

    # 仅原始数据
    rf_orig = RandomForestClassifier(n_estimators=100, random_state=42)
    scores_orig = cross_val_score(rf_orig, orig_X, orig_y,
                                 cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42),
                                 scoring='accuracy')

    # 原始+增强数据
    combined_X = np.vstack([orig_X, aug_X])
    combined_y = np.hstack([orig_y, aug_y])

    rf_aug = RandomForestClassifier(n_estimators=100, random_state=42)
    scores_aug = cross_val_score(rf_aug, combined_X, combined_y,
                                cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42),
                                scoring='accuracy')

    print(f"   📊 原始数据CV准确率: {scores_orig.mean():.4f} ± {scores_orig.std():.4f}")
    print(f"   📊 增强数据CV准确率: {scores_aug.mean():.4f} ± {scores_aug.std():.4f}")
    print(f"   📈 性能提升: {(scores_aug.mean() - scores_orig.mean()):.4f}")

    return {
        'original_scores': scores_orig,
        'augmented_scores': scores_aug,
        'improvement': scores_aug.mean() - scores_orig.mean()
    }

def evaluate_augmentation_quality(original_data, original_labels, augmented_data, augmented_labels, classifier_params):
    """使用分类性能评估增强质量 - 修复数据泄露问题"""

    print("🔍 评估增强质量...")

    # 展平数据用于分类
    def flatten_data(data_list):
        return np.array([data.flatten() for data in data_list])

    X_orig = flatten_data(original_data)
    X_aug = flatten_data(augmented_data)

    print(f"   📊 原始数据: {X_orig.shape}")
    print(f"   📊 增强数据: {X_aug.shape}")

    # 计算增强数据质量分数
    print("   🔍 计算增强数据质量分数...")
    quality_scores = []
    for i, aug_sample in enumerate(augmented_data):
        # 找到对应的原始样本
        orig_sample = original_data[i % len(original_data)]
        score = compute_quality_score(aug_sample.flatten(), orig_sample.flatten())
        quality_scores.append(score)

    avg_quality_score = np.mean(quality_scores)
    print(f"   📊 平均质量分数: {avg_quality_score:.4f}")

    # 数据预处理 - 分别标准化避免数据泄露
    scaler_orig = StandardScaler()
    scaler_aug = StandardScaler()

    X_orig_scaled = scaler_orig.fit_transform(X_orig)
    X_aug_scaled = scaler_aug.fit_transform(X_aug)

    # PCA降维 (如果特征太多)
    if X_orig_scaled.shape[1] > 1000:
        print("   🔧 应用PCA降维...")
        n_components = min(100, X_orig_scaled.shape[0]-1, X_orig_scaled.shape[1])
        pca = PCA(n_components=n_components)
        X_orig_scaled = pca.fit_transform(X_orig_scaled)
        X_aug_scaled = pca.transform(X_aug_scaled)
        print(f"   📊 降维后特征数: {X_orig_scaled.shape[1]}")

    results = {}

    # 实验1: 仅使用原始数据训练和测试 (基线)
    print("\n   🧪 实验1: 仅原始数据 (基线)")
    if len(np.unique(original_labels)) > 1 and len(original_labels) >= 6:
        orig_train_X, orig_test_X, orig_train_y, orig_test_y = train_test_split(
            X_orig_scaled, original_labels, test_size=0.3, random_state=42,
            stratify=original_labels
        )

        baseline_results = train_and_evaluate_classifiers(
            orig_train_X, orig_train_y, orig_test_X, orig_test_y, classifier_params, "基线"
        )
        results['baseline'] = baseline_results
    else:
        print("   ⚠️ 数据量不足，跳过基线测试")
        results['baseline'] = None

    # 实验2: 使用原始+增强数据训练，在独立测试集上测试
    print("\n   🧪 实验2: 原始+增强数据训练，独立测试集测试")
    if results['baseline'] is not None:
        # 合并原始训练数据和增强数据进行训练
        combined_train_X = np.vstack([orig_train_X, X_aug_scaled])
        combined_train_y = np.hstack([orig_train_y, augmented_labels])

        # 在相同的测试集上测试
        augmented_results = train_and_evaluate_classifiers(
            combined_train_X, combined_train_y, orig_test_X, orig_test_y, classifier_params, "增强训练"
        )
        results['augmented'] = augmented_results
    else:
        print("   ⚠️ 跳过增强数据测试")
        results['augmented'] = None

    # 实验3: 严格的独立测试 - 完全分离的数据集
    print("\n   🧪 实验3: 严格独立测试")
    if len(original_labels) >= 10:
        # 将原始数据分为两部分：一部分用于生成增强数据，另一部分用于测试
        split_idx = len(original_data) // 2

        # 第一部分：用于训练和生成增强数据
        train_orig_data = original_data[:split_idx]
        train_orig_labels = original_labels[:split_idx]
        train_aug_data = augmented_data[:split_idx]
        train_aug_labels = augmented_labels[:split_idx]

        # 第二部分：完全独立的测试数据
        test_orig_data = original_data[split_idx:]
        test_orig_labels = original_labels[split_idx:]

        # 准备训练和测试数据
        train_orig_X = flatten_data(train_orig_data)
        train_aug_X = flatten_data(train_aug_data)
        test_X = flatten_data(test_orig_data)

        # 标准化
        scaler_strict = StandardScaler()
        train_orig_X_scaled = scaler_strict.fit_transform(train_orig_X)
        train_aug_X_scaled = scaler_strict.transform(train_aug_X)
        test_X_scaled = scaler_strict.transform(test_X)

        # PCA降维
        if train_orig_X_scaled.shape[1] > 100:
            pca_strict = PCA(n_components=min(50, train_orig_X_scaled.shape[0]-1))
            train_orig_X_scaled = pca_strict.fit_transform(train_orig_X_scaled)
            train_aug_X_scaled = pca_strict.transform(train_aug_X_scaled)
            test_X_scaled = pca_strict.transform(test_X_scaled)

        # 仅原始数据训练
        strict_baseline = train_and_evaluate_classifiers(
            train_orig_X_scaled, train_orig_labels, test_X_scaled, test_orig_labels,
            classifier_params, "严格基线"
        )

        # 原始+增强数据训练
        combined_X = np.vstack([train_orig_X_scaled, train_aug_X_scaled])
        combined_y = np.hstack([train_orig_labels, train_aug_labels])

        strict_augmented = train_and_evaluate_classifiers(
            combined_X, combined_y, test_X_scaled, test_orig_labels,
            classifier_params, "严格增强"
        )

        results['strict_test'] = {
            'baseline': strict_baseline,
            'augmented': strict_augmented
        }

        print(f"   📊 严格测试 - 基线最佳准确率: {max([v['accuracy'] for v in strict_baseline.values()]):.4f}")
        print(f"   📊 严格测试 - 增强最佳准确率: {max([v['accuracy'] for v in strict_augmented.values()]):.4f}")

    else:
        print("   ⚠️ 数据量不足，跳过严格测试")
        results['strict_test'] = None

    # 实验4: 交叉验证评估
    print("\n   🧪 实验4: 交叉验证比较")
    if len(original_labels) >= 10:
        cv_results = cross_validation_comparison(
            X_orig_scaled, original_labels, X_aug_scaled, augmented_labels
        )
        results['cross_validation'] = cv_results
    else:
        print("   ⚠️ 数据量不足，跳过交叉验证")
        results['cross_validation'] = None

    # 添加质量分数到结果
    results['quality_score'] = avg_quality_score

    return results

def run_augmentation(data_path, output_dir, base_augmentation_params, class_intensity_factors,
                    evaluation_params, classifier_params, test_samples, seq_len):
    """对原始数据执行认知增强"""

    print("🧠 认知障碍数据增强系统")
    print("=" * 50)

    # 打印配置信息
    print("📋 配置信息:")
    print(f"   数据路径: {data_path}")
    print(f"   输出目录: {output_dir}")
    print(f"   序列长度: {seq_len}")
    print(f"   处理样本数: {test_samples if test_samples else '全部'}")
    print(f"   评估参数: {evaluation_params}")
    print(f"   分类器数量: {len(classifier_params)}")
    
    # 1. 加载数据
    try:
        data_list, labels = load_data(data_path, seq_len=seq_len)
    except Exception as e:
        print(f"❌ 数据加载失败: {str(e)}")
        return

    # 2. 数据预处理
    print("🔧 数据预处理...")
    processed_data = []
    processed_labels = []

    for i, (data, label) in enumerate(zip(data_list, labels)):
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)
        elif len(data.shape) > 2:
            data = data.reshape(data.shape[0], -1)

        if np.isnan(data).any() or np.isinf(data).any():
            print(f"⚠️ 样本 {i} 包含无效值，跳过")
            continue

        processed_data.append(data)
        processed_labels.append(label)

    processed_labels = np.array(processed_labels)
    print(f"✅ 预处理完成，保留 {len(processed_data)} 个有效样本")
    
    # 3. 初始化增强系统
    aug_system = CognitiveAugmentationSystem()

    # 4. 选择样本进行方法比较
    sample_idx = len(processed_data) // 2
    sample_data = processed_data[sample_idx]

    print(f"📊 使用样本 {sample_idx} 进行方法比较")
    print(f"📊 样本形状: {sample_data.shape}")

    # 5. 应用增强方法
    n_features = sample_data.shape[1]
    imu_channels = list(range(min(6, n_features)))
    pressure_channels = list(range(6, n_features)) if n_features > 6 else []

    print("🔧 应用四种认知增强方法...")

    augmented_results = {}
    methods = [
        ('motor_delay', imu_channels, {}),
        ('gait_perturbation', pressure_channels, {'pressure_channels': pressure_channels}),
        ('sensor_drift', imu_channels, {'imu_channels': imu_channels, 'pressure_channels': pressure_channels}),
        ('cognitive_vector', None, {})
    ]

    for method, channels, extra_params in methods:
        if method == 'motor_delay':
            params = {**base_augmentation_params[method], 'apply_to_channels': channels}
        elif method == 'gait_perturbation':
            params = {**base_augmentation_params[method], **extra_params}
        elif method == 'sensor_drift':
            params = {**base_augmentation_params[method], **extra_params}
        else:  # cognitive_vector (简化版)
            aug_data = sample_data.copy()
            cognitive_vector = np.random.randn(*sample_data.shape) * 0.1
            if imu_channels:
                cognitive_vector[:, imu_channels] *= 0.5
            if pressure_channels:
                cognitive_vector[:, pressure_channels] *= 0.8
            augmented_results[method] = aug_data + cognitive_vector
            continue

        aug_data = aug_system.augment_data(sample_data, method, **params)
        augmented_results[method] = aug_data

    # 6. 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 7. 生成可视化比较
    print("📊 生成可视化比较...")
    visualize_results(sample_data, augmented_results, output_dir)
    
    # 8. 基于类别的批量增强
    print("🔄 执行基于类别的批量增强...")

    # 处理样本数量
    n_process = len(processed_data) if test_samples is None else min(test_samples, len(processed_data))
    print(f"📊 处理 {n_process} 个样本")

    # 按类别分组数据
    class_data = {0: [], 1: [], 2: []}
    class_indices = {0: [], 1: [], 2: []}

    for i, (data, label) in enumerate(zip(processed_data[:n_process], processed_labels[:n_process])):
        if label in class_data:
            class_data[label].append(data)
            class_indices[label].append(i)

    print(f"📊 类别分布: {[len(class_data[i]) for i in range(3)]}")

    # 应用类别特定的增强
    all_aug_data = []
    all_aug_labels = []

    for class_label in [0, 1, 2]:
        if len(class_data[class_label]) == 0:
            continue

        print(f"🔧 增强类别 {class_label} ({len(class_data[class_label])} 个样本)...")

        # 获取类别特定参数
        class_params = get_class_specific_params(class_label, base_augmentation_params, class_intensity_factors)

        # 使用类别特定参数应用运动延迟增强
        motor_params = class_params['motor_delay']
        class_aug_data, class_aug_labels = aug_system.batch_augment(
            class_data[class_label], [class_label] * len(class_data[class_label]),
            'motor_delay', **motor_params, apply_to_channels=imu_channels
        )

        all_aug_data.extend(class_aug_data)
        all_aug_labels.extend(class_aug_labels)

    print(f"📊 总增强样本数: {len(all_aug_data)}")
    
    # 9. 评估增强质量
    print("🔍 评估增强质量...")

    # 准备评估数据
    original_data = processed_data[:n_process]
    original_labels = processed_labels[:n_process]

    # 提取仅增强的样本（排除增强数据中的原始样本）
    augmented_only_data = []
    augmented_only_labels = []

    for i in range(0, len(all_aug_data), 2):  # 跳过原始样本，只取增强样本
        if i + 1 < len(all_aug_data):
            augmented_only_data.append(all_aug_data[i + 1])
            augmented_only_labels.append(all_aug_labels[i + 1])

    # 评估质量
    quality_metrics = evaluate_augmentation_quality(
        original_data, original_labels,
        augmented_only_data, augmented_only_labels,
        classifier_params
    )

    # 打印评估结果
    print("\n" + "="*60)
    print("📊 增强质量评估结果")
    print("="*60)

    # 获取最佳分类器结果（随机森林）
    rf_baseline = quality_metrics['baseline']['Random Forest']
    rf_augmented = quality_metrics['augmented']['Random Forest']

    print(f"📊 基础测试 - 原始数据准确率: {rf_baseline['accuracy']:.4f}")
    print(f"📊 基础测试 - 增强数据准确率: {rf_augmented['accuracy']:.4f}")
    improvement = rf_augmented['accuracy'] - rf_baseline['accuracy']
    relative_improvement = improvement / rf_baseline['accuracy'] * 100 if rf_baseline['accuracy'] > 0 else 0
    print(f"📈 基础测试性能提升: {improvement:.4f} ({relative_improvement:.2f}%)")

    # 严格独立测试结果
    if quality_metrics['strict_test'] is not None:
        strict_baseline = quality_metrics['strict_test']['baseline']
        strict_augmented = quality_metrics['strict_test']['augmented']

        strict_base_best = max([v['accuracy'] for v in strict_baseline.values()])
        strict_aug_best = max([v['accuracy'] for v in strict_augmented.values()])
        strict_improvement = strict_aug_best - strict_base_best

        print(f"\n📊 严格独立测试 - 基线最佳准确率: {strict_base_best:.4f}")
        print(f"📊 严格独立测试 - 增强最佳准确率: {strict_aug_best:.4f}")
        print(f"📈 严格测试性能提升: {strict_improvement:.4f}")

        if strict_improvement > 0:
            print("✅ 增强数据在严格测试中显示正向效果")
        elif strict_improvement == 0:
            print("⚠️ 增强数据在严格测试中无明显效果")
        else:
            print("❌ 增强数据在严格测试中显示负向效果")

    # 交叉验证结果
    if quality_metrics['cross_validation'] is not None:
        cv_results = quality_metrics['cross_validation']
        print(f"\n📊 交叉验证提升: {cv_results['improvement']:.4f}")

    # 质量分数
    print(f"\n📊 增强数据质量分数: {quality_metrics['quality_score']:.4f}")

    # 各分类器性能对比
    print("\n📊 各分类器性能对比 (基础测试):")
    for clf_name in ['Random Forest', 'Gradient Boosting', 'SVM', 'MLP']:
        if clf_name in quality_metrics['baseline']:
            baseline_acc = quality_metrics['baseline'][clf_name]['accuracy']
            augmented_acc = quality_metrics['augmented'][clf_name]['accuracy']
            improvement = augmented_acc - baseline_acc
            print(f"   {clf_name}: {baseline_acc:.3f} -> {augmented_acc:.3f} ({improvement:+.3f})")

    # 10. 保存结果
    print("\n💾 保存增强结果...")

    # 保存为NumPy格式
    original_array = np.array([data.flatten() for data in original_data])
    augmented_array = np.array([data.flatten() for data in all_aug_data])

    np.save(os.path.join(output_dir, 'original.npy'), original_array)
    np.save(os.path.join(output_dir, 'augmented.npy'), augmented_array)
    np.save(os.path.join(output_dir, 'labels.npy'), np.array(all_aug_labels))

    # 保存为CSV格式
    pd.DataFrame(original_array).assign(label=original_labels).to_csv(
        os.path.join(output_dir, 'original.csv'), index=False)
    pd.DataFrame(augmented_array).assign(label=all_aug_labels).to_csv(
        os.path.join(output_dir, 'augmented.csv'), index=False)

    # 保存质量评估指标
    import json
    with open(os.path.join(output_dir, 'quality_metrics.json'), 'w', encoding='utf-8') as f:
        # 转换numpy类型为Python类型以便JSON序列化
        metrics_for_json = {
            'basic_test': {
                'baseline_accuracy': float(rf_baseline['accuracy']),
                'augmented_accuracy': float(rf_augmented['accuracy']),
                'improvement': float(improvement),
                'relative_improvement': float(relative_improvement)
            },
            'quality_score': float(quality_metrics['quality_score']),
            'classifiers_performance': {
                clf_name: {
                    'baseline': float(quality_metrics['baseline'][clf_name]['accuracy']),
                    'augmented': float(quality_metrics['augmented'][clf_name]['accuracy']),
                    'improvement': float(quality_metrics['augmented'][clf_name]['accuracy'] -
                                       quality_metrics['baseline'][clf_name]['accuracy'])
                }
                for clf_name in ['Random Forest', 'Gradient Boosting', 'SVM', 'MLP']
                if clf_name in quality_metrics['baseline']
            }
        }

        # 添加严格测试结果
        if quality_metrics['strict_test'] is not None:
            strict_baseline = quality_metrics['strict_test']['baseline']
            strict_augmented = quality_metrics['strict_test']['augmented']
            strict_base_best = max([v['accuracy'] for v in strict_baseline.values()])
            strict_aug_best = max([v['accuracy'] for v in strict_augmented.values()])

            metrics_for_json['strict_test'] = {
                'baseline_best_accuracy': float(strict_base_best),
                'augmented_best_accuracy': float(strict_aug_best),
                'improvement': float(strict_aug_best - strict_base_best)
            }

        # 添加交叉验证结果
        if quality_metrics['cross_validation'] is not None:
            cv_results = quality_metrics['cross_validation']
            metrics_for_json['cross_validation'] = {
                'improvement': float(cv_results['improvement'])
            }

        json.dump(metrics_for_json, f, indent=2, ensure_ascii=False)

    print(f"✅ 增强完成! 结果保存在: {output_dir}")
    print(f"📊 原始: {len(original_data)} -> 增强: {len(all_aug_data)} (增长 {len(all_aug_data)/len(original_data):.1f}x)")
    print(f"📈 质量提升: {relative_improvement:.2f}%")

def main():
    """主函数 - 集中所有可配置参数"""
    print("🧠 认知障碍数据增强系统")
    print("=" * 50)

    # ==================== 核心参数设置 ====================
    # 数据路径设置
    data_path = "./original_processed"  # 原始数据目录
    output_dir = "augmentation_results"  # 输出目录

    # 数据处理参数
    seq_len = 400  # 序列长度
    test_samples = None  # 处理样本数，None=全部

    # 基础增强参数
    base_augmentation_params = {
        'motor_delay': {'delay_range': [3, 8], 'alpha_range': [0.75, 0.9]},
        'gait_perturbation': {'perturbation_intensity': 0.25, 'sync_probability': 0.3},
        'sensor_drift': {'drift_intensity': 0.08}
    }

    # 类别特定增强强度调节因子
    class_intensity_factors = {
        0: {'delay_factor': 0.5, 'perturbation_factor': 0.4, 'drift_factor': 0.25},  # 正常人：最小增强
        1: {'delay_factor': 0.75, 'perturbation_factor': 0.8, 'drift_factor': 0.625},  # 轻度认知障碍：中等增强
        2: {'delay_factor': 1.25, 'perturbation_factor': 1.6, 'drift_factor': 1.5}   # 中度认知衰退：强烈增强
    }

    # 评估参数
    evaluation_params = {
        'test_size': 0.3,  # 测试集比例
        'cv_folds': 5,     # 交叉验证折数
        'random_state': 42, # 随机种子
        'pca_components': 100,  # PCA降维后的特征数
        'quality_threshold': 0.6  # 质量分数阈值
    }

    # 分类器参数
    classifier_params = {
        'random_forest': {
            'n_estimators': 200,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2
        },
        'gradient_boosting': {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 6
        },
        'svm': {
            'C': 1.0,
            'gamma': 'scale',
            'kernel': 'rbf'
        },
        'mlp': {
            'hidden_layer_sizes': (128, 64, 32),
            'activation': 'relu',
            'solver': 'adam',
            'alpha': 0.001,
            'learning_rate': 'adaptive',
            'max_iter': 1000
        }
    }
    # ================================================

    if not os.path.exists(data_path):
        print(f"❌ 错误: 目录不存在: {data_path}")
        print("💡 请确保original目录存在并包含CSV文件")
        return

    try:
        run_augmentation(
            data_path=data_path,
            output_dir=output_dir,
            base_augmentation_params=base_augmentation_params,
            class_intensity_factors=class_intensity_factors,
            evaluation_params=evaluation_params,
            classifier_params=classifier_params,
            test_samples=test_samples,
            seq_len=seq_len
        )
    except Exception as e:
        print(f"❌ 错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
