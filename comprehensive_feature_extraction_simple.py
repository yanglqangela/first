# 简化版全面特征提取系统
# 只使用前400列数据，减少终端输出

import os
import numpy as np
import pandas as pd
from scipy import signal, integrate
from scipy.spatial import ConvexHull
from scipy.stats import pearsonr
import glob
import warnings
warnings.filterwarnings('ignore')

def compute_velocity(acc, fs):
    """计算速度"""
    velocity = np.zeros_like(acc)
    for i in range(acc.shape[1]):
        acc_corrected = acc[:, i] - np.mean(acc[:, i])
        vel = integrate.cumtrapz(acc_corrected, dx=1/fs, initial=0)
        if not np.all(np.isfinite(vel)):
            vel = np.nan_to_num(vel, nan=0.0, posinf=0.0, neginf=0.0)
        vel = signal.detrend(vel)
        velocity[:, i] = vel
    return velocity

def compute_displacement(velocity, fs):
    """计算位移"""
    displacement = np.zeros_like(velocity)
    for i in range(velocity.shape[1]):
        disp = integrate.cumtrapz(velocity[:, i], dx=1/fs, initial=0)
        if not np.all(np.isfinite(disp)):
            disp = np.nan_to_num(disp, nan=0.0, posinf=0.0, neginf=0.0)
        disp = signal.detrend(disp)
        displacement[:, i] = disp
    return displacement

def detect_data_format(df_segment):
    """检测数据格式"""
    first_col = str(df_segment.columns[0])
    return 'augmented' if first_col.isdigit() else 'original'

def clean_zero_values(df):
    """清理连续的0值，用小的随机值替代"""
    df_clean = df.copy()

    # 排除标签列
    data_cols = [col for col in df.columns if col != 'label']

    for col in data_cols:
        if df_clean[col].dtype in ['float64', 'int64']:
            # 检测连续的0值
            zero_mask = (df_clean[col] == 0)

            # 如果有连续的0值，用小的随机噪声替代
            if zero_mask.sum() > len(df_clean) * 0.1:  # 如果0值超过10%
                # 计算非零值的标准差
                non_zero_values = df_clean[col][~zero_mask]
                if len(non_zero_values) > 0:
                    noise_std = non_zero_values.std() * 0.001  # 1‰的噪声
                    if noise_std > 0:
                        # 用小的随机噪声替代0值
                        noise = np.random.normal(0, noise_std, zero_mask.sum())
                        df_clean.loc[zero_mask, col] = noise

    return df_clean

def calculate_target_segments(data_length, window_size, stride):
    """计算目标段数，固定提取8次"""
    return 8

def identify_sensor_columns(df_segment, data_format):
    """识别传感器列"""
    if data_format == 'augmented':
        # 增强数据格式
        pressure_cols = df_segment.columns[90:106]
        left_pressure_cols = pressure_cols[:8]
        right_pressure_cols = pressure_cols[8:16]
        waist_acc_cols = [df_segment.columns[26], df_segment.columns[27], df_segment.columns[28]]
        waist_gyro_cols = [df_segment.columns[29], df_segment.columns[30], df_segment.columns[31]]
    else:
        # 原始数据格式 - 根据实际列名结构识别
        # 左脚压力传感器：左脚1号压力点数据 + 2-8号压力点数据（不带.1后缀）
        left_pressure_cols = []
        for col in df_segment.columns:
            if '左脚1号压力点数据' in col:
                left_pressure_cols.append(col)
            elif '压力点数据' in col and '压力点数据.' not in col and '左脚' not in col and '右脚' not in col:
                # 2-8号压力点数据（不带后缀的是左脚）
                left_pressure_cols.append(col)
        
        # 右脚压力传感器：右脚1号压力点数据 + 2-8号压力点数据.1
        right_pressure_cols = []
        for col in df_segment.columns:
            if '右脚1号压力点数据' in col:
                right_pressure_cols.append(col)
            elif '压力点数据.1' in col:
                # 2-8号压力点数据.1（带.1后缀的是右脚）
                right_pressure_cols.append(col)
        
        # 腰部传感器（节点3）- 加速度
        waist_acc_cols = []
        for col in df_segment.columns:
            if '节点3的x轴加速度' in col:
                waist_acc_cols.append(col)
            elif 'y轴加速度.2' in col:  # 节点3的y轴加速度
                waist_acc_cols.append(col)
            elif 'z轴加速度.2' in col:  # 节点3的z轴加速度
                waist_acc_cols.append(col)
        
        # 腰部传感器（节点3）- 角速度
        waist_gyro_cols = []
        for col in df_segment.columns:
            if 'x轴角速度.2' in col:  # 节点3的x轴角速度
                waist_gyro_cols.append(col)
            elif 'y轴角速度.2' in col:  # 节点3的y轴角速度
                waist_gyro_cols.append(col)
            elif 'z轴角速度.2' in col:  # 节点3的z轴角速度
                waist_gyro_cols.append(col)
    
    return {
        'left_pressure': left_pressure_cols,
        'right_pressure': right_pressure_cols,
        'waist_acc': waist_acc_cols,
        'waist_gyro': waist_gyro_cols
    }

def extract_gait_features(df_segment, fs):
    """提取步态特征"""
    data_format = detect_data_format(df_segment)
    sensor_cols = identify_sensor_columns(df_segment, data_format)
    
    features = {}
    
    # 获取传感器数据
    left_pressure = df_segment[sensor_cols['left_pressure']]
    right_pressure = df_segment[sensor_cols['right_pressure']]
    
    # 1. 步频 (Hz) - 改为每秒步数，量级约1-3
    left_sum = left_pressure.sum(axis=1)
    right_sum = right_pressure.sum(axis=1)
    left_peaks, _ = signal.find_peaks(left_sum, height=left_sum.mean()*0.3, distance=int(fs*0.3))
    right_peaks, _ = signal.find_peaks(right_sum, height=right_sum.mean()*0.3, distance=int(fs*0.3))

    total_steps = len(left_peaks) + len(right_peaks)
    duration = len(df_segment) / fs
    step_frequency_hz = total_steps / duration if duration > 0 else 0
    features['step_frequency_hz'] = step_frequency_hz
    
    # 2. 步速 (cm/s) - 改为厘米每秒，量级约50-200
    if sensor_cols['waist_acc'] and len(sensor_cols['waist_acc']) >= 3:
        waist_acc = df_segment[sensor_cols['waist_acc']].values
        # 去除重力影响（假设z轴为垂直方向）
        waist_acc[:, 2] = waist_acc[:, 2] - 9.8  # 去除重力
        velocity = compute_velocity(waist_acc, fs)
        vel_mag = np.linalg.norm(velocity, axis=1)
        step_speed_ms = np.mean(vel_mag)
        step_speed_cms = step_speed_ms * 100  # 转换为cm/s
        features['step_speed_cms'] = step_speed_cms

        # 3. 步长 (cm) - 改为厘米，量级约30-100
        step_length_m = step_speed_ms / step_frequency_hz if step_frequency_hz > 0 else 0
        step_length_cm = step_length_m * 100  # 转换为cm
        features['step_length_cm'] = step_length_cm
        
        # 4. 左右脚步长差异 (cm) - 改为厘米，量级约0-10
        displacement = compute_displacement(velocity, fs)
        # 根据左右脚着地时间点计算步长差异
        if len(left_peaks) > 0 and len(right_peaks) > 0:
            left_step_lengths = []
            right_step_lengths = []

            # 计算左脚步长
            for i in range(len(left_peaks)-1):
                start_idx = left_peaks[i]
                end_idx = left_peaks[i+1]
                step_disp = np.linalg.norm(displacement[end_idx] - displacement[start_idx])
                left_step_lengths.append(step_disp)

            # 计算右脚步长
            for i in range(len(right_peaks)-1):
                start_idx = right_peaks[i]
                end_idx = right_peaks[i+1]
                step_disp = np.linalg.norm(displacement[end_idx] - displacement[start_idx])
                right_step_lengths.append(step_disp)

            if left_step_lengths and right_step_lengths:
                left_avg = np.mean(left_step_lengths)
                right_avg = np.mean(right_step_lengths)
                step_length_diff_m = abs(left_avg - right_avg)
                features['step_length_difference_cm'] = step_length_diff_m * 100  # 转换为cm
            else:
                features['step_length_difference_cm'] = 0
        else:
            features['step_length_difference_cm'] = 0
        
        # 5. 躯干倾斜角标准差 (°) - 量级约5-30，保持不变
        ax, ay, az = waist_acc[:, 0], waist_acc[:, 1], waist_acc[:, 2]
        pitch = np.arctan2(ax, np.sqrt(ay**2 + az**2)) * 180 / np.pi
        roll = np.arctan2(ay, np.sqrt(ax**2 + az**2)) * 180 / np.pi
        trunk_tilt_std = np.sqrt(np.std(pitch)**2 + np.std(roll)**2)
        features['trunk_tilt_std_deg'] = trunk_tilt_std

        # 6. 躯干最大偏移幅度 (°) - 量级约10-80，保持不变
        pitch_range = np.max(pitch) - np.min(pitch)
        roll_range = np.max(roll) - np.min(roll)
        trunk_max_deviation = max(pitch_range, roll_range)
        features['trunk_max_deviation_deg'] = trunk_max_deviation
        
    else:
        features['step_speed_cms'] = 0
        features['step_length_cm'] = 0
        features['step_length_difference_cm'] = 0
        features['trunk_tilt_std_deg'] = 0
        features['trunk_max_deviation_deg'] = 0

    # 7. 躯干角速度熵值 (*10) - 调整量级约10-40
    if sensor_cols['waist_gyro'] and len(sensor_cols['waist_gyro']) >= 3:
        try:
            waist_gyro = df_segment[sensor_cols['waist_gyro']].values
            entropies = []
            for i in range(waist_gyro.shape[1]):
                # 使用简化的Shannon熵计算
                signal_data = waist_gyro[:, i]
                # 将信号离散化
                hist, _ = np.histogram(signal_data, bins=10)
                hist = hist[hist > 0]  # 移除零值
                if len(hist) > 0:
                    prob = hist / np.sum(hist)
                    entropy = -np.sum(prob * np.log2(prob))
                    entropies.append(entropy)
                else:
                    entropies.append(0)
            entropy_mean = np.mean(entropies) if entropies else 0
            features['trunk_angular_velocity_entropy_x10'] = entropy_mean * 10  # 放大10倍
        except:
            features['trunk_angular_velocity_entropy_x10'] = 0
    else:
        features['trunk_angular_velocity_entropy_x10'] = 0
    
    return features

def extract_pressure_features(df_segment, fs):
    """提取足底压力特征"""
    data_format = detect_data_format(df_segment)
    sensor_cols = identify_sensor_columns(df_segment, data_format)

    features = {}

    # 获取压力数据
    left_pressure = df_segment[sensor_cols['left_pressure']]
    right_pressure = df_segment[sensor_cols['right_pressure']]

    left_sum = left_pressure.sum(axis=1)
    right_sum = right_pressure.sum(axis=1)
    total_pressure = left_sum + right_sum

    # 1. 支撑期占比 (%)
    threshold = total_pressure.mean() * 0.3
    left_support = (left_sum > threshold).sum() / len(left_sum) * 100
    right_support = (right_sum > threshold).sum() / len(right_sum) * 100
    features['left_support_ratio_pct'] = left_support
    features['right_support_ratio_pct'] = right_support

    # 2. 双支撑期时长 (%)
    double_support = ((left_sum > threshold) & (right_sum > threshold)).sum() / len(left_sum) * 100
    features['double_support_ratio_pct'] = double_support

    # 3. COP轨迹面积 (cm²)
    def compute_cop_area(pressure_data):
        coords = np.array([[0,0],[0,1],[0,2],[1,0],[1,1],[1,2],[2,1],[2,2]])
        coords = coords[:len(pressure_data.columns)]
        P = pressure_data.values
        total_p = np.sum(P, axis=1).reshape(-1, 1)
        total_p[total_p == 0] = 1e-6
        norm_P = P / total_p
        cop_x = (norm_P @ coords[:, 0])
        cop_y = (norm_P @ coords[:, 1])
        cop = np.stack([cop_x, cop_y], axis=1)
        valid = ~np.isnan(cop).any(axis=1)

        try:
            hull = ConvexHull(cop[valid])
            return hull.volume  # 2D中volume就是面积
        except:
            return 0

    features['left_cop_area_cm2'] = compute_cop_area(left_pressure)
    features['right_cop_area_cm2'] = compute_cop_area(right_pressure)

    # 4. COP轨迹速度 (cm/s)
    def compute_cop_velocity(pressure_data):
        coords = np.array([[0,0],[0,1],[0,2],[1,0],[1,1],[1,2],[2,1],[2,2]])
        coords = coords[:len(pressure_data.columns)]
        P = pressure_data.values
        total_p = np.sum(P, axis=1).reshape(-1, 1)
        total_p[total_p == 0] = 1e-6
        norm_P = P / total_p
        cop_x = (norm_P @ coords[:, 0])
        cop_y = (norm_P @ coords[:, 1])
        cop = np.stack([cop_x, cop_y], axis=1)
        valid = ~np.isnan(cop).any(axis=1)

        if np.sum(valid) > 1:
            cop_valid = cop[valid]
            distances = np.linalg.norm(np.diff(cop_valid, axis=0), axis=1)
            return np.mean(distances) * fs  # 转换为速度
        return 0

    features['left_cop_velocity_cms'] = compute_cop_velocity(left_pressure)
    features['right_cop_velocity_cms'] = compute_cop_velocity(right_pressure)

    # 5. 压力对称性指数 (%)
    left_mean = left_sum.mean()
    right_mean = right_sum.mean()
    if (left_mean + right_mean) > 0:
        symmetry_index = abs(left_mean - right_mean) / ((left_mean + right_mean) / 2) * 100
    else:
        symmetry_index = 0
    features['pressure_symmetry_index_pct'] = symmetry_index

    # 6. 左/右足压力变异系数 (%)
    features['left_pressure_cv_pct'] = (left_sum.std() / left_sum.mean() * 100) if left_sum.mean() > 0 else 0
    features['right_pressure_cv_pct'] = (right_sum.std() / right_sum.mean() * 100) if right_sum.mean() > 0 else 0

    # 7. 最大压力 (N)
    features['left_max_pressure_N'] = left_sum.max()
    features['right_max_pressure_N'] = right_sum.max()

    # 8. 压力分布差异 (RMSE)
    ln = left_pressure.values
    rn = right_pressure.values
    ln_sum = ln.sum(axis=1).reshape(-1, 1)
    rn_sum = rn.sum(axis=1).reshape(-1, 1)
    ln_sum[ln_sum == 0] = 1e-6
    rn_sum[rn_sum == 0] = 1e-6
    ln_norm = ln / ln_sum
    rn_norm = rn / rn_sum
    rmse = np.sqrt(np.mean((ln_norm - rn_norm)**2))
    features['pressure_distribution_rmse'] = rmse

    return features

def extract_coupling_features(df_segment, fs):
    """提取足-惯融合耦合特征"""
    data_format = detect_data_format(df_segment)
    sensor_cols = identify_sensor_columns(df_segment, data_format)

    features = {}

    # 获取传感器数据
    left_pressure = df_segment[sensor_cols['left_pressure']]
    right_pressure = df_segment[sensor_cols['right_pressure']]
    left_sum = left_pressure.sum(axis=1)
    right_sum = right_pressure.sum(axis=1)
    total_pressure = left_sum + right_sum

    if sensor_cols['waist_acc'] and len(sensor_cols['waist_acc']) >= 3:
        waist_acc = df_segment[sensor_cols['waist_acc']].values
        waist_acc[:, 2] = waist_acc[:, 2] - 9.8  # 去除重力
        velocity = compute_velocity(waist_acc, fs)
        vel_mag = np.linalg.norm(velocity, axis=1)

        # 1. 足部压力-躯干稳定性耦合指标
        # 计算躯干角度
        ax, ay, az = waist_acc[:, 0], waist_acc[:, 1], waist_acc[:, 2]
        pitch = np.arctan2(ax, np.sqrt(ay**2 + az**2))
        roll = np.arctan2(ay, np.sqrt(ax**2 + az**2))
        trunk_stability = np.sqrt(pitch**2 + roll**2)  # 躯干稳定性指标

        # 计算相关系数
        try:
            pressure_stability_corr, _ = pearsonr(total_pressure, trunk_stability)
            features['pressure_trunk_stability_corr'] = pressure_stability_corr if not np.isnan(pressure_stability_corr) else 0
        except:
            features['pressure_trunk_stability_corr'] = 0

        # 2. 能量耗散指标
        # 速度模 × 总压力 × dt 的积分
        energy_dissipation = np.sum(vel_mag * total_pressure) / fs
        features['energy_dissipation_index'] = energy_dissipation

        # 3. 足-躯协调时延指标
        # 找到足底最大压力时间点
        max_pressure_idx = np.argmax(total_pressure)
        max_pressure_time = max_pressure_idx / fs

        # 找到躯干最大角度偏移时间点
        max_trunk_deviation_idx = np.argmax(trunk_stability)
        max_trunk_deviation_time = max_trunk_deviation_idx / fs

        # 计算时间差
        coordination_delay = abs(max_pressure_time - max_trunk_deviation_time)
        features['foot_trunk_coordination_delay_s'] = coordination_delay

        # 4. 步态相位-足底压力耦合特征
        # 检测步态周期
        left_peaks, _ = signal.find_peaks(left_sum, height=left_sum.mean()*0.3, distance=int(fs*0.3))
        right_peaks, _ = signal.find_peaks(right_sum, height=right_sum.mean()*0.3, distance=int(fs*0.3))

        if len(left_peaks) > 1 and len(right_peaks) > 1:
            # 计算步态周期内压力变化的变异系数
            all_peaks = np.sort(np.concatenate([left_peaks, right_peaks]))
            if len(all_peaks) > 2:
                cycle_pressures = []
                for i in range(len(all_peaks)-1):
                    start_idx = all_peaks[i]
                    end_idx = all_peaks[i+1]
                    cycle_pressure = np.mean(total_pressure[start_idx:end_idx])
                    cycle_pressures.append(cycle_pressure)

                if cycle_pressures:
                    gait_phase_pressure_cv = np.std(cycle_pressures) / np.mean(cycle_pressures) if np.mean(cycle_pressures) > 0 else 0
                    features['gait_phase_pressure_coupling_cv'] = gait_phase_pressure_cv
                else:
                    features['gait_phase_pressure_coupling_cv'] = 0
            else:
                features['gait_phase_pressure_coupling_cv'] = 0
        else:
            features['gait_phase_pressure_coupling_cv'] = 0

    else:
        # 如果没有腰部加速度数据，设置默认值
        features['pressure_trunk_stability_corr'] = 0
        features['energy_dissipation_index'] = 0
        features['foot_trunk_coordination_delay_s'] = 0
        features['gait_phase_pressure_coupling_cv'] = 0

    # 5. 角速度相关的耦合特征
    if sensor_cols['waist_gyro'] and len(sensor_cols['waist_gyro']) >= 3:
        waist_gyro = df_segment[sensor_cols['waist_gyro']].values
        gyro_mag = np.linalg.norm(waist_gyro, axis=1)

        # 角速度与压力的相关性
        try:
            gyro_pressure_corr, _ = pearsonr(gyro_mag, total_pressure)
            features['gyro_pressure_corr'] = gyro_pressure_corr if not np.isnan(gyro_pressure_corr) else 0
        except:
            features['gyro_pressure_corr'] = 0
    else:
        features['gyro_pressure_corr'] = 0

    return features

def extract_features_segment(df_segment, fs, feature_types=['gait', 'pressure', 'coupling'], label=None):
    """提取单个数据段的特征"""
    results = {}

    if 'gait' in feature_types:
        gait_features = extract_gait_features(df_segment, fs)
        # 处理NaN值
        for key, value in gait_features.items():
            if np.isnan(value) or np.isinf(value):
                gait_features[key] = 0.0
        if label is not None:
            gait_features['label'] = label
        results['gait'] = pd.DataFrame([gait_features])

    if 'pressure' in feature_types:
        pressure_features = extract_pressure_features(df_segment, fs)
        # 处理NaN值
        for key, value in pressure_features.items():
            if np.isnan(value) or np.isinf(value):
                pressure_features[key] = 0.0
        if label is not None:
            pressure_features['label'] = label
        results['pressure'] = pd.DataFrame([pressure_features])

    if 'coupling' in feature_types:
        coupling_features = extract_coupling_features(df_segment, fs)
        # 处理NaN值
        for key, value in coupling_features.items():
            if np.isnan(value) or np.isinf(value):
                coupling_features[key] = 0.0
        if label is not None:
            coupling_features['label'] = label
        results['coupling'] = pd.DataFrame([coupling_features])

    return results

def extract_features(data_file, save_dirs=None, fs=100, window_size=100, stride=100,
                    mode='sliding', feature_types=['gait', 'pressure', 'coupling']):
    """从数据文件中提取特征"""
    try:
        df = pd.read_csv(data_file)
        # 只使用前400列数据
        if df.shape[1] > 400:
            # 保留前399列数据和最后一列标签
            df = pd.concat([df.iloc[:, :399], df.iloc[:, -1:]], axis=1)

        # 对于滑窗模式，只使用前400行数据
        if mode == 'sliding' and len(df) > 400:
            # 保留前400行数据，但保持标签信息
            label_col = df['label'].iloc[0] if 'label' in df.columns else None
            df = df.iloc[:400].copy()
            if label_col is not None:
                df['label'] = label_col

        # 检查并处理连续0值问题
        df = clean_zero_values(df)

    except Exception as e:
        print(f"读取文件失败 {data_file}: {e}")
        return None

    # 提取标签
    label = None
    if 'label' in df.columns:
        label = df['label'].iloc[0]

    results = {ftype: [] for ftype in feature_types}

    if mode == 'static':
        # 静态模式：整个文件作为一个特征向量
        feature_dfs = extract_features_segment(df.reset_index(drop=True), fs, feature_types, label)

        for ftype in feature_types:
            if ftype in feature_dfs:
                feature_df = feature_dfs[ftype].copy()
                results[ftype].append(feature_df)

    else:
        # 滑窗模式：使用滑动窗口，确保每个文件提取相同数量的特征段
        target_segments = calculate_target_segments(len(df), window_size, stride)

        if len(df) < window_size:
            # 数据长度不足，使用静态模式
            feature_dfs = extract_features_segment(df.reset_index(drop=True), fs, feature_types, label)

            for ftype in feature_types:
                if ftype in feature_dfs:
                    feature_df = feature_dfs[ftype].copy()
                    results[ftype].append(feature_df)
        else:
            # 固定提取8次，每50步提取一次
            segment_count = 0
            for i in range(target_segments):
                start = i * 50  # 每50步提取一次
                if start + window_size > len(df):
                    break

                segment = df.iloc[start:start+window_size].reset_index(drop=True)
                if len(segment) < window_size:
                    continue

                feature_dfs = extract_features_segment(segment, fs, feature_types, label)

                for ftype in feature_types:
                    if ftype in feature_dfs:
                        feature_df = feature_dfs[ftype].copy()
                        results[ftype].append(feature_df)

                segment_count += 1

    # 合并每种类型的特征
    final_results = {}
    for ftype in feature_types:
        if results[ftype]:
            final_results[ftype] = pd.concat(results[ftype], ignore_index=True)
        else:
            final_results[ftype] = None

    # 保存文件
    if save_dirs:
        for ftype in feature_types:
            if ftype in save_dirs and final_results[ftype] is not None:
                os.makedirs(save_dirs[ftype], exist_ok=True)
                # 简化文件名，只保留原始文件名
                base_name = os.path.splitext(os.path.basename(data_file))[0]
                filename = f"{base_name}_{ftype}.csv"
                out_path = os.path.join(save_dirs[ftype], filename)
                final_results[ftype].to_csv(out_path, index=False)

    return final_results

def process_directory(input_dir, output_base_dir, fs=100, window_size=100, stride=100,
                     mode='sliding', feature_types=['gait', 'pressure', 'coupling']):
    """处理整个目录的数据文件"""
    print(f"开始处理目录: {input_dir} ({mode}模式)")

    # 创建保存目录
    save_dirs = {}
    for ftype in feature_types:
        save_dirs[ftype] = os.path.join(output_base_dir, f"{ftype}_features_{mode}")
        os.makedirs(save_dirs[ftype], exist_ok=True)

    # 获取所有CSV文件
    files = glob.glob(os.path.join(input_dir, '*.csv'))
    print(f"找到 {len(files)} 个CSV文件")

    if not files:
        print("未找到任何CSV文件")
        return

    # 处理每个文件
    all_features = {ftype: [] for ftype in feature_types}
    successful_files = 0

    for i, file in enumerate(files):
        if (i + 1) % 10 == 0 or i == 0:  # 每10个文件显示一次进度
            print(f"处理进度: {i+1}/{len(files)}")

        file_features = extract_features(
            file, save_dirs, fs, window_size, stride, mode, feature_types
        )

        if file_features:
            for ftype in feature_types:
                if file_features[ftype] is not None:
                    all_features[ftype].append(file_features[ftype])
            successful_files += 1

    # 统计处理结果和一致性检查
    print(f"\n处理结果统计:")
    for ftype in feature_types:
        if all_features[ftype]:
            combined_features = pd.concat(all_features[ftype], ignore_index=True)

            # 计算特征维度（排除标签列）
            feature_cols = [col for col in combined_features.columns if col != 'label']

            # 显示标签分布
            label_dist = ""
            if 'label' in combined_features.columns:
                label_counts = combined_features['label'].value_counts().sort_index()
                label_dist = ", ".join([f"{int(label)}:{count}" for label, count in label_counts.items()])

            # 一致性检查：每个文件的特征段数
            if mode == 'sliding':
                segments_per_file = []
                for feature_df in all_features[ftype]:
                    segments_per_file.append(len(feature_df))

                unique_counts = set(segments_per_file)
                if len(unique_counts) == 1:
                    segments_info = f", 每文件{list(unique_counts)[0]}段"
                else:
                    segments_info = f", 段数不一致{unique_counts}"
            else:
                segments_info = ""

            print(f"✅ {ftype}特征: {len(combined_features)}个样本, {len(feature_cols)}维特征, 标签分布[{label_dist}]{segments_info}")
        else:
            print(f"❌ {ftype}特征：处理失败")

    print(f"完成! 成功处理 {successful_files}/{len(files)} 个文件")

def main():
    """主函数：演示全面特征提取方案"""

    # 配置参数
    input_dir = 'my_augmented_results0'  # 输入目录
    base_output_dir = 'comprehensive_features00'  # 基础输出目录
    fs = 100  # 采样频率
    window_size = 100  # 滑窗大小（100个时间步，1秒）
    stride = 50  # 滑窗步长（每50步提取一次）

    print("全面特征提取程序 (量级归一化版本)")
    print("=" * 50)
    print("特征量级调整:")
    print("- 步频: Hz (约1-3)")
    print("- 步速: cm/s (约50-200)")
    print("- 步长: cm (约30-100)")
    print("- 步长差异: cm (约0-10)")
    print("- 躯干倾斜角标准差: 度 (约5-30)")
    print("- 躯干最大偏移幅度: 度 (约10-80)")
    print("- 角速度熵值: x10 (约10-40)")
    print("- 时序特征: 每50步提取一次，共8次")
    print("=" * 50)

    # 方案1：静态特征提取
    print("\n方案1：静态模式特征提取")
    output_dir_1 = os.path.join(base_output_dir, 'static')
    process_directory(
        input_dir=input_dir,
        output_base_dir=output_dir_1,
        fs=fs,
        mode='static',
        feature_types=['gait', 'pressure', 'coupling']
    )

    # 方案2：滑窗特征提取
    print("\n方案2：滑窗模式特征提取")
    output_dir_2 = os.path.join(base_output_dir, 'sliding')
    process_directory(
        input_dir=input_dir,
        output_base_dir=output_dir_2,
        fs=fs,
        window_size=window_size,
        stride=stride,
        mode='sliding',
        feature_types=['gait', 'pressure', 'coupling']
    )

    print("\n所有特征提取方案完成!")

if __name__ == "__main__":
    main()
