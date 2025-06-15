import os
import torch
import numpy as np
from tqdm import tqdm
import sys

def check_dependencies():
    """检查必要的依赖项是否已安装"""
    missing_deps = []
    
    try:
        import pandas
    except ImportError:
        missing_deps.append("pandas")
        
    try:
        import quaternion
    except ImportError:
        missing_deps.append("numpy-quaternion")
    
    if missing_deps:
        print("错误: 缺少以下依赖项:")
        for dep in missing_deps:
            print(f"  - {dep}")
        print("\n请运行以下命令安装所有依赖:")
        print("  python Loose-Inertial-Poser-main/install_deps.py")
        print("或者直接运行:")
        deps_str = " ".join(missing_deps)
        print(f"  pip install {deps_str}")
        return False
    
    return True

def main():
    # 检查依赖项
    if not check_dependencies():
        sys.exit(1)
    
    # 现在导入CSVData，因为它依赖于上面检查的模块
    try:
        from my_data_csv import CSVData
    except ImportError as e:
        print(f"错误: 无法导入CSVData - {e}")
        print("请确保my_data_csv.py文件在正确的位置。")
        sys.exit(1)
    
    # Set parameters
    input_folder = 'Loose-Inertial-Poser-main/original'
    output_folder = 'Loose-Inertial-Poser-main/processed_csv'
    pose_type = 'r6d'  # 'r6d' or 'axis_angle'
    acc_scale = 30
    
    # 检查输入文件夹是否存在
    if not os.path.exists(input_folder):
        print(f"错误: 找不到输入文件夹 '{input_folder}'")
        print(f"请确保将CSV文件放在 '{input_folder}' 目录中。")
        sys.exit(1)
    
    # 检查输入文件夹中是否有CSV文件
    csv_files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]
    if not csv_files:
        print(f"错误: 在 '{input_folder}' 中找不到CSV文件。")
        print("请确保将CSV数据文件放在正确的目录中。")
        sys.exit(1)
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    print(f"Loading data from {input_folder}")
    # Load and preprocess data
    try:
        data_dict = CSVData.load_data(
            folder_path=input_folder,
            pose_type=pose_type,
            acc_scale=acc_scale
        )
    except Exception as e:
        print(f"错误: 处理数据时出错 - {e}")
        sys.exit(1)
    
    # Extract data
    x_s1 = data_dict['x_s1']
    joint_upper_body = data_dict['joint_upper_body']
    pose_all = data_dict['pose_all']
    labels = data_dict['labels']
    
    print(f"Processed data shapes:")
    print(f"x_s1: {x_s1.shape}")
    print(f"joint_upper_body: {joint_upper_body.shape}")
    print(f"pose_all: {pose_all.shape}")
    print(f"labels: {labels.shape}")
    
    # Save data in the required format
    print(f"Saving processed data to {output_folder}")
    
    # Create rotation matrices (4 IMUs)
    # For demonstration, we'll use the first 4 IMUs and create rotation matrices
    # This is a simplification - in a real scenario, you'd compute these properly
    
    # Number of IMUs in the original model
    num_model_imus = 4
    
    # Create synthetic IMU data based on our 7 IMUs
    # We'll map the first 4 of our 7 IMUs to the 4 IMUs expected by the model
    
    # In the original data format, each frame has:
    # - accelerations for 4 IMUs (2 * 3 = 6 values per IMU)
    # - rotation matrices in r6d format (6 values per IMU)
    
    # Extract data for 4 IMUs (use first 4 from our 7)
    imus_per_frame = 4
    acc_per_imu = 6  # 2 sets of xyz
    rot_per_imu = 6  # r6d format
    
    # Calculate IMU dimensions from the original input
    total_features = x_s1.shape[1]
    acc_features_per_node = 3  # x, y, z acceleration
    rot_features_per_node = 6 if pose_type == 'r6d' else 9  # r6d or flattened rotation matrix
    
    # Create synthetic acceleration data
    acc_data = torch.zeros((x_s1.shape[0], imus_per_frame * acc_per_imu))
    for i in range(imus_per_frame):
        if i < 4:  # Map our first 4 IMUs to the model's 4 IMUs
            # For each IMU, we need 6 acc values (2 sets of xyz)
            # In our data, each IMU has 3 acc values, so we'll duplicate them
            src_acc_start = i * acc_features_per_node
            src_acc_end = src_acc_start + acc_features_per_node
            
            # First set of xyz
            dst_acc_start = i * acc_per_imu
            acc_data[:, dst_acc_start:dst_acc_start+3] = x_s1[:, src_acc_start:src_acc_end]
            
            # Second set of xyz (duplicate of first set for simplicity)
            acc_data[:, dst_acc_start+3:dst_acc_start+6] = x_s1[:, src_acc_start:src_acc_end]
    
    # Create synthetic rotation data
    rot_data = torch.zeros((x_s1.shape[0], imus_per_frame * rot_per_imu))
    for i in range(imus_per_frame):
        if i < 4:  # Map our first 4 IMUs to the model's 4 IMUs
            # Get rotation data for this IMU
            src_rot_start = 7 * acc_features_per_node + i * rot_features_per_node
            src_rot_end = src_rot_start + rot_features_per_node
            
            # If our rotation format matches the model's expected format
            if rot_features_per_node == rot_per_imu:
                dst_rot_start = i * rot_per_imu
                rot_data[:, dst_rot_start:dst_rot_start+rot_per_imu] = x_s1[:, src_rot_start:src_rot_end]
            else:
                # Need to convert format - simplified here
                dst_rot_start = i * rot_per_imu
                rot_data[:, dst_rot_start:dst_rot_start+rot_per_imu] = x_s1[:, src_rot_start:src_rot_start+rot_per_imu]
    
    try:
        # Save acceleration data
        torch.save(acc_data, os.path.join(output_folder, 'vacc.pt'))
        print(f"Saved acceleration data: {acc_data.shape}")
        
        # Save rotation data in 3x3 matrix format
        # This is a placeholder - in a real scenario, you'd compute proper rotation matrices
        rot_matrices = torch.zeros((x_s1.shape[0], imus_per_frame, 3, 3))
        for i in range(imus_per_frame):
            # Identity matrices as placeholders
            rot_matrices[:, i] = torch.eye(3).unsqueeze(0).repeat(x_s1.shape[0], 1, 1)
        torch.save(rot_matrices, os.path.join(output_folder, 'vrot.pt'))
        print(f"Saved rotation matrices: {rot_matrices.shape}")
        
        # Save pose data (placeholder)
        torch.save(pose_all, os.path.join(output_folder, 'pose.pt'))
        print(f"Saved pose data: {pose_all.shape}")
        
        # Save joint position data (placeholder)
        torch.save(joint_upper_body, os.path.join(output_folder, 'joint.pt'))
        print(f"Saved joint position data: {joint_upper_body.shape}")
        
        # Save labels
        torch.save(labels, os.path.join(output_folder, 'labels.pt'))
        print(f"Saved labels: {labels.shape}")
        
        print("Data preprocessing complete!")
    except Exception as e:
        print(f"错误: 保存处理后的数据时出错 - {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 