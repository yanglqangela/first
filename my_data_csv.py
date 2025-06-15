import numpy as np
import torch
import pandas as pd
import os
from tqdm import tqdm

# 尝试导入必要的模块，提供友好的错误信息
try:
    from Aplus.data import BaseDataset, random_index
except ImportError as e:
    print(f"错误: 无法导入Aplus模块 - {e}")
    print("请确保项目依赖已正确安装。")
    
try:
    from articulate.math import quaternion_to_rotation_matrix, rotation_matrix_to_r6d
except ImportError as e:
    print(f"错误: 无法导入articulate.math模块 - {e}")
    print("请确保项目依赖已正确安装。")
    
try:
    from config import joint_set
except ImportError as e:
    print(f"错误: 无法导入config模块 - {e}")
    print("请确保项目文件结构正确。")

# 检查quaternion模块
try:
    import quaternion
    HAS_QUATERNION = True
except ImportError:
    HAS_QUATERNION = False
    print("警告: 缺少'quaternion'模块。请运行以下命令安装:")
    print("    python Loose-Inertial-Poser-main/install_deps.py")
    print("或者直接安装:")
    print("    pip install numpy-quaternion")

class CSVData(BaseDataset):
    def __init__(self, x: torch.Tensor, y: torch.Tensor, y2=None, seq_len=20, shuffle=False, step=1):
        # 首先检查quaternion模块是否可用
        if not HAS_QUATERNION:
            raise ImportError("缺少'quaternion'模块。请先运行 'python Loose-Inertial-Poser-main/install_deps.py' 安装所有依赖。")
            
        self.x = x[::step]
        self.y = y[::step]
        self.y2 = y2[::step] if y2 is not None else None
        self.data_len = len(self.x) - seq_len
        self.seq_len = seq_len
        if shuffle:
            self.indexer = random_index(data_len=self.data_len, seed=42)
        else:
            self.indexer = [i for i in range(self.data_len)]

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        i = (index % self.data_len)
        data = [self.x[self.indexer[i]:self.indexer[i] + self.seq_len], self.y[self.indexer[i]:self.indexer[i] + self.seq_len]]
        if self.y2 is not None:
            data.append(self.y2[self.indexer[i]:self.indexer[i] + self.seq_len])
        return tuple(data)

    @staticmethod
    def load_data(folder_path='Loose-Inertial-Poser-main/original', pose_type='r6d', acc_scale=30) -> dict:
        """
        Load data from CSV files. This function processes the CSV format data with 7 IMU nodes,
        foot pressure data, and labels.
        
        Args:
            folder_path: Path to the folder containing CSV files
            pose_type: Type of pose representation, 'r6d' or 'axis_angle'
            acc_scale: Scaling factor for acceleration data
            
        Returns:
            Dict of processed data
        """
        # 首先检查quaternion模块是否可用
        if not HAS_QUATERNION:
            raise ImportError("缺少'quaternion'模块。请先运行 'python Loose-Inertial-Poser-main/install_deps.py' 安装所有依赖。")
            
        all_data = []
        
        # List all CSV files in the directory
        csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
        print(f"Found {len(csv_files)} CSV files in {folder_path}")
        
        for csv_file in tqdm(csv_files, desc="Processing CSV files"):
            file_path = os.path.join(folder_path, csv_file)
            try:
                # Read CSV file
                df = pd.read_csv(file_path, encoding='utf-8')
                all_data.append(df)
            except UnicodeDecodeError:
                # If utf-8 fails, try with a different encoding
                try:
                    df = pd.read_csv(file_path, encoding='gbk')
                    all_data.append(df)
                except Exception as e:
                    print(f"Error reading {csv_file}: {e}")
            except Exception as e:
                print(f"Error processing {csv_file}: {e}")
                
        if not all_data:
            raise ValueError("No data was loaded from the CSV files.")
        
        # Concatenate all dataframes
        combined_data = pd.concat(all_data, ignore_index=True)
        print(f"Combined data shape: {combined_data.shape}")
        
        # Number of IMU nodes
        num_imu_nodes = 7
        
        # Process data
        processed_data = []
        labels = []
        
        # Extract IMU data and convert to the format required by the model
        imu_data = []
        
        # Process each row in the dataframe
        for _, row in tqdm(combined_data.iterrows(), total=len(combined_data), desc="Processing rows"):
            # Process each IMU node
            row_imu_data = []
            
            for node_idx in range(num_imu_nodes):
                # Extract acceleration data (3 axes)
                acc_start_idx = node_idx * 13  # Each node has 13 values (3 acc + 3 gyro + 3 mag + 4 quat)
                acc_data = row.iloc[acc_start_idx:acc_start_idx+3].values.astype(float)
                
                # Extract gyroscope data (3 axes)
                gyro_start_idx = acc_start_idx + 3
                gyro_data = row.iloc[gyro_start_idx:gyro_start_idx+3].values.astype(float)
                
                # Extract magnetometer data (3 axes)
                mag_start_idx = gyro_start_idx + 3
                mag_data = row.iloc[mag_start_idx:mag_start_idx+3].values.astype(float)
                
                # Extract quaternion data (4 values)
                quat_start_idx = mag_start_idx + 3
                quat_data = row.iloc[quat_start_idx:quat_start_idx+4].values.astype(float)
                
                # Convert quaternion to rotation matrix
                quat_tensor = torch.tensor([quat_data], dtype=torch.float32)
                rot_matrix = quaternion_to_rotation_matrix(quat_tensor)
                
                # Scale acceleration data
                acc_data = acc_data / acc_scale
                
                # Append data for this node
                node_data = {
                    'acc': acc_data,
                    'gyro': gyro_data,
                    'mag': mag_data,
                    'quat': quat_data,
                    'rot_matrix': rot_matrix.numpy()[0]
                }
                row_imu_data.append(node_data)
            
            # Process left and right foot pressure data
            left_foot_start_idx = num_imu_nodes * 13
            left_foot_data = row.iloc[left_foot_start_idx:left_foot_start_idx+8].values.astype(float)
            
            right_foot_start_idx = left_foot_start_idx + 8
            right_foot_data = row.iloc[right_foot_start_idx:right_foot_start_idx+8].values.astype(float)
            
            # Get label
            label = row.iloc[-1]
            
            # Store processed data
            processed_data.append({
                'imu_data': row_imu_data,
                'left_foot': left_foot_data,
                'right_foot': right_foot_data
            })
            labels.append(label)
        
        # Convert to PyTorch tensors
        # Create input for the model (acc + rot)
        all_acc = []
        all_rot = []
        
        for data_point in processed_data:
            imu_nodes = data_point['imu_data']
            
            # Extract acc and rot data for each IMU node
            acc_data = []
            rot_data = []
            
            for node in imu_nodes:
                acc_data.append(node['acc'])
                rot_matrix = node['rot_matrix']
                rot_matrix_tensor = torch.tensor(rot_matrix, dtype=torch.float32)
                
                if pose_type == 'r6d':
                    rot_r6d = rotation_matrix_to_r6d(rot_matrix_tensor.unsqueeze(0))[0]
                    rot_data.append(rot_r6d.numpy())
                else:
                    # Use rotation matrix as is
                    rot_data.append(rot_matrix.flatten())
            
            # Concatenate acc data from all nodes
            acc_tensor = torch.tensor(np.concatenate(acc_data), dtype=torch.float32)
            all_acc.append(acc_tensor)
            
            # Concatenate rot data from all nodes
            rot_tensor = torch.tensor(np.concatenate(rot_data), dtype=torch.float32)
            all_rot.append(rot_tensor)
        
        # Stack all data
        all_acc_tensor = torch.stack(all_acc)
        all_rot_tensor = torch.stack(all_rot)
        
        # Combine acc and rot data for model input
        x_s1 = torch.cat((all_acc_tensor, all_rot_tensor), dim=1)
        
        # For pose data, we'll use a placeholder since we don't have actual pose data
        # In a real scenario, you might want to derive this from the IMU data or have it as ground truth
        pose_placeholder = torch.zeros((len(x_s1), len(joint_set.index_pose) * (6 if pose_type == 'r6d' else 3)))
        
        # For joint positions, another placeholder
        joint_placeholder = torch.zeros((len(x_s1), (len(joint_set.index_joint)) * 3))
        
        # Convert labels to tensor if needed
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        
        return {
            'x_s1': x_s1,
            'joint_upper_body': joint_placeholder,
            'pose_all': pose_placeholder,
            'labels': labels_tensor
        } 