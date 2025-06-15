import sys
import os

# 首先检查依赖项，在导入其他模块前
def check_dependencies():
    """检查必要的依赖项是否已安装"""
    missing_deps = []
    
    required_packages = [
        "torch", "numpy", "pandas", "tqdm", "quaternion", "matplotlib", "openpyxl"
    ]
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            if package == "quaternion":
                missing_deps.append("numpy-quaternion")
            else:
                missing_deps.append(package)
    
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

# 检查依赖项
if not check_dependencies():
    sys.exit(1)

# 导入其他模块
try:
    import torch.nn as nn
    import torch
    import random
    import pandas as pd
    import numpy as np
    from my_model import *
    from config import paths, joint_set
    from my_data import *
    from my_trainner import MyTrainer, MyEvaluator
    from Aplus.models import EasyLSTM
    from config import joint_set, paths
    from my_model import SemoAE
    from my_data_csv import CSVData
except ImportError as e:
    print(f"错误: 导入模块时出错 - {e}")
    print("请确保所有依赖项已正确安装。")
    sys.exit(1)


seq_len = 128
use_elbow_angle = False

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Data preparation for training and validation split
processed_data_path = 'Loose-Inertial-Poser-main/processed_csv'

# If processed data doesn't exist, run the preprocessing script
if not os.path.exists(processed_data_path):
    print("Processed data not found. Running preprocessing script...")
    try:
        from preprocess_csv_data import main as preprocess_main
        preprocess_main()
    except Exception as e:
        print(f"错误: 运行预处理脚本时出错 - {e}")
        print("请尝试手动运行预处理脚本:")
        print("  python Loose-Inertial-Poser-main/preprocess_csv_data.py")
        sys.exit(1)

# 再次检查处理后的数据是否存在
if not os.path.exists(processed_data_path):
    print(f"错误: 预处理后仍找不到数据目录 '{processed_data_path}'")
    sys.exit(1)

# 检查必要的PT文件是否存在
required_files = ['vacc.pt', 'vrot.pt', 'joint.pt', 'pose.pt', 'labels.pt']
for file in required_files:
    file_path = os.path.join(processed_data_path, file)
    if not os.path.exists(file_path):
        print(f"错误: 找不到必要的数据文件 '{file_path}'")
        print("请确保预处理步骤正确完成。")
        sys.exit(1)

# Load the processed CSV data
print("Loading processed CSV data...")

# Function to load the saved PT files
def load_pt_data(folder_path):
    try:
        x_s1 = torch.load(os.path.join(folder_path, 'vacc.pt'))
        rot_data = torch.load(os.path.join(folder_path, 'vrot.pt'))
        
        # Convert rotation matrices to format expected by the model
        rot_dim = 6  # r6d format
        batch_size, num_imus, _, _ = rot_data.shape
        
        # Flatten and reorganize rotation data
        rot_flattened = rot_data.reshape(batch_size, -1)
        
        # Combine acc and rot data
        x_s1_combined = torch.cat((x_s1, rot_flattened), dim=1)
        
        # Load other data
        joint_upper_body = torch.load(os.path.join(folder_path, 'joint.pt'))
        pose_all = torch.load(os.path.join(folder_path, 'pose.pt'))
        labels = torch.load(os.path.join(folder_path, 'labels.pt'))
        
        return {
            'x_s1': x_s1_combined,
            'joint_upper_body': joint_upper_body,
            'pose_all': pose_all,
            'labels': labels
        }
    except Exception as e:
        print(f"错误: 加载预处理数据时出错 - {e}")
        sys.exit(1)

try:
    csv_data = load_pt_data(processed_data_path)
    
    # 检查数据是否有足够的样本
    if len(csv_data['x_s1']) < seq_len * 2:  # 确保至少有足够的样本来创建训练序列
        print(f"错误: 数据样本数量不足 ({len(csv_data['x_s1'])} 样本)")
        print(f"需要至少 {seq_len * 2} 个样本来创建有效的训练序列。")
        sys.exit(1)

    # Split the data: 80% training, 20% testing
    total_samples = len(csv_data['x_s1'])
    train_size = int(0.8 * total_samples)
    test_size = total_samples - train_size
    
    # Shuffle indices
    indices = torch.randperm(total_samples)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    
    # Create training and testing datasets
    train_x_s1 = csv_data['x_s1'][train_indices]
    train_joint_upper_body = csv_data['joint_upper_body'][train_indices]
    train_pose_all = csv_data['pose_all'][train_indices]
    
    test_x_s1 = csv_data['x_s1'][test_indices]
    test_joint_upper_body = csv_data['joint_upper_body'][test_indices]
    test_pose_all = csv_data['pose_all'][test_indices]
    
    # Create dataset objects
    data_train = CSVData(x=train_x_s1,
                         y=train_joint_upper_body,
                         y2=train_pose_all, 
                         seq_len=seq_len, 
                         step=2)
    
    data_test = CSVData(x=test_x_s1,
                        y=test_joint_upper_body,
                        y2=test_pose_all, 
                        seq_len=seq_len, 
                        step=2)
    
    print(f"Training dataset: {len(data_train)} sequences")
    print(f"Testing dataset: {len(data_test)} sequences")
    
    # Define the network architecture
    # Calculate input size based on the actual data
    n_input_s1 = train_x_s1.shape[1]  # Number of features in the input
    n_output_s1 = train_joint_upper_body.shape[1]  # Number of features in the joint output
    n_output_s2 = train_pose_all.shape[1]  # Number of features in the pose output
    
    print(f"Input features: {n_input_s1}, Output S1: {n_output_s1}, Output S2: {n_output_s2}")
    
    model_s1 = EasyLSTM(
        n_input=n_input_s1, 
        n_hidden=256, 
        n_output=n_output_s1, 
        n_lstm_layer=2, 
        bidirectional=False, 
        output_type='seq', 
        dropout=0.2
    ).to(device)
    
    model_s2 = EasyLSTM(
        n_input=n_input_s1 + n_output_s1, 
        n_hidden=256, 
        n_output=n_output_s2, 
        n_lstm_layer=2, 
        bidirectional=False, 
        output_type='seq', 
        dropout=0.2
    ).to(device)
    
    # Initialize the combined poser model
    poser_model = BiPoser(net_s1=model_s1, net_s2=model_s2).to(device)
    
    # Try to load SemoAE model if available
    SemoAE_model = None
    try:
        SemoAE_model = SemoAE(feat_dim=60, encode_dim=16).to(device)
        #SemoAE_model = SemoAE(feat_dim=12+24, encode_dim=16).to(device)
        SemoAE_model.restore(checkpoint_path='./checkpoint/SemoAE_10.pth')
        print("Loaded SemoAE model successfully")
    except Exception as e:
        print(f"警告: 无法加载SemoAE模型: {e}")
        print("将继续训练，但不使用SemoAE模型。")
    
    # Setup optimizer and loss function
    optimizer = torch.optim.Adam(poser_model.parameters(), lr=5e-4, weight_decay=1e-5)
    criterion = nn.MSELoss()
    
    # Initialize trainer and evaluator
    trainner = BiPoserTrainner(
        model=poser_model, 
        data=data_train, 
        optimizer=optimizer, 
        batch_size=256, 
        loss_func=criterion,
        SemoAE=SemoAE_model
    )
    
    evaluator = BiPoserEvaluator.from_trainner(trainner, data_eval=data_test)
    
    # Try to restore previous checkpoint if available
    try:
        trainner.restore(checkpoint_path=f'./checkpoint/CSV_10.pth', load_optimizer=True)
        print("Restored previous training checkpoint")
    except Exception as e:
        print(f"Starting training from scratch: {e}")
    
    # 确保checkpoint和log目录存在
    os.makedirs('./checkpoint', exist_ok=True)
    os.makedirs('./log', exist_ok=True)
    
    # Training loop
    epochs_per_save = 1
    total_epochs = 10
    
    for i in range(total_epochs):
        model_name = 'CSV'
        print(f"Training epoch {i+1}/{total_epochs}")
        trainner.run(epoch=epochs_per_save, evaluator=evaluator, data_shuffle=True, eta=2)
        trainner.save(folder_path='./checkpoint', model_name=model_name)
        trainner.log_export(f'./log/{model_name}.xlsx')
    
    print("Training complete!")


except Exception as e:
    print(f"训练过程中出错: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1) 