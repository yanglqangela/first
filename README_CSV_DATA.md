# CSV Data Processing for Loose-Inertial-Poser

This guide explains how to use the custom CSV data processing pipeline with the Loose-Inertial-Poser project.

## Data Format

The CSV data should be organized as follows in the `original` folder:
- Multiple CSV files (data1.csv, data2.csv, etc.)
- Each file contains multiple rows of sensor data
- Each row contains:
  - 7 IMU nodes with data in the order: [x-acc, y-acc, z-acc, x-gyro, y-gyro, z-gyro, x-mag, y-mag, z-mag, q0, q1, q2, q3] for each node
  - 16 foot pressure sensor values (8 for left foot, 8 for right foot)
  - 1 label value at the end

## Setup

1. Place your CSV data files in the `Loose-Inertial-Poser-main/original` directory
2. Install required dependencies:
   ```bash
   python Loose-Inertial-Poser-main/install_deps.py
   ```
   This script will install all necessary packages including `numpy-quaternion` which is required by the project.

## Processing the Data

The data processing pipeline has three main components:

1. **my_data_csv.py**: Contains the `CSVData` class that inherits from `BaseDataset` and provides methods to load and process CSV data.

2. **preprocess_csv_data.py**: A script that uses the `CSVData` class to preprocess the CSV data and save it in the format required by the Loose-Inertial-Poser model.

3. **train_poser_csv.py**: A script that trains the Loose-Inertial-Poser model on the preprocessed CSV data.

### Running the Pipeline

You have two options:

#### Option 1: Run the preprocessing and training separately

```bash
# First, preprocess the CSV data
python Loose-Inertial-Poser-main/preprocess_csv_data.py

# Then, train the model on the preprocessed data
python Loose-Inertial-Poser-main/train_poser_csv.py
```

#### Option 2: Run the training script directly

The training script will automatically run the preprocessing if the processed data doesn't exist:

```bash
python Loose-Inertial-Poser-main/train_poser_csv.py
```

## Understanding Data Processing

The CSV data processing pipeline does the following:

1. Reads the CSV files from the `original` folder
2. Extracts IMU data (acceleration, gyroscope, magnetometer, and quaternion)
3. Converts quaternions to rotation matrices
4. Processes foot pressure data
5. Extracts labels
6. Formats the data to match the expected input format for the Loose-Inertial-Poser model
7. Saves the processed data in the `processed_csv` folder as PyTorch tensor files (.pt)

## Training with Custom Data

The training script (`train_poser_csv.py`) automatically:

1. Loads the processed data (or runs preprocessing if needed)
2. Splits the data into training and testing sets (80/20)
3. Creates dataset objects
4. Sets up the network architecture based on the data dimensions
5. Initializes the model, optimizer, and loss function
6. Trains the model for 10 epochs, saving checkpoints after each epoch

## Output

The trained model checkpoints will be saved in the `checkpoint` folder with the prefix `CSV_`.

Training logs will be saved in the `log` folder as `CSV.xlsx`.

## Customizing the Pipeline

You can modify the following parameters in the scripts:

- **my_data_csv.py**: Data loading and processing parameters
- **preprocess_csv_data.py**: Input/output paths, pose representation, acceleration scaling
- **train_poser_csv.py**: Sequence length, network architecture, learning rate, batch size, number of epochs

## Troubleshooting

### Common Issues

1. **Missing quaternion module**:
   Error: `ModuleNotFoundError: No module named 'quaternion'`
   Solution: Run the dependency installation script:
   ```bash
   python Loose-Inertial-Poser-main/install_deps.py
   ```

2. **CSV encoding issues**:
   If you encounter encoding issues with the CSV files, try modifying the encoding parameter in `my_data_csv.py`. The script currently attempts to read with 'utf-8' encoding first, then falls back to 'gbk' if that fails.

3. **Different CSV formats**:
   If some of your CSV files have different column structures, you may need to adjust the data extraction logic in the `load_data` method of the `CSVData` class. 