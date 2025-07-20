#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cognitive Augmentation System - Core Model
Four cognitive decline simulation methods:
1. Motor Response Delay
2. Gait Signal Perturbation  
3. Sensor Drift Simulation
4. Latent Space Cognitive Decline Vector
"""

import numpy as np
from scipy import signal
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional, Union

class CognitiveAugmentationSystem:
    """
    Cognitive Augmentation System
    
    Implements four cognitive decline simulation methods:
    1. Motor Response Delay - Simulates cognitive reaction delays
    2. Gait Signal Perturbation - Simulates gait instability
    3. Sensor Drift Simulation - Simulates sensor aging
    4. Latent Space Cognitive Decline Vector - Simulates complex cognitive decline
    """
    
    def __init__(self, random_seed=42):
        """Initialize cognitive augmentation system"""
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
        # Augmentation methods
        self.augmentation_methods = {
            'motor_delay': self._motor_response_delay,
            'gait_perturbation': self._gait_signal_perturbation,
            'sensor_drift': self._sensor_drift_simulation,
            'cognitive_vector': self._cognitive_decline_vector
        }
        
        print("Cognitive Augmentation System initialized")
        print("Available augmentation methods:")
        for i, method in enumerate(self.augmentation_methods.keys(), 1):
            print(f"  {i}. {method}")
    
    def _motor_response_delay(self, data: np.ndarray, **kwargs) -> np.ndarray:
        """
        Method 1: Motor response delay simulation
        Formula: x'(t) = α·x(t-δ) + (1-α)·x(t)
        where δ∈[3,10] samples, α∈[0.7,0.95]
        """
        delay_range = kwargs.get('delay_range', [3, 10])
        alpha_range = kwargs.get('alpha_range', [0.7, 0.95])
        apply_to_channels = kwargs.get('apply_to_channels', None)
        
        augmented_data = data.copy()
        n_samples, n_features = data.shape
        
        # Determine channels to process
        if apply_to_channels is None:
            channels = range(n_features)
        else:
            channels = apply_to_channels
        
        for channel in channels:
            # Random delay and mixing coefficient
            delta = np.random.randint(delay_range[0], delay_range[1] + 1)
            alpha = np.random.uniform(alpha_range[0], alpha_range[1])
            
            # Create delayed signal
            delayed_signal = np.zeros_like(data[:, channel])
            delayed_signal[delta:] = data[:-delta, channel]
            delayed_signal[:delta] = data[0, channel]  # Fill with first value
            
            # Apply delay mixing formula
            augmented_data[:, channel] = alpha * delayed_signal + (1 - alpha) * data[:, channel]
        
        return augmented_data
    
    def _gait_signal_perturbation(self, data: np.ndarray, **kwargs) -> np.ndarray:
        """
        Method 2: Gait signal perturbation
        Includes:
        - Plantar pressure intensity mutations
        - Left-right foot pseudo-synchronization
        - DTW time warping
        """
        perturbation_intensity = kwargs.get('perturbation_intensity', 0.3)
        pressure_channels = kwargs.get('pressure_channels', None)
        sync_probability = kwargs.get('sync_probability', 0.2)
        
        augmented_data = data.copy()
        n_samples, n_features = data.shape
        
        # 1. Plantar pressure intensity mutations
        if pressure_channels is not None:
            for channel in pressure_channels:
                # Random mutation points
                n_perturbations = np.random.randint(1, 5)
                for _ in range(n_perturbations):
                    start_idx = np.random.randint(0, n_samples - 10)
                    end_idx = min(start_idx + np.random.randint(5, 15), n_samples)
                    
                    # Apply mutation
                    perturbation = np.random.uniform(0.5, 1.5) * perturbation_intensity
                    augmented_data[start_idx:end_idx, channel] *= perturbation
        
        # 2. Left-right foot pseudo-synchronization
        if np.random.random() < sync_probability:
            mid_point = n_features // 2
            left_channels = range(0, mid_point)
            right_channels = range(mid_point, n_features)
            
            # Random synchronization segment
            sync_start = np.random.randint(0, n_samples - 20)
            sync_end = min(sync_start + np.random.randint(10, 30), n_samples)
            
            # Create pseudo-synchronization
            for left_ch, right_ch in zip(left_channels, right_channels):
                if right_ch < n_features:
                    sync_weight = np.random.uniform(0.3, 0.7)
                    augmented_data[sync_start:sync_end, right_ch] = (
                        sync_weight * augmented_data[sync_start:sync_end, left_ch] +
                        (1 - sync_weight) * augmented_data[sync_start:sync_end, right_ch]
                    )
        
        # 3. DTW time warping
        if np.random.random() < 0.3:  # 30% probability
            n_warp_channels = min(3, n_features)
            warp_channels = np.random.choice(n_features, n_warp_channels, replace=False)
            
            for channel in warp_channels:
                # Create time warping path
                warp_strength = np.random.uniform(0.1, 0.3)
                time_indices = np.arange(n_samples)
                warped_indices = time_indices + warp_strength * np.sin(
                    2 * np.pi * time_indices / n_samples * np.random.uniform(1, 3)
                )
                warped_indices = np.clip(warped_indices, 0, n_samples - 1)
                
                # Interpolate to new time points
                f = interp1d(time_indices, data[:, channel], kind='linear', 
                           bounds_error=False, fill_value='extrapolate')
                augmented_data[:, channel] = f(warped_indices)
        
        return augmented_data
    
    def _sensor_drift_simulation(self, data: np.ndarray, **kwargs) -> np.ndarray:
        """
        Method 3: IMU and pressure sensor drift simulation
        Includes:
        - IMU three-axis drift (pitch, roll micro-drift)
        - Plantar pressure distribution adjustment
        """
        imu_channels = kwargs.get('imu_channels', [0, 1, 2])
        pressure_channels = kwargs.get('pressure_channels', None)
        drift_intensity = kwargs.get('drift_intensity', 0.05)
        
        augmented_data = data.copy()
        n_samples, n_features = data.shape
        
        # 1. IMU drift simulation
        for i, channel in enumerate(imu_channels):
            if channel < n_features:
                # Generate progressive drift
                drift_pattern = np.linspace(0, 1, n_samples)
                drift_magnitude = np.random.normal(0, drift_intensity)
                
                # Add periodic drift (simulating temperature drift)
                periodic_drift = drift_intensity * 0.5 * np.sin(
                    2 * np.pi * drift_pattern * np.random.uniform(0.5, 2.0)
                )
                
                # Apply drift
                total_drift = drift_magnitude * drift_pattern + periodic_drift
                augmented_data[:, channel] += total_drift
        
        # 2. Pressure distribution adjustment
        if pressure_channels is not None and len(pressure_channels) >= 4:
            n_pressure = len(pressure_channels)
            front_channels = pressure_channels[:n_pressure//2]
            heel_channels = pressure_channels[n_pressure//2:]
            
            # Simulate pressure center shift
            shift_intensity = np.random.uniform(0.85, 0.95)
            
            for front_ch, heel_ch in zip(front_channels, heel_channels):
                # Transfer part of forefoot pressure to heel
                pressure_shift = augmented_data[:, front_ch] * (1 - shift_intensity)
                augmented_data[:, front_ch] *= shift_intensity
                augmented_data[:, heel_ch] += pressure_shift * 0.7  # Partial energy loss
        
        return augmented_data
    
    def _cognitive_decline_vector(self, data: np.ndarray, **kwargs) -> np.ndarray:
        """
        Method 4: Latent space cognitive decline vector (simplified version)
        Adds structured cognitive decline patterns in feature space
        """
        cognitive_class = kwargs.get('cognitive_class', 1)
        vector_intensity = kwargs.get('vector_intensity', 0.1)
        
        augmented_data = data.copy()
        
        # Generate cognitive decline vector
        cognitive_vector = np.random.randn(*data.shape) * vector_intensity
        
        # Apply different intensities to different feature types
        n_features = data.shape[1]
        imu_channels = list(range(min(6, n_features)))
        pressure_channels = list(range(6, n_features)) if n_features > 6 else []
        
        # IMU features: mild impact
        if imu_channels:
            cognitive_vector[:, imu_channels] *= 0.5
        
        # Pressure features: moderate impact
        if pressure_channels:
            cognitive_vector[:, pressure_channels] *= 0.8
        
        # Add structured decline pattern
        for i in range(0, data.shape[1], 4):
            cognitive_vector[:, i:i+2] *= 1.2  # Enhance certain dimensions
        
        return augmented_data + cognitive_vector
    
    def augment_data(self, data: np.ndarray, method: str, **kwargs) -> np.ndarray:
        """Apply specified augmentation method to data"""
        if method not in self.augmentation_methods:
            raise ValueError(f"Unknown augmentation method: {method}")
        
        return self.augmentation_methods[method](data, **kwargs)
    
    def batch_augment(self, data_list: List[np.ndarray], labels: List[int], 
                     method: str, **kwargs) -> Tuple[List[np.ndarray], List[int]]:
        """Batch data augmentation"""
        augmented_data = []
        augmented_labels = []
        
        print(f"Batch augmenting {len(data_list)} samples using {method} method...")
        
        for i, (data, label) in enumerate(tqdm(zip(data_list, labels))):
            # Original data
            augmented_data.append(data)
            augmented_labels.append(label)
            
            # Augmented data
            aug_data = self.augment_data(data, method, **kwargs)
            augmented_data.append(aug_data)
            augmented_labels.append(label)
        
        print(f"Batch augmentation complete, from {len(data_list)} samples to {len(augmented_data)} samples")
        return augmented_data, augmented_labels
