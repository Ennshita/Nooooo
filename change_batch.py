#!/usr/bin/env python3
"""
Script to scale batch sizes and learning rates in model and dataloader configuration files.
Scales validation batch size and learning rate parameters linearly based on new training batch size.
"""

import yaml
import argparse
import os
from pathlib import Path


def load_yaml_file(filepath):
    """Load YAML file with safe loading."""
    try:
        with open(filepath, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: File {filepath} not found.")
        return None
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file {filepath}: {e}")
        return None


def save_yaml_file(data, filepath, backup=True):
    """Save data to YAML file with optional backup."""
    if backup and os.path.exists(filepath):
        backup_path = f"{filepath}.backup"
        os.rename(filepath, backup_path)
        print(f"Backup created: {backup_path}")
    
    try:
        with open(filepath, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, indent=2)
        print(f"Updated configuration saved to: {filepath}")
    except Exception as e:
        print(f"Error saving file {filepath}: {e}")


def scale_learning_rates(optimizer_config, scale_factor):
    """Scale learning rate parameters in optimizer configuration."""
    if 'lr' in optimizer_config:
        old_lr = optimizer_config['lr']
        optimizer_config['lr'] = old_lr * scale_factor
        print(f"Scaled main lr: {old_lr} -> {optimizer_config['lr']}")
    
    if 'params' in optimizer_config:
        for param_group in optimizer_config['params']:
            if 'lr' in param_group:
                old_lr = param_group['lr']
                param_group['lr'] = old_lr * scale_factor
                print(f"Scaled param group lr: {old_lr} -> {param_group['lr']}")


def update_model_config(model_config_path, scale_factor, no_backup=False):
    """Update model configuration with scaled learning rates."""
    model_config = load_yaml_file(model_config_path)
    if model_config is None:
        return False
    
    # Scale learning rates in optimizer
    if 'optimizer' in model_config:
        print("Scaling learning rates in model config:")
        scale_learning_rates(model_config['optimizer'], scale_factor)
    else:
        print("No optimizer section found in model config")
    
    # Save updated model config
    save_yaml_file(model_config, model_config_path, backup=not no_backup)
    return True


def update_dataloader_config(dataloader_config_path, new_train_batch_size, scale_factor, no_backup=False):
    """Update dataloader configuration with new batch sizes."""
    dataloader_config = load_yaml_file(dataloader_config_path)
    if dataloader_config is None:
        return False
    
    # Get current batch sizes
    current_train_batch_size = None
    current_val_batch_size = None
    
    if 'train_dataloader' in dataloader_config:
        current_train_batch_size = dataloader_config['train_dataloader'].get('total_batch_size')
        dataloader_config['train_dataloader']['total_batch_size'] = new_train_batch_size
        print(f"Updated train batch size: {current_train_batch_size} -> {new_train_batch_size}")
    
    if 'val_dataloader' in dataloader_config:
        current_val_batch_size = dataloader_config['val_dataloader'].get('total_batch_size')
        if current_val_batch_size is not None:
            new_val_batch_size = int(current_val_batch_size * scale_factor)
            dataloader_config['val_dataloader']['total_batch_size'] = new_val_batch_size
            print(f"Updated val batch size: {current_val_batch_size} -> {new_val_batch_size}")
    
    # Save updated dataloader config
    save_yaml_file(dataloader_config, dataloader_config_path, backup=not no_backup)
    return current_train_batch_size


def main():
    parser = argparse.ArgumentParser(description='Scale batch sizes and learning rates in separate model and dataloader configuration files')
    parser.add_argument('model_config', help='Path to model configuration YAML file')
    parser.add_argument('dataloader_config', help='Path to dataloader configuration YAML file')
    parser.add_argument('new_train_batch_size', type=int, help='New training batch size')
    parser.add_argument('--no-backup', action='store_true', help='Do not create backup files')
    
    args = parser.parse_args()
    
    # Validate files exist
    if not os.path.exists(args.model_config):
        print(f"Error: Model config file {args.model_config} not found.")
        return 1
    
    if not os.path.exists(args.dataloader_config):
        print(f"Error: Dataloader config file {args.dataloader_config} not found.")
        return 1
    
    # Load dataloader config to get current train batch size
    dataloader_config = load_yaml_file(args.dataloader_config)
    if dataloader_config is None:
        return 1
    
    current_train_batch_size = None
    if 'train_dataloader' in dataloader_config:
        current_train_batch_size = dataloader_config['train_dataloader'].get('total_batch_size')
    
    if current_train_batch_size is None:
        print("Error: Could not find train_dataloader.total_batch_size in dataloader config")
        return 1
    
    # Calculate scaling factor
    scale_factor = args.new_train_batch_size / current_train_batch_size
    
    print(f"Model config: {args.model_config}")
    print(f"Dataloader config: {args.dataloader_config}")
    print(f"Current train batch size: {current_train_batch_size}")
    print(f"New train batch size: {args.new_train_batch_size}")
    print(f"Scaling factor: {scale_factor:.6f}")
    print("=" * 60)
    
    # Update model config
    print("Updating model configuration...")
    success = update_model_config(args.model_config, scale_factor, args.no_backup)
    if not success:
        return 1
    
    print("-" * 40)
    
    # Update dataloader config
    print("Updating dataloader configuration...")
    current_batch = update_dataloader_config(args.dataloader_config, args.new_train_batch_size, scale_factor, args.no_backup)
    if current_batch is False:
        return 1
    
    print("=" * 60)
    print("Configuration scaling completed successfully!")
    return 0


if __name__ == "__main__":
    exit(main())