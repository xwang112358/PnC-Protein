#!/usr/bin/env python3
"""
Example script demonstrating how to collect test partitions after training.

Usage:
    python example_collect_partitions.py --dataset=proteinshake --dataset_name=EnzymeCommissionDataset --collect_partitions=True
"""

import sys
import os
sys.path.append(".")

from PnC_Protein import main
import argparse
import utils.parsing as parse

def run_with_partition_collection():
    """
    Example of running training with test partition collection enabled.
    """
    
    # Example arguments - modify these for your specific use case
    example_args = {
        # Basic setup
        "mode": "train",
        "seed": 42,
        "np_seed": 42,
        "num_threads": 1,
        
        # Dataset
        "dataset": "proteinshake",
        "dataset_name": "EnzymeCommissionDataset", 
        "root_folder": "./datasets",
        "results_folder": "example_results",
        
        # Model 
        "model_name": "MPNN_edge",
        "d_out": [64, 64],
        "d_h": [64, 64],
        
        # Training
        "num_epochs": 5,  # Small number for example
        "batch_size": 32,
        "lr_policy": 0.001,
        "lr_dict": 0.1,
        
        # Compression settings
        "n_h_max_dict": 10,
        "n_h_min_dict": 2,
        "max_dict_size": 100,
        "universe_type": "adaptive",
        
        # Enable partition collection
        "collect_partitions": True,
        
        # Other settings
        "GPU": True,
        "device_idx": 0,
        "wandb": False,  # Disable wandb for this example
        "visualise": False,
        "split": "given",
        "folds": [0],  # Single fold for example
    }
    
    print("Running training with partition collection...")
    print("This will:")
    print("1. Train the model")
    print("2. Collect partitions from test data")
    print("3. Save partitions to JSON files")
    print("4. Generate visualization scripts")
    print()
    
    # Run the main training function
    main(example_args)
    
    print("\nExample complete!")
    print("Check the results folder for partition files and visualization scripts.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Example script for collecting test partitions")
    
    # Allow overriding some key parameters
    parser.add_argument("--dataset", type=str, default="proteinshake", help="Dataset type")
    parser.add_argument("--dataset_name", type=str, default="EnzymeCommissionDataset", help="Dataset name")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--collect_partitions", type=parse.str2bool, default=True, help="Collect test partitions")
    parser.add_argument("--device_idx", type=int, default=0, help="GPU device index")
    
    args = parser.parse_args()
    
    print(f"Running with dataset: {args.dataset}/{args.dataset_name}")
    print(f"Epochs: {args.num_epochs}")
    print(f"Collect partitions: {args.collect_partitions}")
    print()
    
    # You can modify the example_args dictionary based on command line arguments
    run_with_partition_collection()
