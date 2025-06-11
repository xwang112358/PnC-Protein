#!/usr/bin/env python3
"""
Debug script to understand what's happening with proteinshake node attributes
"""

import sys
sys.path.append('.')

import torch

def debug_proteinshake():
    print("Starting debug...")
    
    try:
        from utils.proteinshake_data import my_transform
        from proteinshake.tasks import EnzymeClassTask
        from utils.attributes import AttrMapping
        
        print("Imports successful")
        
        # Load enzyme classification task
        task = EnzymeClassTask()
        print("Task loaded")
        
        # Convert proteins to graphs with spatial proximity (eps=6Å, k=36 neighbors)
        dataset = task.dataset.to_graph(eps=6, k=36).pyg(transform=my_transform)
        print("Dataset loaded")
        
        # Get one sample from validation set
        sample_idx = task.val_index[0]
        sample = dataset[sample_idx]
        
        print(f"Sample graph: {sample}")
        print(f"Node features shape: {sample.x.shape}")
        print(f"Node features dtype: {sample.x.dtype}")
        print(f"First few node features:")
        print(sample.x[:5])
        
        # Check if the data is actually one-hot encoded
        print(f"Sum of each node feature (should be 1.0 for one-hot): {sample.x[:5].sum(dim=1)}")
        print(f"Argmax of each node feature (amino acid indices): {sample.x[:5].argmax(dim=1)}")
        
        # Test attr_mapping.map()
        print("=== Testing attr_mapping.map() ===")
        attr_mapping = AttrMapping("proteinshake", "one_hot", 1, None)
        print(f"AttrMapping node_attr_values: {attr_mapping.node_attr_values}")
        
        # Test the mapping function
        node_attrs, _ = attr_mapping.map(sample.x[:5])
        print(f"Mapped node attributes: {node_attrs}")
        print(f"Mapped node attributes shape: {node_attrs.shape}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_proteinshake()
