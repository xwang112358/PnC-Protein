from proteinshake.datasets import EnzymeCommissionDataset
from proteinshake.tasks import EnzymeClassTask
import torch
from torch_geometric.data import Data


def my_transform(item):
    data, protein_dict = item
    new_data = Data()
    
    # nodes - one-hot encoding
    if hasattr(data, 'x') and data.x is not None:
        x = data.x
        x = torch.clamp(x, 0, 19)
        x_onehot = torch.nn.functional.one_hot(x, num_classes=20).float()
        new_data.x = x_onehot
    
    # Copy edge information
    if hasattr(data, 'edge_index'):
        new_data.edge_index = data.edge_index
        
    # if hasattr(data, 'edge_attr'):
    #     new_data.edge_attr = data.edge_attr
    
    # Add graph statistics
    num_nodes = new_data.x.shape[0]
    num_edges = new_data.edge_index.shape[1] if hasattr(new_data, 'edge_index') else 0
    
    new_data.graph_size = float(num_nodes)
    new_data.edge_size = float(num_edges)  
    
    # Calculate node degrees
    if hasattr(new_data, 'edge_index'):
        from torch_geometric.utils import degree
        row, col = new_data.edge_index
        degrees = degree(row, num_nodes=num_nodes, dtype=torch.long)
        new_data.degrees = degrees
    else:
        new_data.degrees = torch.zeros(num_nodes, dtype=torch.long)
    
    return new_data 


# --------- Loading the dataset and transforming it ---------

# dataset = EnzymeCommissionDataset()
# dataset = EnzymeCommissionDataset().to_graph(eps=6,k=36).pyg(transform=my_transform)
# print(type(dataset[0]))
# print(dataset[0])


# --------- Loading the task and splitting it ---------
task = EnzymeClassTask()
print('num classes:', task.num_classes)
dataset = task.dataset.to_graph(eps=6,k=36).pyg(transform=my_transform)
train_data, val_data, test_data = dataset[task.train_index], dataset[task.val_index], dataset[task.test_index]
print(f"Train data size: {len(train_data)}")
print(f"Validation data size: {len(val_data)}")
print(f"Test data size: {len(test_data)}")

print(dataset[0])
