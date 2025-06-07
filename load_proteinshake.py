from proteinshake.datasets import EnzymeCommissionDataset
import torch
from torch_geometric.data import Data


def my_transform(item):
    data, protein_dict = item
    new_data = Data()
    # nodes
    if hasattr(data, 'x') and data.x is not None:
        x = data.x
        x = torch.clamp(x, 0, 19)
        x_onehot = torch.nn.functional.one_hot(x, num_classes=20).float()
        new_data.x = x_onehot
    
    # Copy other attributes from original data
    for attr in ['edge_index', 'edge_attr']:
        if hasattr(data, attr):
            setattr(new_data, attr, getattr(data, attr))
    
    return new_data 



dataset = EnzymeCommissionDataset()
# proteins = dataset.proteins(resolution='residue')
dataset = EnzymeCommissionDataset().to_graph(eps=6,k=36).pyg(transform=my_transform)

print(type(dataset[0]))
print(dataset[0])
print('node features:', dataset[0].x.shape)


# print(dataset[0][0])
# print(dataset[0][1])

