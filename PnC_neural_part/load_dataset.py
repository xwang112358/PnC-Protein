import sys
sys.path.append('../')
import os
import numpy as np
import torch
from utils.prepare_data import prepare_dataset, prepape_input_features, prepare_dataloaders
from utils.prepare_dictionary import prepare_dictionary
from utils.isomorphism_modules import prepare_isomorphism_module
from utils.prepare_arguments import prepare_environment_args
from encoding_decoding.environment import CompressionEnvironment


def load_dataset_and_environment(args, device, fold_idx):
    """
    Load and prepare dataset, dictionary, features, and environment for training/testing.
    
    Args:
        args: Dictionary containing all configuration arguments
        device: PyTorch device (CPU or CUDA)
        fold_idx: Current fold index for cross-validation
        
    Returns:
        tuple: (graphs_ptg, in_features_dims_dict, attr_mapping, H_set_gt, 
                environment, loader_train, loader_test, loader_val)
    """
    
    # Construct dataset path
    path = os.path.join(args['root_folder'], args['dataset'], args['dataset_name'])
    
    # Prepare dictionary with initial dictionary atoms (optional)
    if 'motifs' in args['atom_types']:
        # Initial dataset loading for motif-based dictionary
        graphs_ptg = prepare_dataset(path,
                                     args['dataset'],
                                     args['dataset_name'],
                                     directed=args['directed'])
        
        loader_train, _, _ = prepare_dataloaders(args, graphs_ptg, path, fold_idx, False, 0)
        dataset_train = loader_train.dataset
        
        # Determine split folder
        if args['split'] == 'random':
            split_folder = 'split_idx_random_' + str(args['split_seed']) + '/' + str(fold_idx)
        else:
            split_folder = 'split_idx' + '/' + str(fold_idx)
        
    #     Prepare dictionary with motifs
        H_set_gt = prepare_dictionary(args, path=path, graphs_ptg=dataset_train, split_folder=split_folder)
        args['max_dict_size'] = len(H_set_gt)
    else:
        # Prepare dictionary without motifs
        H_set_gt = prepare_dictionary(args)

    # assert False
    
    # Generate/load main dataset with detected/loaded subgraphs
    graphs_ptg = prepare_dataset(path,
                                 args['dataset'],
                                 args['dataset_name'],
                                 directed=args['directed'],
                                 H_set=H_set_gt,
                                 multiprocessing=args['multiprocessing'],
                                 num_processes=args['num_processes'],
                                 candidate_subgraphs=args['candidate_subgraphs'])


    print(20 * '=')
    print(graphs_ptg[0].x)
    print(20 * '=')
    
    # One-hot encoding and other input feature preparations
    graphs_ptg, in_features_dims_dict, attr_mapping = prepape_input_features(args, graphs_ptg, path)
    
    # Prepare and instantiate isomorphism module
    isomorphism_module = prepare_isomorphism_module(args['isomorphism_type'],
                                                    node_attr_dims=None if args['node_attr_encoding'] is None
                                                    else attr_mapping.node_attr_dims,
                                                    edge_attr_dims=None if args['edge_attr_encoding'] is None
                                                    else attr_mapping.edge_attr_dims)
    
    # Prepare compression environment
    environment_args = prepare_environment_args(args,
                                                graphs_ptg,
                                                H_set_gt,
                                                device,
                                                isomorphism_module,
                                                in_features_dims_dict['node_attr_unique_values'],
                                                in_features_dims_dict['edge_attr_unique_values'])
    environment = CompressionEnvironment(**environment_args)
    
    # Print dataset statistics
    print('Num graphs: {}'.format(len(graphs_ptg)))
    print('Max degree: {}'.format(in_features_dims_dict['degree_unique_values'][0] - 1))
    n_mean = np.mean([graph.x.shape[0] for graph in graphs_ptg])
    print('Avg/Max num nodes: {:.2f}, {}'.format(n_mean, environment_args['n_max']))
    
    # Prepare final dataloaders
    loader_train, loader_test, loader_val = prepare_dataloaders(args,
                                                                graphs_ptg,
                                                                path,
                                                                fold_idx,
                                                                args['candidate_subgraphs'],
                                                                len(H_set_gt))
    
    return (graphs_ptg, in_features_dims_dict, attr_mapping, H_set_gt, 
            environment, loader_train, loader_test, loader_val)


def get_dataset_path(args):
    """
    Get the dataset path from arguments.
    
    Args:
        args: Dictionary containing configuration arguments
        
    Returns:
        str: Path to the dataset
    """
    return os.path.join(args['root_folder'], args['dataset'], args['dataset_name'])


def prepare_split_folder(args, fold_idx):
    """
    Prepare the split folder path based on split type.
    
    Args:
        args: Dictionary containing configuration arguments
        fold_idx: Current fold index
        
    Returns:
        str: Split folder path
    """
    if args['split'] == 'random':
        return 'split_idx_random_' + str(args['split_seed']) + '/' + str(fold_idx)
    else:
        return 'split_idx' + '/' + str(fold_idx)


def print_dataset_info(graphs_ptg, in_features_dims_dict, environment_args):
    """
    Print dataset information and statistics.
    
    Args:
        graphs_ptg: List of PyTorch Geometric graphs
        in_features_dims_dict: Dictionary containing feature dimensions
        environment_args: Environment arguments dictionary
    """
    print('Num graphs: {}'.format(len(graphs_ptg)))
    print('Max degree: {}'.format(in_features_dims_dict['degree_unique_values'][0] - 1))
    n_mean = np.mean([graph.x.shape[0] for graph in graphs_ptg])
    print('Avg/Max num nodes: {:.2f}, {}'.format(n_mean, environment_args['n_max']))


def create_sample_args():
    """
    Create a sample arguments dictionary with default values for testing.
    
    Returns:
        dict: Sample arguments dictionary
    """
    args = {
        # Dataset configuration
        'root_folder': '../dataset/',
        'dataset': 'proteinshake',
        'dataset_name': 'PROTEINS',
        'directed': False,
        'fold_idx': [0],
        'split': 'None',
        'split_seed': 0,
        'batch_size': 1,
        'num_workers': 0,
        'seed':0,
        
        # Dictionary and compression settings
        'atom_types': [],  # Empty list means no motifs
        'max_dict_size': 10000,
        'universe_type': 'adaptive',
        'multiprocessing': False,
        'num_processes': 1,
        'candidate_subgraphs': False,
        
        # Encoding settings
        'node_attr_encoding': None,
        'edge_attr_encoding': None,
        'isomorphism_type': 'exact',
        
        # Environment constants
        'precision': None,
        'n_max': None,
        'e_max': None,
        'd_max': None,
        'c_max': None,
        'b_max': None,
        'b_min': None,
        'n_h_max_dict': -1,
        'n_h_min_dict': 1,
        'n_h_max': -1,
        'n_h_min': 1,
        
        # Encoding schemes
        'dictionary_encoding': 'graphs',
        'num_nodes_atom_encoding': 'uniform',
        'num_edges_atom_encoding': 'uniform',
        'adj_matrix_atom_encoding': 'erdos_renyi',
        'dict_subgraphs_encoding': 'multinomial',
        'num_nodes_encoding': 'uniform',
        'num_edges_encoding': 'uniform',
        'adj_matrix_encoding': 'erdos_renyi',
        'cut_encoding': 'joint',
        'cut_size_encoding': 'uniform',
        'cut_edges_encoding': 'erdos_renyi',
        'num_nodes_baseline_encoding': 'uniform',
        'num_edges_baseline_encoding': 'uniform',
        'adj_matrix_baseline_encoding': 'erdos_renyi',
        
        # Other settings
        'ema_coeff': 0.5,
        'shuffle': True,
    }
    
    return args


def main():
    """
    Main function to demonstrate dataset loading with sample arguments.
    """
    print("Loading sample dataset...")
    
    # Create sample arguments
    args = create_sample_args()
    
    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Set fold index
    fold_idx = 0
    
    # try:
        # Load dataset and environment
    print(f"Loading dataset: {args['dataset']}/{args['dataset_name']}")
    print(f"Dataset path: {get_dataset_path(args)}")
    
    (graphs_ptg, in_features_dims_dict, attr_mapping, H_set_gt, 
        environment, loader_train, loader_test, loader_val) = load_dataset_and_environment(args, device, fold_idx)

    print(graphs_ptg[0])
    print('node attribute', graphs_ptg[0].x.shape)
    # print(graphs_ptg[0].x)
    print(type(graphs_ptg))
    

    # input features 

    d_in_node_features = 1 if graphs_ptg[0].x.dim()==1 else graphs_ptg[0].x.shape[1]
    print(f"Input node feature dimensions: {d_in_node_features}")

    # assert False


    
    print("\n" + "="*50)
    print("DATASET LOADING SUCCESSFUL!")
    print("="*50)
    
    # Print detailed information
    print(f"Dictionary size: {len(H_set_gt)}")
    print(f"Feature dimensions: {in_features_dims_dict}")
    
    print(f"\nDataloader sizes:")
    print(f"  - Training loader: {len(loader_train)} batches")
    if loader_test is not None:
        print(f"  - Test loader: {len(loader_test)} batches")
    if loader_val is not None:
        print(f"  - Validation loader: {len(loader_val)} batches")
    
    # Show sample from training loader
    if len(loader_train) > 0:
        sample_batch = next(iter(loader_train))
        print(f"\nSample batch info:")
        print(f"  - Batch type: {type(sample_batch)}")
        if hasattr(sample_batch, 'x'):
            print(f"  - Node features shape: {sample_batch.x.shape}")
        if hasattr(sample_batch, 'edge_index'):
            print(f"  - Edge indices shape: {sample_batch.edge_index.shape}")
        
    # except Exception as e:
    #     print(f"\nError loading dataset: {str(e)}")
    #     print("Please make sure:")
    #     print("1. The dataset path exists")
    #     print("2. All required dependencies are installed")
    #     print("3. The utils modules are accessible")
    #     return False
    
    return True


if __name__ == "__main__":
    main()
