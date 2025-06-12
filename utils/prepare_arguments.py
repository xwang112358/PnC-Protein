from torch_geometric.utils import degree
import math

def prepare_environment_args(args,
                             graphs_ptg, 
                             dictionary,
                             device,
                             isomorphism_module,
                             node_attr_unique_values,
                             edge_attr_unique_values):
    """
    Prepares environment arguments by processing input args and graph data.
    
    Args:
        args: Dictionary of input arguments
        graphs_ptg: List of PyTorch Geometric graph objects
        dictionary: Dictionary object
        device: PyTorch device
        isomorphism_module: Module for graph isomorphism checking
        node_attr_unique_values: Unique node attribute values
        edge_attr_unique_values: Unique edge attribute values
        
    Returns:
        Dictionary of processed environment arguments
    """
    
    # Set n_max (maximum number of nodes) based on input args and actual max graph size
    if args['n_max'] is not None:
        args['n_max'] = max(args['n_max'], max([graph.graph_size for graph in graphs_ptg]))
    else:
        args['n_max'] = max([graph.graph_size for graph in graphs_ptg])
        
    # Set e_max (maximum number of edges) based on whether graph is directed
    if args['e_max'] is not None:
        if args['directed']:
            args['e_max'] = max(args['e_max'], max([graph.edge_index.shape[1] for graph in graphs_ptg]))
        else:
            args['e_max'] = max(args['e_max'], int(max([graph.edge_index.shape[1] for graph in graphs_ptg])/2))
            
    # Set d_max (maximum degree) for directed/undirected graphs
    if args['d_max'] is not None:
        if args['directed']:
            args['d_max'] = [max(args['d_max'][0], max([int(max(degree(graph.edge_index[0])).item()) for graph in graphs_ptg])),
                             max(args['d_max'][1], max([int(max(degree(graph.edge_index[1])).item()) for graph in graphs_ptg]))]
        else:
            args['d_max'] = [max(args['d_max'], max([int(max(degree(graph.edge_index[0])).item()) for graph in graphs_ptg]))]

    # Set dictionary size bounds
    if args['n_h_max_dict'] is None or args['n_h_max_dict']==-1:
        args['n_h_max_dict'] = int(args['n_max'])
    if args['n_h_min_dict'] is None:
        args['n_h_min_dict'] = 1

    # Set subgraph size bounds
    if args['n_h_max'] is None or args['n_h_max']==-1:
        args['n_h_max'] = int(args['n_max'])
    if args['n_h_min'] is None:
        args['n_h_min'] = 1

    # Set block size bounds
    args['b_max'] = int(args['n_max']) if args['b_max'] is None else max(args['b_max'], int(args['n_max']))
    args['b_min'] = math.ceil(int(args['n_max'])/args['n_h_max']) if args['b_min'] is None \
        else min(args['b_min'], math.ceil(int(args['n_max'])/args['n_h_max']))

    # Create environment args dictionary with selected keys from input args
    environment_args={}
    for key in ['directed',
                'universe_type', 'max_dict_size',
                'dictionary_encoding', 'num_nodes_atom_encoding', 'num_edges_atom_encoding', 'adj_matrix_atom_encoding',
                'dict_subgraphs_encoding',
                'num_nodes_encoding', 'num_edges_encoding', 'adj_matrix_encoding',
                'cut_encoding', 'cut_size_encoding', 'cut_edges_encoding',
                'node_attr_encoding', 'edge_attr_encoding',
                'num_nodes_baseline_encoding', 'num_edges_baseline_encoding', 'adj_matrix_baseline_encoding',
                'precision', 'n_max', 'e_max', 'd_max', 'c_max',
                'n_h_max_dict', 'n_h_min_dict', 'b_min',
                'ema_coeff']:
        environment_args[key] = args[key]
     
    # Add additional environment arguments
    environment_args['dictionary'] = dictionary
    environment_args['device'] = device
    environment_args['isomorphism_module'] = isomorphism_module
    environment_args['node_attr_unique_values'] = node_attr_unique_values
    environment_args['edge_attr_unique_values'] = edge_attr_unique_values
    return environment_args

def prepare_model_args(args, dictionary_size):
    """
    Prepares model arguments by processing input args and dictionary size.
    
    Args:
        args: Dictionary of input arguments
        dictionary_size: Size of the dictionary
        
    Returns:
        Dictionary of processed model arguments
    """
    print('Preparing model arguments....')
    model_args = {}
    
    # Configure degree feature injection settings
    if args['inject_degrees']:
        model_args['degree_as_tag'] = [args['degree_as_tag'] for _ in range(args['num_layers'])]
    else:
        model_args['degree_as_tag'] = [args['degree_as_tag']] + [False for _ in range(args['num_layers'] - 1)]
        
    # Configure feature retention settings
    model_args['retain_features'] = [args['retain_features']] + [True for _ in range(args['num_layers'] - 1)]
    
    # Set embedding dimensions for nodes, edges and degrees
    if args['d_out_node_embedding'] is None:
        model_args['d_out_node_embedding'] = args['d_out']
    if args['d_out_edge_embedding'] is None:
        model_args['d_out_edge_embedding'] = [args['d_out'] for _ in range(args['num_layers'])]
    else:
        model_args['d_out_edge_embedding'] = [args['d_out_edge_embedding'] for _ in range(args['num_layers'])]
    if args['d_out_degree_embedding'] is None:
        model_args['d_out_degree_embedding'] = args['d_out']
        
    # Configure message passing dimensions
    if args['d_msg'] == -1:
        model_args['d_msg'] = [None for _ in range(args['num_layers'])]
    elif args['d_msg'] is None:
        model_args['d_msg'] = [args['d_out'] for _ in range(args['num_layers'])]
    else:
        model_args['d_msg'] = [args['d_msg'] for _ in range(args['num_layers'])]
        
    # Configure MLP hidden dimensions
    if args['d_h'] is None:
        model_args['d_h'] = [[args['d_out']] * (args['num_mlp_layers'] - 1) for _ in range(args['num_layers'])]
    else:
        model_args['d_h'] = [[args['d_h']] * (args['num_mlp_layers'] - 1) for _ in range(args['num_layers'])]
        
    # Set training parameters per layer
    model_args['train_eps'] = [args['train_eps'] for _ in range(args['num_layers'])]
    model_args['bn'] = [args['bn'] for _ in range(args['num_layers'])]
    if len(args['final_projection']) == 1:
        model_args['final_projection'] = [args['final_projection'][0] for _ in range(args['num_layers'])] + [True]
    model_args['dropout'] = [args['dropout'] for _ in range(args['num_layers'])] + [args['dropout']]
    
    # Configure output dimensions based on partitioning algorithm
    if not args['candidate_subgraphs']:
        if args['partitioning_algorithm'] == 'subgraph_selection':
            # Categorical on possible subgraph sizes + categorical on vertices
            model_args['out_graph_features'] = args['n_h_max'] - args['n_h_min'] + 1 \
                if (args['n_h_max'] is not None) and (args['n_h_min'] is not None) else None
            model_args['out_node_features'] = 1
        elif args['partitioning_algorithm'] == 'subgraph_selection_w_termination':
            # Bernoulli (continue/terminate) + categorical on vertices
            model_args['out_graph_features'] = 1
            model_args['out_node_features'] = 1
        elif args['partitioning_algorithm'] =='contraction':
            # Categorical on clusters + categorical on edges
            model_args['out_graph_features'] = args['n_h_max'] - args['n_h_min'] + 1 \
                if (args['n_h_max'] is not None) and (args['n_h_min'] is not None) else None
            model_args['out_edge_features'] = 1
            model_args['directed'] = args['directed']
        elif args['partitioning_algorithm'] =='clustering':
            # Categorical on clusters + latent space clustering
            model_args['out_graph_features'] = args['n_h_max'] - args['n_h_min'] + 1 \
                if (args['n_h_max'] is not None) and (args['n_h_min'] is not None) else None
            model_args['out_node_features'] = args['d_out']
            model_args['clustering_iters'] = args['clustering_iters']
            model_args['clustering_temp'] =  args['clustering_temp']
        else:
            raise NotImplementedError('partitioning algorithm {} not implemented'.format(args['partitioning_algorithm']))
    else:
        # Configure for independent set problem with iterative subgraph sampling
        model_args['out_subgraph_features'] = 1
        model_args['dictionary_size'] = dictionary_size
        if args['degree_as_tag_pool'] is None:
            model_args['degree_as_tag_pool'] = args['degree_as_tag'][0]
        if args['retain_features_pool'] is None:
            model_args['retain_features_pool'] = args['retain_features'][0]
        if args['f_fn_type'] is None:
            model_args['f_fn_type'] = 'general'
        if args['phi_fn_type'] is None:
            model_args['phi_fn_type'] = 'general'
        if args['f_d_out'] is None:
            model_args['f_d_out'] = args['d_out']
        if args['d_h_pool'] is None:
            model_args['d_h_pool'] = args['d_h'][0]
        if args['aggr_pool'] is None:
            model_args['aggr_pool'] = args['aggr']
            
    # Set output dimension for each layer
    model_args['d_out'] = [args['d_out'] for _ in range(args['num_layers'])]
    
    # Copy remaining arguments
    for key in ['input_node_embedding', 'edge_embedding', 'degree_embedding',
                'inject_edge_features', 'multi_embedding_aggr',
                'aggr', 'flow', 'extend_dims',
                'activation_mlp', 'bn_mlp', 'activation',
                'model_name','aggr_fn', 'final_projection_layer','readout']:
        model_args[key] = args[key]

    return model_args
