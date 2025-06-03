# Standard PyTorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F

# Custom GNN layer imports
from gnn_layers.MPNN_sparse import MPNN_sparse
from gnn_layers.MPNN_edge_sparse import MPNN_edge_sparse
from gnn_layers.MPNN import MPNN
from gnn_layers.MPNN_edge import MPNN_edge

# Utility imports for neural network layers and pooling operations
from utils.nn_layers import mlp, choose_activation, DiscreteEmbedding, global_add_pool_sparse, global_mean_pool_sparse
from torch_geometric.nn import global_add_pool, global_mean_pool

# Sets of supported GNN layer types
pyg_layers = {'MPNN', 'MPNN_edge'}
sparse_layers = {'MPNN_sparse', 'MPNN_edge_sparse'}

        
class NeuralPartGNN(torch.nn.Module):
    """
    Neural network model for graph partitioning that uses message passing neural networks (GNNs).
    Processes both node and edge features to learn graph representations.
    """

    def __init__(self,
                 d_in_node_features,
                 d_in_node_embedding=None,
                 d_in_edge_features=None, 
                 d_in_edge_embedding=None,
                 d_in_degree_embedding=None,
                 **kwargs):
        """
        Initialize the NeuralPartGNN model.
        
        Args:
            d_in_node_features: Dimension of input node features
            d_in_node_embedding: Dimension of node embedding for discrete inputs
            d_in_edge_features: Dimension of input edge features  
            d_in_edge_embedding: Dimension of edge embedding for discrete inputs
            d_in_degree_embedding: Dimension of degree embedding
            **kwargs: Additional model configuration parameters
        """

        super(NeuralPartGNN, self).__init__()
        # Extract model configuration from kwargs
        self.model_name = kwargs['model_name']
        input_node_embedding = kwargs['input_node_embedding']
        d_out_node_embedding = kwargs['d_out_node_embedding']
        edge_embedding = kwargs['edge_embedding']
        d_out_edge_embedding = kwargs['d_out_edge_embedding']
        self.inject_edge_features = kwargs['inject_edge_features']
        degree_embedding = kwargs['degree_embedding'] if kwargs['degree_as_tag'][0] else 'None'
        d_out_degree_embedding = kwargs['d_out_degree_embedding']
        
        # Message passing configuration
        d_msg = kwargs['d_msg']
        d_out = kwargs['d_out']
        d_h = kwargs['d_h']
        aggr = kwargs['aggr'] if kwargs['aggr'] is not None else 'add'
        flow = kwargs['flow'] if kwargs['flow'] is not None else 'target_to_source'
        aggr_fn = kwargs['aggr_fn'] if kwargs['aggr_fn'] is not None else 'general'
        train_eps = kwargs['train_eps'] if kwargs['train_eps'] is not None else [False for _ in range(len(d_out))]
        activation_mlp = kwargs['activation_mlp']
        bn_mlp = kwargs['bn_mlp']
        degree_as_tag = kwargs['degree_as_tag'] 
        retain_features = kwargs['retain_features']
        extend_dims = kwargs['extend_dims']
        
        # Layer configuration between and after message passing
        self.bn = kwargs['bn']
        self.activation = choose_activation(kwargs['activation'])
        self.dropout = kwargs['dropout']
        self.final_projection = kwargs['final_projection']
        self.final_projection_layer = kwargs['final_projection_layer']
        out_node_features = kwargs['out_node_features']
        out_graph_features = kwargs['out_graph_features']
        self.readout = kwargs['readout']
        
        # Common arguments for embedding layers
        embeddings_kwargs = {'activation_mlp': activation_mlp,
                           'bn_mlp': bn_mlp,
                           'aggr': kwargs['multi_embedding_aggr']}
                           
        #-------------- Input node embedding initialization
        self.input_node_embedding = DiscreteEmbedding(input_node_embedding, 
                                                    d_in_node_features,
                                                    d_in_node_embedding,
                                                    d_out_node_embedding,
                                                    **embeddings_kwargs)
        d_in = self.input_node_embedding.d_out
        
        #-------------- Edge embedding initialization
        self.edge_embedding = []
        d_ef = []
        num_edge_embeddings = len(d_out) if self.inject_edge_features else 1
        for i in range(num_edge_embeddings):
            edge_embedding_layer = DiscreteEmbedding(edge_embedding, 
                                                   d_in_edge_features,
                                                   d_in_edge_embedding,
                                                   d_out_edge_embedding[i],
                                                   **embeddings_kwargs)
            self.edge_embedding.append(edge_embedding_layer)
            d_ef.append(edge_embedding_layer.d_out)
        self.edge_embedding  = nn.ModuleList(self.edge_embedding)
        
        #-------------- Degree embedding initialization
        self.degree_embedding = DiscreteEmbedding(degree_embedding,
                                                1,
                                                d_in_degree_embedding,
                                                d_out_degree_embedding,
                                                **embeddings_kwargs)
        d_degree = self.degree_embedding.d_out
        
        #-------------- GNN layers initialization
        self.conv = []
        self.batch_norms = []
        self.out_node_layers = []
        self.out_graph_layers = []
        
        # Initialize each GNN layer with its corresponding components
        for i in range(len(d_out)):
            out_node_layer = None
            out_graph_layer = None
            
            # Setup output projection layers if specified
            if self.final_projection[i]:
                if self.final_projection_layer == 'mlp':
                    out_node_layer = mlp(d_in, d_out[-1], d_h[i], activation_mlp, bn_mlp)
                    out_graph_layer = mlp(d_in, d_out[-1], d_h[i], activation_mlp, bn_mlp)
                elif self.final_projection_layer == 'linear':
                    out_node_layer = nn.Linear(d_in, d_out[-1])
                    out_graph_layer = nn.Linear(d_in, d_out[-1])
                else:
                    raise NotImplementedError("output projection layer {} is not currently supported.".
                                              format(self.final_projection_layer))
                                              
            self.out_node_layers.append(out_node_layer)
            self.out_graph_layers.append(out_graph_layer)
            
            # Configure layer-specific parameters
            kwargs_layer = {
                 'd_in': d_in,
                 'd_degree': d_degree,
                 'degree_as_tag': degree_as_tag[i],
                 'retain_features': retain_features[i],
                 'd_msg': d_msg[i],
                 'd_up': d_out[i],
                 'd_h': d_h[i],
                 'd_ef': d_ef[i] if self.inject_edge_features else d_ef[0],
                 'activation_name': activation_mlp,
                 'bn': bn_mlp,
                 'aggr': aggr,
                 'aggr_fn': aggr_fn,
                 'eps': 0,
                 'train_eps': train_eps[i],
                 'flow': flow,
                 'edge_embedding': edge_embedding,
                 'extend_dims': extend_dims}
                 
            # Determine whether to use edge features
            use_efs = ((i > 0 and self.inject_edge_features) or (i == 0)) and\
                      (self.model_name in {'MPNN_edge', 'MPNN_edge_sparse'})
                      
            # Select appropriate GNN layer type
            if self.model_name in pyg_layers:
                layer = MPNN_edge if use_efs else MPNN
            elif self.model_name in sparse_layers:
                layer = MPNN_edge_sparse if use_efs else MPNN_sparse
            else:
                raise NotImplementedError("GNN layer {} is not currently supported.".format(self.model_name))
                
            self.conv.append(layer(**kwargs_layer))
            bn_layer = nn.BatchNorm1d(d_out[i]) if self.bn[i] else None
            self.batch_norms.append(bn_layer)
            d_in = d_out[i]
            
        # Initialize final output layers
        if self.final_projection[-1]:
            if self.final_projection_layer == 'mlp':
                final_out_node_layer = mlp(d_in,  d_out[-1], d_h[-1], activation_mlp, bn_mlp)
                final_out_graph_layer = mlp(d_in,  d_out[-1], d_h[-1], activation_mlp, bn_mlp)
            elif self.final_projection_layer == 'linear':
                final_out_node_layer = nn.Linear(d_in, d_out[-1])
                final_out_graph_layer = nn.Linear(d_in, d_out[-1])
            else:
                raise NotImplementedError("output projection layer {} is not currently supported.".
                                          format(self.final_projection_layer))
        else:
            final_out_node_layer, final_out_graph_layer = None, None
            
        # Finalize layer lists
        self.out_node_layers.append(final_out_node_layer)
        self.out_node_layers = nn.ModuleList(self.out_node_layers)
        self.out_graph_layers.append(final_out_graph_layer)
        self.out_graph_layers = nn.ModuleList(self.out_graph_layers)
        self.conv = nn.ModuleList(self.conv)
        self.batch_norms = nn.ModuleList(self.batch_norms)
        
        #-------------- Readout layer initialization
        if self.readout == 'sum':
            self.global_pool = global_add_pool_sparse if self.model_name in sparse_layers else global_add_pool
        elif self.readout == 'mean':
            self.global_pool = global_mean_pool_sparse if self.model_name in sparse_layers else global_mean_pool
        else:
            raise ValueError("Invalid graph pooling type.")
            
        # Initialize context and output transformation layers
        if self.final_projection_layer == 'mlp':
            self.context_f = mlp(d_out[-1], d_out[-1], d_h[-1], activation_mlp, bn_mlp)
            self.context_phi = mlp(d_out[-1], d_out[-1], d_h[-1], activation_mlp, bn_mlp)
            self.global_context_update = mlp(d_out[-1], d_out[-1], d_h[-1], activation_mlp, bn_mlp)
            self.node_lvl_phi = mlp(2 * d_out[-1], out_node_features, d_h[-1], activation_mlp, bn_mlp)
            self.graph_lvl_phi = mlp(2 * d_out[-1], out_graph_features, d_h[-1], activation_mlp, bn_mlp)
        elif self.final_projection_layer == 'linear':
            self.context_f = nn.Linear(d_out[-1], d_out[-1])
            self.context_phi = nn.Linear(d_out[-1], d_out[-1])
            self.global_context_update = nn.Linear(d_out[-1], d_out[-1])
            self.node_lvl_phi = nn.Linear(2 * d_out[-1], out_node_features)
            self.graph_lvl_phi = nn.Linear(2 * d_out[-1] , out_graph_features)
        self.global_context_feat = d_out[-1]
        return

    def forward(self, data):
        """
        Forward pass of the model.
        
        Args:
            data: Input graph data object containing node features, edge indices, etc.
            
        Returns:
            tuple: (node_features, graph_features) - Final node and graph level representations
        """
        kwargs = {}
        kwargs['degrees'] = self.degree_embedding(data.degrees)
        x = self.input_node_embedding(data.x)
        x_interm = [x]
        edge_index = data.edge_index
        
        # Process through GNN layers
        for i in range(0, len(self.conv)):
            # Handle edge features if present
            if hasattr(data, 'edge_features'):
                kwargs['edge_features'] = self.edge_embedding[i](data.edge_features) \
                    if self.inject_edge_features else self.edge_embedding[0](data.edge_features)
            else:
                kwargs['edge_features'] = None
                
            # Apply convolution, batch norm, and activation
            x = self.conv[i](x, edge_index, **kwargs)
            if self.bn[i]:
                x = self.batch_norms[i](x)
            x = self.activation(x)
            x_interm.append(x)
            
        # Compute final node representations
        x_node_final = 0
        for i in range(0, len(self.conv) + 1):
            if self.final_projection[i]:
                x_node_final += F.dropout(self.out_node_layers[i](x_interm[i]),
                                          p=self.dropout[i], training=self.training)
                                          
        # Compute final graph representations
        x_graph_final = 0
        for i in range(0, len(self.conv) + 1):
            if self.final_projection[i]:
                x_graph_temp = self.global_pool(x_interm[i], data.batch)
                x_graph_final += F.dropout(self.out_graph_layers[i](x_graph_temp),
                                           p=self.dropout[i], training=self.training)
        return x_node_final, x_graph_final


    def aggregate_context(self, data, h_nodes, subgraph_nodes, global_context=None):
        """
        Aggregate context information from nodes in the graph.
        
        Args:
            data: Input graph data
            h_nodes: Node hidden representations
            subgraph_nodes: Nodes to consider for context
            global_context: Existing global context to update (optional)
            
        Returns:
            tuple: (global_context, updated_global_context)
        """
        num_nodes = h_nodes.shape[0]
        h_nodes_context = torch.zeros((num_nodes, self.global_context_feat), device=h_nodes.device)
        h_nodes_context[subgraph_nodes] = self.context_f(h_nodes[subgraph_nodes])
        context_t = self.context_phi(self.global_pool(h_nodes_context, data.batch))
        global_context = context_t if global_context is None else global_context + context_t
        global_context_updated = self.global_context_update(global_context)
        return global_context, global_context_updated

    def predict_graph(self, h_graph, global_context_updated):
        """
        Generate graph-level predictions using graph representations and context.
        
        Args:
            h_graph: Graph representations
            global_context_updated: Updated global context
            
        Returns:
            torch.Tensor: Graph-level predictions
        """
        x_graph = self.graph_lvl_phi(torch.cat((global_context_updated, h_graph), 1))
        return x_graph

    def predict_node(self, h_nodes, global_context_updated, batch):
        """
        Generate node-level predictions using node representations and context.
        
        Args:
            h_nodes: Node representations
            global_context_updated: Updated global context
            batch: Batch assignments for nodes
            
        Returns:
            torch.Tensor: Node-level predictions
        """
        x_nodes = self.node_lvl_phi(torch.cat((global_context_updated[batch], h_nodes), 1))
        return x_nodes