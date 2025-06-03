from sknetwork.topology import are_isomorphic
import networkx as nx
import networkx.algorithms.isomorphism as iso
from utils.conversions import convert_to_csr, convert_to_nx
import timeout_decorator

class WLIsomoprhism:
    """
    Implements Weisfeiler-Lehman isomorphism test for graphs.
    Currently only supports uncolored/unattributed graphs.
    
    Args:
        node_attr_dims: Number of node attribute dimensions (not implemented)
        edge_attr_dims: Number of edge attribute dimensions (not implemented)
    """
    def __init__(self, node_attr_dims=None, edge_attr_dims=None):
        self.node_attr_dims = node_attr_dims
        self.edge_attr_dims = edge_attr_dims
        self.convert_graph = convert_to_csr
        
    def match(self, G1, G2):
        """Tests if two graphs are isomorphic using WL algorithm"""
        return are_isomorphic(G1, G2)

class ExactIsomoprhism:
    """
    Implements exact graph isomorphism test using NetworkX.
    Supports both unattributed and attributed graphs.
    
    Args:
        node_attr_dims: Number of node attribute dimensions 
        edge_attr_dims: Number of edge attribute dimensions
    """
    def __init__(self, node_attr_dims=None, edge_attr_dims=None):
        self.node_attr_dims = node_attr_dims
        self.edge_attr_dims = edge_attr_dims
        self.convert_graph = convert_to_nx
        self.nm = None if node_attr_dims is None else iso.categorical_node_match([str(i) for i in range(node_attr_dims)],
                                                                                 [0 for _ in range(node_attr_dims)])
        self.em = None if edge_attr_dims is None else iso.categorical_edge_match([str(i) for i in range(edge_attr_dims)],
                                                                                 [0 for _ in range(edge_attr_dims)])

    @timeout_decorator.timeout(1)
    def match(self, G1, G2):
        """Tests if two graphs are isomorphic with 1 second timeout"""
        return nx.is_isomorphic(G1, G2, node_match=self.nm, edge_match=self.em)


def prepare_isomorphism_module(isomorphism_type, node_attr_dims=None, edge_attr_dims=None):
    """
    Factory function to create appropriate isomorphism testing module.
    
    Args:
        isomorphism_type: Type of isomorphism test ('WL' or 'exact')
        node_attr_dims: Number of node attribute dimensions
        edge_attr_dims: Number of edge attribute dimensions
        
    Returns:
        Isomorphism testing module instance
        
    Raises:
        NotImplementedError: If WL is requested with attributes
    """
    if isomorphism_type == 'WL':
        if node_attr_dims is not None or edge_attr_dims is not None:
            raise NotImplementedError('colored WL not implemented')
        isomorphism_module = WLIsomoprhism(node_attr_dims=node_attr_dims, edge_attr_dims=edge_attr_dims)
    else:
        isomorphism_module = ExactIsomoprhism(node_attr_dims=node_attr_dims, edge_attr_dims=edge_attr_dims)
    return isomorphism_module
