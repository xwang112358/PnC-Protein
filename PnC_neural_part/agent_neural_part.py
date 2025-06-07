import torch
from torch.distributions.categorical import Categorical
from utils.utils_subgraphs import induced_subgraph, compute_cut_size_pairs, compute_combinations
from utils.visualisation import visualise_intermediate

class CompressionAgentNeuralPart:
    """
    Neural compression agent that iteratively decomposes graphs into subgraphs using a neural policy.
    
    This agent implements a reinforcement learning approach to graph compression where:
    1. A neural policy network selects subgraphs from the input graph
    2. Selected subgraphs are mapped to a learned dictionary of structural patterns
    3. The process continues until the entire graph is decomposed
    """
    def __init__(self,
                 policy_network,
                 env,
                 train=True,
                 **kwargs):
        """
        Initialize the compression agent.
        
        Args:
            policy_network: Neural network that makes subgraph selection decisions
            env: Environment containing the dictionary and compression logic
            train: Whether the agent is in training mode (affects dictionary updates)
            **kwargs: Additional configuration parameters including:
                - attr_mapping: Mapping for node/edge attributes
                - n_h_max/n_h_min: Maximum/minimum subgraph sizes for selection
                - n_h_max_dict/n_h_min_dict: Dictionary size constraints
                - sampling_mechanism: Method for subgraph sampling
        """
        self.policy = policy_network
        self.env = env
        self.train = train
        self.attr_mapping = kwargs['attr_mapping']
        self.n_h_max = kwargs['n_h_max']
        self.n_h_min = kwargs['n_h_min']
        self.n_h_max_dict = kwargs['n_h_max_dict']
        self.n_h_min_dict = kwargs['n_h_min_dict']
        self.sampling_mechanism = kwargs['sampling_mechanism']
        # self.sample_k = kwargs['sample_k']
        # self.connected_sampling = kwargs['connected_sampling']

    def termination_criterion(self, graph_sizes):
        """
        Check if the compression process should terminate.
        
        Args:
            graph_sizes: Tensor of remaining graph sizes
            
        Returns:
            bool: True if all graphs are fully decomposed (size 0)
        """
        return (graph_sizes == 0).all()

    def compress(self, graph, **kwargs):
        """
        Main compression method that decomposes input graphs into subgraphs.
        
        This method implements the core compression algorithm:
        1. Initialize graph representations using the policy network
        2. Iteratively select subgraphs using the neural policy
        3. Map selected subgraphs to dictionary atoms
        4. Update the remaining graph and continue until termination
        
        Args:
            graph: Input graph data object to compress
            **kwargs: Additional arguments including:
                - visualise: Whether to generate visualizations
                - inds_to_visualise: Indices for visualization
                - x_a: Dictionary selection probabilities (inference mode)
                
        Returns:
            tuple: (subgraphs_dict, log_probs, visualise_data_dict)
                - subgraphs_dict: Dictionary containing decomposition results
                - log_probs: Log probabilities of selected actions
                - visualise_data_dict: Data for visualization
        """
        device = graph.graph_size.device
        # visualisation arguments
        visualise = kwargs['visualise'] if 'visualise' in kwargs else False
        inds_to_visualise = kwargs['inds_to_visualise'] if 'inds_to_visualise' in kwargs else 0
        visualise_data_dict = {}
        
        # train vs inference mode: IMPORTANT
        # In training: use all dictionary atoms; In inference: use selected subset
        if self.train:
            self.policy.train()
            self.query_inds = self.env.sorted_atoms_in_dict()
        else:
            self.policy.eval()
            self.query_inds = self.env.sorted_atoms_in_dict(kwargs['x_a'])
            
        # initialisation of parameters
        init_graph = graph
        node_labels_original = torch.arange(int(init_graph.graph_size.sum()), device=device)
        mask_t = torch.ones((init_graph.x.shape[0],), device=device).bool()
        n_h, e_h, atom_indices = [], [], []  # subgraph sizes and dictionary mappings
        b = (-1) * torch.ones((len(init_graph.graph_size),), dtype=torch.long, device=device)  # decomposition depth
        subgraph_nodes_t, subgraph_nodes_all, log_probs = [], [], []
        
        # check termination criterion
        termination = self.termination_criterion(init_graph.graph_size)
        
        # get initial GNN predictions for all nodes and graphs
        h_0_nodes, h_0_graph = self.policy(init_graph)
        global_context = None
        
        # ----------- iterative subgraph decomposition
        t = 0
        while not termination:
            # update context vectors based on previously selected subgraphs
            global_context, global_context_updated = self.policy.aggregate_context(init_graph,
                                                                                   h_0_nodes,
                                                                                   subgraph_nodes_t,
                                                                                   global_context)
            
            # restrict node features to remaining nodes
            h_0_nodes_restricted = h_0_nodes[node_labels_original]
            
            # select subgraphs using the neural policy
            subgraphs_t, log_probs_t = self.select_subgraph(graph,
                                                            h_0_nodes_restricted,
                                                            h_0_graph,
                                                            global_context_updated,
                                                            self.policy)
            
            # store subgraph information
            n_h.append(subgraphs_t['n_h'])
            e_h.append(subgraphs_t['e_h'])
            atom_indices.append(subgraphs_t['atom_indices'])
            
            # map the nodes to their initial indices for proper tracking
            subgraphs_t_nodes_relabelled = [node_labels_original[subgraphs_t['nodes'][i]]
                                            for i in range(len(subgraphs_t['nodes']))]
            subgraph_nodes_all.append(subgraphs_t_nodes_relabelled)
            subgraph_nodes_t = torch.cat(subgraphs_t_nodes_relabelled)
            log_probs.append(log_probs_t)
            
            # visualise intermediate decomposition steps
            if visualise:
                visualise_data_dict = visualise_intermediate(visualise_data_dict,
                                                             inds_to_visualise,
                                                             init_graph,
                                                             mask_t,
                                                             subgraphs_t_nodes_relabelled,
                                                             subgraphs_t['atom_indices'],
                                                             attr_mapping=self.attr_mapping,
                                                             node_attr_dims=self.env.isomorphism_module.node_attr_dims,
                                                             edge_attr_dims=self.env.isomorphism_module.edge_attr_dims)
                new_mask = torch.ones_like(mask_t)
                new_mask[subgraph_nodes_t] = 0
                mask_t = new_mask & mask_t
                
            # remove the selected subgraph to reduce memory footprint and speed-up computations
            new_graph, remaining_nodes, _ = self.env.step(graph,
                                                          subgraphs_t['nodes'],
                                                          subgraphs_t['n_h'],
                                                          subgraphs_t['e_h'],
                                                          c_s=None,
                                                          relabel_nodes=True,
                                                          candidate_subgraphs=False)

            # check termination criterion and update decomposition depth
            b[(new_graph.graph_size == 0) & (b == -1)] = t + 1
            graph = new_graph
            termination = self.termination_criterion(graph.graph_size)
            
            # keep track of the initial indices for proper node mapping
            node_labels_original = node_labels_original[remaining_nodes]
            if termination:
                break
            else:
                t += 1
                
        # reorganize results: transpose to get (batch_size, num_steps) format
        n_h = torch.stack(n_h).transpose(1, 0)
        e_h = torch.stack(e_h).transpose(1, 0)
        atom_indices = torch.stack(atom_indices).transpose(1, 0)
        
        # compute cut sizes between all pairs of subgraphs for compression calculation
        cut_matrices = []
        subgraph_nodes_all = list(map(list, zip(*subgraph_nodes_all)))
        for i in range(len(init_graph.graph_size)):
            cut_matrices.append(compute_cut_size_pairs(init_graph,
                                                       subgraph_nodes_all[i],
                                                       directed=self.env.directed))
        cut_matrices = torch.stack(cut_matrices)
        
        # extract upper triangular part for undirected graphs, all entries for directed
        c_ij = cut_matrices[torch.triu(torch.ones_like(cut_matrices), diagonal=1) == 1].view(cut_matrices.shape[0], -1) \
            if not self.env.directed else cut_matrices.view(cut_matrices.shape[0], -1)
            
        # compute pairwise combinations of subgraph sizes for compression calculation
        n_h_ij = compute_combinations(n_h, directed=self.env.directed)
        e_h_ij = compute_combinations(e_h, directed=self.env.directed)
        
        # compile final subgraph information
        subgraphs = {'n_0': init_graph.graph_size,       # original graph sizes
                     'e_0': init_graph.edge_size,        # original edge counts
                     'b': b,                             # decomposition depths
                     'n_h': n_h,                         # subgraph node counts
                     'e_h': e_h,                         # subgraph edge counts
                     'atom_indices': atom_indices,       # dictionary mappings
                     'c_ij': c_ij,                       # pairwise cut sizes
                     'n_h_ij': n_h_ij,                   # pairwise node combinations
                     'e_h_ij': e_h_ij}                   # pairwise edge combinations
                     
        log_probs = torch.stack(log_probs).transpose(1, 0)
        return subgraphs, log_probs, visualise_data_dict



    def select_subgraph(self, graph, h_0_nodes, h_0_graph, global_context_updated, policy):
        """
        Select subgraphs from each graph in the batch using the neural policy.
        
        This method implements the subgraph selection strategy:
        1. For each graph in the batch, sample a subgraph using the policy
        2. Extract the induced subgraph and compute its properties
        3. Map the subgraph to the dictionary if it meets size constraints
        4. Update the dictionary if training and subgraph is novel
        
        Args:
            graph: Current graph data to select from
            h_0_nodes: Node representations from the policy network
            h_0_graph: Graph representations from the policy network
            global_context_updated: Updated global context vector
            policy: Neural policy network for making selections
            
        Returns:
            tuple: (selected_subgraphs_dict, log_probs)
                - selected_subgraphs_dict: Contains subgraph properties and mappings
                - log_probs: Log probabilities of the selection actions
        """
        graph_edge_features = graph.edge_features if hasattr(graph, 'edge_features') else None
        n_h = torch.zeros_like(graph.graph_size)          # number of nodes in each subgraph
        e_h = torch.zeros_like(graph.graph_size)          # number of edges in each subgraph
        atom_indices = -1 * torch.ones_like(graph.graph_size).long()  # dictionary mappings (-1 = not found)
        subgraph_nodes = []                               # selected nodes for each subgraph
        log_probs = torch.zeros_like(graph.graph_size)   # log probabilities of selections
        nodes_all = torch.arange(graph.x.shape[0], device=graph.x.device)
        
        # ------- sample one subgraph for each graph in the batch
        for i in range(len(graph.graph_size)):
            nodes_i = nodes_all[graph.batch == i]
            num_nodes_i = len(nodes_i)
            
            # skip empty graphs
            if num_nodes_i == 0:
                subgraph_nodes.append(nodes_i)
                continue
                
            # select subgraph nodes using the specified sampling mechanism
            if self.sampling_mechanism == 'sampling_without_replacement':
                subgraph_nodes_i, log_prob_nodes = compute_ordered_set_probs(h_0_nodes,
                                                                             h_0_graph[i:i+1],
                                                                             global_context_updated[i:i+1],
                                                                             policy,
                                                                             self.n_h_min,
                                                                             nodes_i,
                                                                             graph)
                log_probs[i] = log_prob_nodes
                subgraph_nodes_i = subgraph_nodes_i.view(-1)
            else:
                raise NotImplementedError(
                    "Sampling mechanism {} is not currently supported.".format(self.sampling_mechanism))
            
            # extract induced subgraph (edges within the selected nodes)
            subgraph_edges_i, subgraph_edge_features_i, relabelling_i = induced_subgraph(subgraph_nodes_i,
                                                                                         graph.edge_index,
                                                                                         edge_attr=graph_edge_features,
                                                                                         relabel_nodes=True,
                                                                                         num_nodes=int(graph.graph_size.sum()))
            
            subgraph_nodes.append(subgraph_nodes_i)
            n_h[i] = len(subgraph_nodes_i)
            # account for directed vs undirected edge counting
            e_h[i] = subgraph_edges_i.shape[1] / 2 if not self.env.directed else subgraph_edges_i.shape[1]
            
            # attempt to map subgraph to dictionary if it meets size constraints
            if n_h[i] <= self.n_h_max_dict and n_h[i] >= self.n_h_min_dict:
                atom_indices[i], G_i = self.env.map_to_dict(n_h[i], e_h[i], subgraph_edges_i,
                                                            self.env.dictionary_num_vertices,
                                                            self.env.dictionary_num_edges, self.env.dictionary,
                                                            self.query_inds, self.env.directed,
                                                            self.attr_mapping, graph.x[subgraph_nodes_i],
                                                            subgraph_edge_features_i)
                
                # if subgraph not found in dictionary, potentially add it during training
                if atom_indices[i] == -1:
                    update_condition = self.train and \
                                       self.env.universe_type == 'adaptive' \
                                       and len(self.env.dictionary) < self.env.max_dict_size
                    if update_condition:
                        atom_indices[i] = len(self.env.dictionary)
                        # update the current version of the dictionary
                        self.query_inds.append(len(self.env.dictionary))
                        self.env.update_dict_atoms([G_i],
                                                   [n_h[i]],
                                                   [e_h[i]])
                                                   
        selected_subgraphs = {'n_h': n_h,
                              'e_h': e_h,
                              'atom_indices': atom_indices,
                              'nodes': subgraph_nodes}

        return selected_subgraphs, log_probs

def compute_ordered_set_probs(h_0_nodes, h_0_graph,
                              global_context_updated,
                              policy,
                              k_min,
                              candidate_nodes, graph):
    """
    Compute probabilities for selecting an ordered set of nodes as a subgraph.
    
    This function implements a sequential node selection process:
    1. First decide on the subgraph size k (between k_min and num_available_nodes)
    2. Then sequentially select k nodes, where each selection influences the next
    3. After selecting a node, expand candidates to include its neighbors
    
    Args:
        h_0_nodes: Node representations from the GNN
        h_0_graph: Graph representation from the GNN
        global_context_updated: Updated global context vector
        policy: Neural policy network for making predictions
        k_min: Minimum number of nodes to select
        candidate_nodes: Initial set of candidate nodes to choose from
        graph: Graph data object containing edge information
        
    Returns:
        tuple: (selected_nodes, total_log_prob)
            - selected_nodes: Tensor of selected node indices
            - total_log_prob: Sum of log probabilities for all selection decisions
    """
    device = h_0_graph.device
    edge_index = graph.edge_index
    subgraph_nodes, log_prob = [], 0
    candidate_nodes, candidate_nodes_set = candidate_nodes.tolist(), set()
    num_nodes = len(candidate_nodes)
    
    # decide on subgraph size k
    if num_nodes <= k_min:
        # if not enough nodes, select all available
        selected_k = torch.tensor(num_nodes, device=device)
    else:
        # use policy to decide subgraph size (k_min to num_nodes)
        graph_logit = policy.predict_graph(h_0_graph, global_context_updated)
        k_distribution = Categorical(logits=graph_logit.view(-1)[0:num_nodes - k_min + 1])
        selected_k = k_distribution.sample() + k_min
        
    # prepare batch information for node-level predictions
    batch = torch.zeros((len(candidate_nodes),), dtype=torch.long, device=device)
    
    # compute initial node selection probabilities
    init_node_logits = torch.zeros((len(h_0_nodes), 1), device=device)
    init_node_logits[candidate_nodes] = policy.predict_node(h_0_nodes[candidate_nodes], global_context_updated, batch)
    
    # sequential node selection process
    while len(candidate_nodes) != 0:
        if len(subgraph_nodes) == selected_k:
            break
            
        # select next node from current candidates
        curr_logits = init_node_logits[candidate_nodes].view(-1)
        node_distribution = Categorical(logits=curr_logits)
        selected_ind = node_distribution.sample()
        log_prob += node_distribution.log_prob(selected_ind)
        
        # add selected node to subgraph
        new_node = candidate_nodes[selected_ind]
        subgraph_nodes.append(new_node)
        
        # expand candidate set to include neighbors of newly selected node
        new_node_nbs = edge_index[1, edge_index[0] == new_node]
        candidate_nodes_set.update(new_node_nbs.tolist())
        candidate_nodes_set.difference_update(subgraph_nodes)  # remove already selected nodes
        candidate_nodes = list(candidate_nodes_set)
        
    subgraph_nodes = torch.tensor(subgraph_nodes, device=device)
    
    # add log probability of size selection decision (if choice was available)
    if num_nodes > k_min:
        log_prob += k_distribution.log_prob(selected_k - k_min)
        
    return subgraph_nodes, log_prob

