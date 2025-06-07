import torch

class ProbabilisticModel(torch.nn.Module):
    """
    Probabilistic model for graph compression that learns distributions over various compression parameters.
    
    This model parameterizes the probability distributions used in the minimum description length (MDL)
    framework for graph compression. It learns:
    - Dictionary membership probabilities (which atoms to include)
    - Block number distributions (how many subgraphs per graph)
    - Atom emission probabilities (which dictionary atoms to use)
    - Delta probabilities (dictionary vs non-dictionary subgraphs)
    - Cut size distributions (edge counts between subgraphs)
    """

    def __init__(self,
                 max_dict_size,
                 b_max,
                 b_distribution='uniform',
                 delta_distribution='uniform',
                 atom_distribution='uniform',
                 cut_size_distribution='uniform',
                 b_min=1,
                 c_max=None):
        """
        Initialize the probabilistic model with learnable parameters.
        
        Args:
            max_dict_size (int): Maximum size of the dictionary
            b_max (int): Maximum number of blocks (subgraphs) per graph
            b_distribution (str): Distribution type for block numbers ('uniform' or 'learnable')
            delta_distribution (str): Distribution type for dictionary vs non-dictionary ('uniform' or 'learnable')
            atom_distribution (str): Distribution type for atom selection ('uniform' or 'learnable')
            cut_size_distribution (str): Distribution type for cut sizes ('uniform' or 'learnable')
            b_min (int): Minimum number of blocks per graph
            c_max (int): Maximum cut size (for learnable cut size distribution)
        """
        super(ProbabilisticModel, self).__init__()
        self.max_dict_size = max_dict_size
        
        # Dictionary membership variables: binary indicators for each potential dictionary atom
        # x_a = binary variable indicating the existence of an atom in the dictionary or not
        self.x_a_logits = torch.nn.Parameter(torch.zeros((max_dict_size,)))
        # Regarding the initialisation: we initialise the membership variables close to 1
        # since we observed it facilitates optimisation
        # (e.g., it allows the NeuralPart to initially focus on good partitions rather than frequent subgraphs)
        torch.nn.init.constant_(self.x_a_logits, 4)

        # Block number distribution parameters
        self.b_max = b_max
        self.b_min = b_min
        self.b_distribution = b_distribution
        if b_distribution=='learnable':
            # p(b) = probability of number of blocks in the partition
            self.b_logits = torch.nn.Parameter(torch.zeros((b_max - b_min + 1,)))
            torch.nn.init.zeros_(self.b_logits)
            
        # Delta distribution: probability of using dictionary vs non-dictionary subgraphs
        self.delta_distribution = delta_distribution
        if delta_distribution=='learnable':
            # delta = probability of emission of a non-dictionary subgraph
            self.subgraph_in_dict_logit = torch.nn.Parameter(torch.zeros((1,)))
            torch.nn.init.zeros_(self.subgraph_in_dict_logit)
            
        # Atom distribution: probabilities for selecting specific dictionary atoms
        self.atom_distribution = atom_distribution
        if atom_distribution == 'learnable':
            # p(a) = probability of emission of atom a
            self.atom_logits = torch.nn.Parameter(torch.zeros((max_dict_size,)))
            torch.nn.init.zeros_(self.atom_logits)
            
        # Cut size distribution: probabilities for different numbers of edges between subgraphs
        self.cut_size_distribution = cut_size_distribution
        if cut_size_distribution=='learnable':
            self.cut_size_logits = torch.nn.Parameter(torch.zeros((c_max,)))
            torch.nn.init.zeros_(self.cut_size_logits)
            
        # Activation functions for converting logits to probabilities
        self.softmax = torch.nn.Softmax(dim=0)
        self.sigmoid = torch.nn.Sigmoid()

    def membership_vars(self, current_universe_size=None):
        """
        Compute dictionary membership probabilities using sigmoid activation.
        
        Args:
            current_universe_size (int, optional): Size of current dictionary. 
                                                  If None, uses full max_dict_size.
        
        Returns:
            torch.Tensor: Sigmoid probabilities indicating dictionary membership for each atom
        """
        if current_universe_size is None:
            current_universe_size = len(self.x_a_logits)
        x_a = torch.zeros_like(self.x_a_logits)
        x_a[0:current_universe_size] = self.sigmoid(self.x_a_logits[0:current_universe_size])
        return x_a

    def b_probs(self):
        """
        Compute probabilities for different block numbers using softmax.
        
        Returns:
            torch.Tensor: Probability distribution over possible block numbers
        """
        return self.softmax(self.b_logits)

    def delta_prob(self):
        """
        Compute probability of emitting a non-dictionary subgraph.
        
        Returns:
            torch.Tensor: Probability that a subgraph is NOT in the dictionary
        """
        return 1 - self.sigmoid(self.subgraph_in_dict_logit)

    def atom_probs(self, x_a):
        """
        Compute probabilities for selecting specific dictionary atoms.
        
        Args:
            x_a (torch.Tensor): Dictionary membership variables
            
        Returns:
            torch.Tensor: Probability distribution over dictionary atoms,
                         normalized only over atoms that are in the dictionary
        """
        p_a = torch.zeros_like(self.atom_logits)
        mask_probs = x_a != 0  # only consider atoms that are in the dictionary
        p_a[mask_probs] = self.softmax(x_a[mask_probs].log() + self.atom_logits[mask_probs])
        return p_a

    def cut_size_probs(self):
        """
        Compute probabilities for different cut sizes.
        
        Returns:
            torch.Tensor: Probability distribution over possible cut sizes
        """
        return self.softmax(self.cut_size_logits)

    def prune_universe(self, train):
        """
        Generate probability distributions for all compression parameters.
        
        This method computes the current probability distributions based on the learned parameters.
        During training, it uses the straight-through estimator for dictionary membership.
        During inference, it uses hard thresholding for dictionary selection.
        
        Args:
            train (bool): Whether the model is in training mode
            
        Returns:
            tuple: (x_a, b_probs, delta_prob, atom_probs, cut_size_probs)
                - x_a: Dictionary membership variables (binary in inference, soft in training)
                - b_probs: Probability distribution over block numbers
                - delta_prob: Probability of non-dictionary subgraphs
                - atom_probs: Probability distribution over dictionary atoms
                - cut_size_probs: Probability distribution over cut sizes (or None)
        """
        # Dictionary membership: atoms in dict
        x_a = self.membership_vars()
        if train:
            # Straight-through estimator: forward pass uses soft probabilities,
            # backward pass uses gradients as if we used hard decisions
            x_a_hard = (x_a > 0.5).float()
            x_a = (x_a_hard - x_a).detach() + x_a
        else:
            # Hard thresholding for inference
            x_a = (x_a > 0.5).float()

        # Block number distribution
        if self.b_distribution == 'uniform':
            # Uniform distribution over possible block numbers
            b_probs = 1 / (self.b_max - self.b_min + 1) *\
                      torch.ones((self.b_max - self.b_min + 1,), device=x_a.device)
        elif self.b_distribution == 'learnable':
            # Learned distribution over block numbers
            b_probs = self.b_probs()
        else:
            # Default: all probability mass on maximum blocks
            b_probs = torch.ones((self.b_max - self.b_min + 1,), device=x_a.device)

        # Atom emission probabilities
        if self.atom_distribution == 'uniform':
            # Uniform distribution over dictionary atoms (normalized by dictionary size)
            atom_probs = x_a / x_a.sum() if x_a.sum() != 0 else torch.zeros_like(x_a)
        else:
            # Learned distribution over atoms
            atom_probs = self.atom_probs(x_a)

        # Delta probability: dictionary vs non-dictionary subgraphs
        if self.delta_distribution == 'uniform':
            # Equal probability for dictionary and non-dictionary subgraphs
            delta_prob = 1 / 2 * torch.ones((1,), device=x_a.device)
        elif self.delta_distribution == 'learnable':
            # Learned probability for non-dictionary subgraphs
            delta_prob = self.delta_prob()
        else:
            # Default: always use dictionary
            delta_prob = torch.ones((1,), device=x_a.device)

        # Cut size distribution
        if self.cut_size_distribution == 'uniform':
            # No specific distribution (handled elsewhere)
            cut_size_probs = None
        else:
            # Learned distribution over cut sizes
            cut_size_probs = self.cut_size_probs()
            
        return x_a, b_probs, delta_prob, atom_probs, cut_size_probs

    def count_params(self, x_a):
        """
        Count the effective number of parameters in the model.
        
        This is used for computing the model description length in the MDL framework.
        Only parameters corresponding to active components are counted.
        
        Args:
            x_a (torch.Tensor): Dictionary membership variables
            
        Returns:
            int: Total number of effective parameters
        """
        num_params = 0
        
        # Count block distribution parameters
        if self.b_distribution == 'learnable':
            num_params += self.b_logits.numel()
            
        # Count delta distribution parameters
        if self.delta_distribution == 'learnable':
            num_params += self.subgraph_in_dict_logit.numel()
            
        # Count atom distribution parameters (only for atoms in dictionary)
        if self.atom_distribution == 'learnable':
            num_params += x_a.sum().item()
            
        # Count cut size distribution parameters
        if self.cut_size_distribution == 'learnable':
            num_params += self.cut_size_logits.numel()
            
        return num_params


