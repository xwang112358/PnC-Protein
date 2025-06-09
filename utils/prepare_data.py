import os
import random
from copy import deepcopy
import torch
from torch_geometric.loader import DataLoader
from utils.attributes import AttrMapping
from utils.generate_dataset import generate_dataset
from utils.feature_encoding import encode_features
from utils.load_dataset import separate_data, separate_data_given_split
from utils.proteinshake_data import my_transform
from proteinshake.tasks import EnzymeClassTask

def prepare_dataset(
    path,
    dataset,
    name,
    directed=False,
    H_set=None,
    multiprocessing=False,
    num_processes=1,
    candidate_subgraphs=False,
):
    """
    Prepare and load dataset based on dataset type and configuration.

    Args:
        path (str): Path to the dataset directory
        dataset (str): Dataset family/type (e.g., 'bioinformatics', 'proteinshake')
        name (str): Specific dataset name (e.g., 'MUTAG', 'PROTEINSHAKE')
        directed (bool): Whether to use directed graphs. Defaults to False
        H_set (list, optional): Set of subgraph patterns for detection. Defaults to None
        multiprocessing (bool): Whether to use multiprocessing for subgraph detection. Defaults to False
        num_processes (int): Number of processes for multiprocessing. Defaults to 1
        candidate_subgraphs (bool): Whether to precompute candidate subgraphs. Defaults to False

    Returns:
        list: List of PyTorch Geometric graph objects

    Raises:
        NotImplementedError: If dataset family is not supported
    """

    # Handle traditional graph datasets (MUTAG, PTC_MR, etc.)
    if dataset in [
        "KarateClub",
        "PPI",
        "Planetoid",
        "Amazon",
        "TUDataset",
        "bioinformatics",
        "social",
        "chemical",
        "ogb",
        "SR_graphs",
        "all_graphs",
    ]:
        # Create processed data directory
        data_folder = os.path.join(path, "processed", "dictionaries")
        if not os.path.exists(data_folder):
            os.makedirs(data_folder)

        # Determine filename based on whether graphs are directed
        graphs_ptg_file = (
            os.path.join(data_folder, "directed_graphs_ptg.pt")
            if directed
            else os.path.join(data_folder, "graphs_ptg.pt")
        )

        # Load existing processed data or generate new dataset
        if not os.path.exists(graphs_ptg_file):
            # Generate dataset from raw data
            graphs_ptg, num_classes, num_node_type, num_edge_type = generate_dataset(
                path, name, directed
            )
            # Save processed dataset
            torch.save((graphs_ptg, num_classes), graphs_ptg_file)
            if num_node_type is not None:
                torch.save(
                    (num_node_type, num_edge_type),
                    os.path.join(data_folder, "num_feature_types.pt"),
                )
        else:
            # Load existing processed dataset
            graphs_ptg, num_classes = torch.load(graphs_ptg_file)

        # Precompute subgraphs if requested
        if candidate_subgraphs:
            from utils.subgraph_detection import detect_load_subgraphs

            graphs_ptg = detect_load_subgraphs(
                graphs_ptg, data_folder, H_set, directed, multiprocessing, num_processes
            )

    # Handle protein datasets from ProteinShake
    elif dataset in ["proteinshake"]:

        # Load enzyme classification task
        task = EnzymeClassTask()
        # Convert proteins to graphs with spatial proximity (eps=6Å, k=36 neighbors)
        dataset = task.dataset.to_graph(eps=6, k=36).pyg(transform=my_transform)

        # Use validation set for now (could be modified to use train/test)
        graphs_ptg = [dataset[i] for i in task.val_index]
        num_classes = task.num_classes
        num_node_type, num_edge_type = None, None

    else:
        raise NotImplementedError(
            "Dataset family {} is not currently supported.".format(dataset)
        )

    return graphs_ptg


def prepape_input_features(args, graphs_ptg, path):
    """
    Prepare and encode input features for graphs including node/edge attributes and degrees.

    Args:
        args (dict): Configuration arguments dictionary
        graphs_ptg (list): List of PyTorch Geometric graph objects
        path (str): Dataset path

    Returns:
        tuple: (processed_graphs, feature_dimensions_dict, attribute_mapping)
            - processed_graphs: List of graphs with encoded features
            - feature_dimensions_dict: Dictionary containing feature dimension information
            - attribute_mapping: AttrMapping object for feature conversion
    """

    # Optionally remove original features and use dummy features
    if "retain_features" in args and not args["retain_features"]:
        for graph in graphs_ptg:
            graph.x = torch.ones((graph.x.shape[0], 1))

    # Determine node feature dimensions
    d_in_node_features = 1 if graphs_ptg[0].x.dim() == 1 else graphs_ptg[0].x.shape[1]

    # Determine edge feature dimensions
    if hasattr(graphs_ptg[0], "edge_features"):
        d_in_edge_features = (
            1
            if graphs_ptg[0].edge_features.dim() == 1
            else graphs_ptg[0].edge_features.shape[1]
        )
    else:
        d_in_edge_features = None

    # Configure attribute mapping based on dataset type
    if args["dataset"] in ["PPI", "KarateClub", "Planetoid", "Amazon"]:
        # Continuous features - use integer encoding
        node_attr_unique_values, node_attr_dims = (
            [2 for _ in range(d_in_node_features)],
            d_in_node_features,
        )
        edge_attr_unique_values, edge_attr_dims = None, None
        attr_mapping = AttrMapping(
            args["dataset_name"], "integer", node_attr_dims, edge_attr_dims
        )

    elif args["dataset"] in ["TUDataset", "social", "bioinformatics"]:
        # Categorical features - use one-hot encoding
        node_attr_unique_values, node_attr_dims = [d_in_node_features], 1
        if d_in_edge_features is not None:
            edge_attr_unique_values, edge_attr_dims = [d_in_edge_features], 1
        else:
            edge_attr_unique_values, edge_attr_dims = None, None
        attr_mapping = AttrMapping(
            args["dataset_name"], "one_hot", node_attr_dims, edge_attr_dims
        )
        print("attribute mapping", node_attr_unique_values, node_attr_dims)

    elif args["dataset"] == "chemical":
        # Load precomputed feature type counts
        node_attr_unique_values, edge_attr_unique_values = torch.load(
            os.path.join(path, "processed/dictionaries", "num_feature_types.pt")
        )
        node_attr_unique_values, edge_attr_unique_values = (
            [node_attr_unique_values],
            [edge_attr_unique_values],
        )
        attr_mapping = AttrMapping(args["dataset_name"], "integer", 1, 1)

    elif args["dataset"] == "ogb":
        # Use OGB's built-in feature dimensions
        from ogb.utils.features import get_atom_feature_dims, get_bond_feature_dims

        node_attr_unique_values = get_atom_feature_dims()
        edge_attr_unique_values = get_bond_feature_dims()
        attr_mapping = AttrMapping(
            args["dataset_name"],
            "integer",
            len(node_attr_unique_values),
            len(edge_attr_unique_values),
        )

    elif args["dataset"] == "proteinshake":
        # Protein datasets - amino acids are already one-hot encoded
        node_attr_unique_values = [20]
        node_attr_dims = 1
        edge_attr_unique_values, edge_attr_dims = None, None
        attr_mapping = AttrMapping(
            args["dataset_name"], "one_hot", node_attr_dims, edge_attr_dims
        )

    else:
        raise NotImplementedError

    # Encode node degrees as additional features
    degree_encoding = (
        args["degree_encoding"]
        if ("degree_as_tag" in args and args["degree_as_tag"])
        else None
    )
    print("Encoding degree features... \n", end="")
    encoding_parameters = {}
    graphs_ptg, degree_unique_values, _ = encode_features(
        graphs_ptg, degree_encoding, None, **encoding_parameters
    )

    # Create feature dimensions dictionary
    in_features_dims_dict = {
        "d_in_node_features": d_in_node_features,
        "node_attr_unique_values": node_attr_unique_values,
        "d_in_edge_features": d_in_edge_features,
        "edge_attr_unique_values": edge_attr_unique_values,
        "degree_unique_values": degree_unique_values,
    }

    return graphs_ptg, in_features_dims_dict, attr_mapping


def prepare_dataloaders(
    args, graphs_ptg, path, fold_idx="", candidate_subgraphs=True, dictionary_size=0
):
    """
    Prepare PyTorch Geometric DataLoaders for training, testing, and validation.

    Args:
        args (dict): Configuration arguments containing batch_size, split type, etc.
        graphs_ptg (list): List of PyTorch Geometric graph objects
        path (str): Dataset path for loading/saving splits
        fold_idx (str or int): Fold index for cross-validation. Defaults to ''
        candidate_subgraphs (bool): Whether to include subgraph detection attributes. Defaults to True
        dictionary_size (int): Size of subgraph dictionary for batch following. Defaults to 0

    Returns:
        tuple: (train_loader, test_loader, val_loader)
            - train_loader: DataLoader for training data
            - test_loader: DataLoader for test data (None if not available)
            - val_loader: DataLoader for validation data (None if not available)
    """
    # Create deep copy to avoid modifying original graphs
    graphs_ptg_modified = deepcopy(graphs_ptg)
    follow_batch = []

    # Process subgraph detection attributes for batching
    if candidate_subgraphs:
        if hasattr(graphs_ptg[0], "subgraph_detections"):
            # Convert subgraph detections to individual attributes
            for graph in graphs_ptg_modified:
                for i in range(dictionary_size):
                    subgraph_index = (
                        graph.subgraph_detections[i].transpose(1, 0)
                        if graph.subgraph_detections[i].numel() != 0
                        else torch.tensor([])
                    )
                    setattr(graph, "subgraph_index_" + str(i), subgraph_index.long())
                del graph.subgraph_detections

        # Add subgraph indices to follow_batch for proper batching
        for i in range(dictionary_size):
            follow_batch.append("subgraph_index_" + str(i))

    # Split data based on configuration
    if args["split"] == "random":
        # Use random split with specified seed
        split_folder = "split_idx_random_" + str(args["split_seed"])
        if os.path.exists(os.path.join(path, split_folder)):
            # Load existing random split
            dataset_train, dataset_test, dataset_val = separate_data_given_split(
                graphs_ptg_modified, path, fold_idx, split_folder
            )
        else:
            # Create new random split
            os.makedirs(os.path.join(path, split_folder))
            dataset_train, dataset_test = separate_data(
                graphs_ptg_modified, args["split_seed"], fold_idx, path, split_folder
            )
        dataset_val = None

    elif args["split"] == "given":
        # Use precomputed/standard split
        dataset_train, dataset_test, dataset_val = separate_data_given_split(
            graphs_ptg_modified, path, fold_idx
        )
    elif args["split"] == "None":
        # Use all data as training (no test/val split)
        dataset_train, dataset_test, dataset_val = graphs_ptg_modified, None, None
    else:
        raise NotImplementedError(
            "data split {} method not implemented".format(args["split"])
        )

    # Create DataLoader for training data
    loader_train = DataLoader(
        dataset_train,
        batch_size=args["batch_size"],
        shuffle=args["shuffle"],
        worker_init_fn=random.seed(args["seed"]),
        num_workers=args["num_workers"],
        follow_batch=follow_batch,
    )

    # Create DataLoader for test data (if available)
    if dataset_test is not None:
        loader_test = DataLoader(
            dataset_test,
            batch_size=args["batch_size"],
            shuffle=False,  # Don't shuffle test data
            worker_init_fn=random.seed(args["seed"]),
            num_workers=args["num_workers"],
            follow_batch=follow_batch,
        )
    else:
        loader_test = None

    # Create DataLoader for validation data (if available)
    if dataset_val is not None:
        loader_val = DataLoader(
            dataset_val,
            batch_size=args["batch_size"],
            shuffle=False,  # Don't shuffle validation data
            worker_init_fn=random.seed(args["seed"]),
            num_workers=args["num_workers"],
            follow_batch=follow_batch,
        )
    else:
        loader_val = None

    return loader_train, loader_test, loader_val
