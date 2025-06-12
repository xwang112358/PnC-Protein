import sys

sys.path.append("../")
import argparse
import utils.parsing as parse
import os
import torch
import numpy as np
import random
from utils.prepare_data import (
    prepare_dataset,
    prepape_input_features,
    prepare_dataloaders,
)
from utils.prepare_dictionary import prepare_dictionary
from utils.isomorphism_modules import prepare_isomorphism_module
from utils.prepare_arguments import prepare_environment_args, prepare_model_args
from encoding_decoding.environment import CompressionEnvironment
from models.probabilistic_model import ProbabilisticModel
from agent_neural_part import CompressionAgentNeuralPart
from models.GNN_neural_part_subgraph_selection import NeuralPartGNN
from utils.optim import setup_optimization, resume_training_phi_theta
import utils.loss_evaluation_fns as loss_evaluation_fns
from utils.test_and_log import prepare_logs, logger
from train_neural_part import train


def main(args):
    ## ----------------------------------- infrastructure

    torch.manual_seed(args["seed"])
    torch.cuda.manual_seed(args["seed"])
    torch.cuda.manual_seed_all(args["seed"])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args["np_seed"])
    os.environ["PYTHONHASHSEED"] = str(args["seed"])
    random.seed(args["seed"])
    print("[info] Setting all random seeds {}".format(args["seed"]))
    torch.set_num_threads(args["num_threads"])
    if args["GPU"]:
        device = torch.device(
            "cuda:" + str(args["device_idx"]) if torch.cuda.is_available() else "cpu"
        )
        print(
            "[info] Training will be performed on {}".format(
                torch.cuda.get_device_name(args["device_idx"])
            )
        )
    else:
        device = torch.device("cpu")
        print("[info] Training will be performed on cpu")
    if args["wandb"]:
        print("Wandb logging activated... ")
        import wandb

        wandb.init(
            sync_tensorboard=False,
            project=args["wandb_project"],
            name=f'{args["dataset"]}_{args["dataset_name"]}_{args["wandb_name"]}',
            reinit=False,
            config=args,
            entity=args["wandb_entity"],
            mode=args["wandb_mode"],
        )
        print("[info] Monitoring with wandb")
    path = os.path.join(args["root_folder"], args["dataset"], args["dataset_name"])
    perf_opt = np.argmin
    train_compression_folds = []
    test_compression_folds = []
    val_compression_folds = []
    total_compression_folds = []
    total_compression_w_params_folds = []
    num_params_folds = []
    fold_idxs = args["fold_idx"]
    assert args["mode"] in ["train", "test"], (
        "Unknown mode. Supported options are  'train','test'"
    )
    for fold_idx in fold_idxs:
        print("############# FOLD NUMBER {:01d} #############".format(fold_idx))
        ## ----------------------------------- preparation of model components
        # prepare dictionary with initial dictionary atoms (removed)

        H_set_gt = prepare_dictionary(args)
        # generate/load dataset + detect/load subgraphs
        graphs_ptg = prepare_dataset(
            path,
            args["dataset"],
            args["dataset_name"],
            directed=args["directed"],
            H_set=H_set_gt,
            multiprocessing=args["multiprocessing"],
            num_processes=args["num_processes"],
            candidate_subgraphs=args["candidate_subgraphs"],
        )

        print(20 * "#", " Dataset loaded ", 20 * "#")
        print(graphs_ptg[0])
        print("d_in_node_features", graphs_ptg[0].x.dim())

        # one-hot encoding etc of input features
        graphs_ptg, in_features_dims_dict, attr_mapping = prepape_input_features(
            args, graphs_ptg, path
        )

        print(20 * "#", " Input features prepared ", 20 * "#")
        print("Input features dimensions: {}".format(in_features_dims_dict))
        print(f"attr_mapping: {attr_mapping.node_attr_values}, {attr_mapping.node_attr_dims}")
        

        # prepare and instantiate isomoprhism module
        isomorphism_module = prepare_isomorphism_module(
            args["isomorphism_type"],
            node_attr_dims=None
            if args["node_attr_encoding"] is None
            else attr_mapping.node_attr_dims,
            edge_attr_dims=None
            if args["edge_attr_encoding"] is None
            else attr_mapping.edge_attr_dims,
        )
        print(20 * "#", " Isomorphism module prepared ", 20 * "#")
        print(args["isomorphism_type"], 'Isomorphism')
        print('node/edge encoding type', args["node_attr_encoding"], args["edge_attr_encoding"])
        
        # prepare compression environment
        environment_args = prepare_environment_args(
            args,
            graphs_ptg,
            H_set_gt,
            device,
            isomorphism_module,
            in_features_dims_dict["node_attr_unique_values"],
            in_features_dims_dict["edge_attr_unique_values"],
        )
        print(20 * "#", " Environment prepared ", 20 * "#")
        print(environment_args)
        # assert False

        environment = CompressionEnvironment(**environment_args)
        print("Num graphs: {}".format(len(graphs_ptg)))
        print(
            "Max degree: {}".format(
                in_features_dims_dict["degree_unique_values"][0] - 1
            )
        )

        
        n_mean = np.mean([graph.x.shape[0] for graph in graphs_ptg])
        print("Avg/Max num nodes: {:.2f}, {}".format(n_mean, environment_args["n_max"]))
        ## ----------------------------------- prepare model (neural partitioning)
        model_args = prepare_model_args(args, len(H_set_gt))
        ## ----------------------------------- prepare evaluators and loggers
        evaluation_fn = getattr(loss_evaluation_fns, args["evaluation_fn"])
        checkpoint_folder = prepare_logs(args, path, fold_idx)
        # prepare dataloaders
        loader_train, loader_test, loader_val = prepare_dataloaders(
            args, graphs_ptg, path, fold_idx, args["candidate_subgraphs"], len(H_set_gt)
        )
        # intialise model
        policy_model = NeuralPartGNN
        policy_network = policy_model(
            d_in_node_features=in_features_dims_dict["d_in_node_features"],
            d_in_node_embedding=in_features_dims_dict["node_attr_unique_values"],
            d_in_edge_features=in_features_dims_dict["d_in_edge_features"],
            d_in_edge_embedding=in_features_dims_dict["edge_attr_unique_values"],
            d_in_degree_embedding=in_features_dims_dict["degree_unique_values"],
            **model_args,
        )
        policy_network = policy_network.to(device)
        # print("Instantiated model:\n{}".format(policy_network))
        # count model params
        params = sum(p.numel() for p in policy_network.parameters() if p.requires_grad)
        print("[info] Total number of parameters is: {}".format(params))
        # initialise dictionary learnable parameters
        dictionary_probs_model = ProbabilisticModel(
            args["max_dict_size"],
            args["b_max"],
            b_distribution=args["b_distribution"],
            delta_distribution=args["delta_distribution"],
            atom_distribution=args["atom_distribution"],
            cut_size_distribution=args["cut_size_distribution"],
            b_min=args["b_min"],
            c_max=None,
        ).to(device)

        # print("Instantiated model:\n{}".format(dictionary_probs_model))
        # count model params
        params = sum(
            p.numel() for p in dictionary_probs_model.parameters() if p.requires_grad
        )
        print("[info] Total number of parameters is: {}".format(params))
        # intialise agent
        kwargs_agent = {
            "attr_mapping": attr_mapping,
            "n_h_max": args["n_h_max"],
            "n_h_min": args["n_h_min"],
            "n_h_max_dict": args["n_h_max_dict"],
            "n_h_min_dict": args["n_h_min_dict"],
            "sampling_mechanism": args["sampling_mechanism"],
        }
        compression_agent = CompressionAgentNeuralPart
        agent = compression_agent(policy_network, environment, **kwargs_agent)

        # assert False

        if args["mode"] == "train":
            print("Training starting now...")
            # optimizer and lr scheduler
            trainable_parameters_phi = [
                p for p in dictionary_probs_model.parameters() if p.requires_grad
            ]
            kargs_optim_phi = {
                "lr": args["lr_dict"],
                "regularization": args["regularization"],
                "scheduler": args["scheduler"],
                "scheduler_mode": args["scheduler_mode"],
                "decay_rate": args["decay_rate"],
                "decay_steps": args["decay_steps"],
                "patience": args["patience"],
            }
            trainable_parameters_theta = [
                p for p in policy_network.parameters() if p.requires_grad
            ]
            kargs_optim_theta = {
                "lr": args["lr_policy"],
                "regularization": args["regularization"],
                "scheduler": args["scheduler"],
                "scheduler_mode": args["scheduler_mode"],
                "decay_rate": args["decay_rate"],
                "decay_steps": args["decay_steps"],
                "patience": args["patience"],
            }
            optim_phi, scheduler_phi = setup_optimization(
                trainable_parameters_phi, optimiser_name="Adam", **kargs_optim_phi
            )
            optim_theta, scheduler_theta = setup_optimization(
                trainable_parameters_theta, optimiser_name="Adam", **kargs_optim_theta
            )
        else:
            optim_phi, scheduler_phi, optim_theta, scheduler_theta = (
                None,
                None,
                None,
                None,
            )
        if args["resume"]:
            resume_folder = (
                args["resume_folder"]
                if args["resume_folder"] is not None
                else checkpoint_folder
            )
            resume_filename = os.path.join(
                resume_folder, args["checkpoint_file"] + ".pth.tar"
            )
            start_epoch = resume_training_phi_theta(
                resume_filename,
                policy_network,
                dictionary_probs_model,
                optim_phi,
                optim_theta,
                scheduler_phi,
                scheduler_theta,
                device,
                env=agent.env,
            )
        else:
            start_epoch = 0
        # logging
        if args["wandb"]:
            wandb.watch(policy_network)
            wandb.watch(dictionary_probs_model)
        checkpoint_filename = os.path.join(
            checkpoint_folder, args["checkpoint_file"] + ".pth.tar"
        )
        kwargs_train_test = {
            "visualise": args["visualise"] and args["wandb"],
            "inds_to_visualise": []
            if args["inds_to_visualise"] is None
            else args["inds_to_visualise"],
            "bits_per_parameter": args["bits_per_parameter"],
            "amortisation_param": args["amortisation_param"],
        }
        ## ----------------------------------- training
        metrics = train(
            agent,
            dictionary_probs_model,
            loader_train,
            loader_test,
            optim_phi,
            optim_theta,
            start_epoch=start_epoch,
            n_epochs=args["num_epochs"],
            n_iters=args["n_iters_train"],
            n_iters_test=args["n_iters_test"],
            eval_freq=args["eval_frequency"],
            loader_val=loader_val,
            evaluation_fn=evaluation_fn,
            scheduler_phi=scheduler_phi,
            scheduler_theta=scheduler_theta,
            min_lr=args["min_lr"],
            checkpoint_file=checkpoint_filename,
            wandb_realtime=args["wandb_realtime"] and args["wandb"],
            fold_idx=fold_idx,
            mode=args["mode"],
            **kwargs_train_test,
        )
        print("Training/Testing complete!")
        (
            train_compression_p_epoch,
            test_compression_p_epoch,
            val_compression_p_epoch,
            total_compression_p_epoch,
            total_compression_w_params_p_epoch,
            num_params_p_epoch,
        ) = metrics
        # log results of training
        train_compression_folds.append(train_compression_p_epoch)
        test_compression_folds.append(test_compression_p_epoch)
        val_compression_folds.append(val_compression_p_epoch)
        total_compression_folds.append(total_compression_p_epoch)
        total_compression_w_params_folds.append(total_compression_w_params_p_epoch)
        num_params_folds.append(num_params_p_epoch)
        best_idx = perf_opt(train_compression_p_epoch)
        print(
            "\tbest epoch {}\n\tbest train accuracy {:.4f}, "
            "\n\tbest test accuracy {:.4f},"
            "\n\tbest total accuracy {:.4f},"
            " \n\tbest total w params accuracy {:.4f}, "
            " \n\tnum params {},".format(
                best_idx,
                train_compression_p_epoch[best_idx],
                test_compression_p_epoch[best_idx],
                total_compression_p_epoch[best_idx],
                total_compression_w_params_p_epoch[best_idx],
                num_params_p_epoch[best_idx],
            )
        )

    if args["mode"] == "train":
        if loader_val is not None:
            compression_folds = [
                train_compression_folds,
                test_compression_folds,
                val_compression_folds,
                total_compression_folds,
                total_compression_w_params_folds,
                num_params_folds,
            ]
            names = ["train", "test", "val", "total", "total_w_params", "num_params"]
        elif loader_test is not None:
            compression_folds = [
                train_compression_folds,
                test_compression_folds,
                total_compression_folds,
                total_compression_w_params_folds,
                num_params_folds,
            ]
            names = ["train", "test", "total", "total_w_params", "num_params"]
        else:
            compression_folds = [
                train_compression_folds,
                total_compression_folds,
                total_compression_w_params_folds,
                num_params_folds,
            ]
            names = ["train", "total", "total_w_params", "num_params"]
        logger(
            compression_folds, names, perf_opt, args["wandb"], args["wandb_realtime"]
        )
    if args["mode"] == "test":
        print(
            "Train accuracy: {:.4f} +/- {:.4f}".format(
                np.mean(train_compression_folds), np.std(train_compression_folds)
            )
        )
        print(
            "Test accuracy: {:.4f} +/- {:.4f}".format(
                np.mean(test_compression_folds), np.std(test_compression_folds)
            )
        )
        if loader_val is not None:
            print(
                "Validation accuracy: {:.4f} +/- {:.4f}".format(
                    np.mean(val_compression_folds), np.std(val_compression_folds)
                )
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # ----------------- seeds
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--np_seed", type=int, default=0)
    parser.add_argument("--num_threads", type=int, default=1)
    # ----------------- infrastructure + dataloader + logging + visualisation
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--wandb", type=parse.str2bool, default=True)
    parser.add_argument("--wandb_realtime", type=parse.str2bool, default=False)
    parser.add_argument("--wandb_project", type=str, default="graph_compression")
    parser.add_argument("--wandb_entity", type=str, default="xwang38438")
    parser.add_argument("--wandb_name", type=str, default=None)
    parser.add_argument(
        "--wandb_mode", type=str, default="online"
    )  # "offline" for local runs
    parser.add_argument("--visualise", type=parse.str2bool, default=True)
    parser.add_argument("--inds_to_visualise", type=parse.str2list2int, default=None)
    parser.add_argument("--GPU", type=parse.str2bool, default=True)
    parser.add_argument("--device_idx", type=int, default=0)
    parser.add_argument(
        "--results_folder", type=str, default="PnC_neural_part_experiments"
    )
    parser.add_argument("--checkpoint_file", type=str, default="checkpoint")
    parser.add_argument("--resume", type=parse.str2bool, default=False)
    parser.add_argument(
        "--resume_folder", type=str, default=None
    )  # useful to load models trained on other distributions
    # ----------------- dataset and split
    parser.add_argument("--root_folder", type=str, default="../datasets/")
    parser.add_argument("--dataset", type=str, default="bioinformatics")
    parser.add_argument("--dataset_name", type=str, default="MUTAG")
    parser.add_argument("--directed", type=parse.str2bool, default=False)
    parser.add_argument("--fold_idx", type=parse.str2list2int, default=[0])
    parser.add_argument("--split", type=str, default="random")
    parser.add_argument(
        "--split_seed", type=int, default=0
    )  # only for random splitting
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    # ----------------- compression environment
    parser.add_argument("--universe_type", type=str, default="adaptive")
    parser.add_argument("--max_dict_size", type=int, default=10000)
    # constants
    parser.add_argument("--precision", type=int, default=None)  # Numerical precision bits
    parser.add_argument("--n_max", type=int, default=None)  # Maximum number of nodes
    parser.add_argument("--e_max", type=int, default=None)  # Maximum number of edges
    parser.add_argument("--d_max", type=int, default=None)  # Maximum node degree
    parser.add_argument("--c_max", type=int, default=None)  # Maximum cut size
    parser.add_argument("--b_max", type=int, default=None)  # Maximum number of blocks
    parser.add_argument("--b_min", type=int, default=None)  # Minimum number of blocks
    parser.add_argument(
        "--n_h_max_dict", type=int, default=-1
    )  # Max nodes in dict subgraphs (-1 = n_max)
    parser.add_argument("--n_h_min_dict", type=int, default=1)  # Min nodes in dictionary subgraphs
    parser.add_argument("--n_h_max", type=int, default=-1)  # Max nodes in subgraphs
    parser.add_argument("--n_h_min", type=int, default=1)  # Min nodes in subgraphs
    # ----------------- encoding schemes
    # dictionary encoding
    parser.add_argument("--dictionary_encoding", type=str, default="graphs")  # Dictionary encoding method
    parser.add_argument("--num_nodes_atom_encoding", type=str, default="uniform")  # Node count encoding for atoms
    parser.add_argument("--num_edges_atom_encoding", type=str, default="uniform")  # Edge count encoding for atoms
    parser.add_argument("--adj_matrix_atom_encoding", type=str, default="erdos_renyi")  # Adjacency matrix encoding for atoms
    # subgraph encoding (dictionary and non-dictionary)
    parser.add_argument("--dict_subgraphs_encoding", type=str, default="multinomial")  # Dictionary subgraph selection encoding
    parser.add_argument("--num_nodes_encoding", type=str, default="uniform")  # Node count encoding method
    parser.add_argument("--num_edges_encoding", type=str, default="uniform")  # Edge count encoding method
    parser.add_argument("--adj_matrix_encoding", type=str, default="erdos_renyi")  # Adjacency matrix encoding method
    # cut encoding
    parser.add_argument("--cut_encoding", type=str, default="joint")  # Cut encoding strategy
    parser.add_argument("--cut_size_encoding", type=str, default="uniform")  # Cut size encoding method
    parser.add_argument("--cut_edges_encoding", type=str, default="erdos_renyi")  # Cut edges encoding method
    # attribute encoding
    parser.add_argument("--node_attr_encoding", type=str, default=None)  # Node attribute encoding method
    parser.add_argument("--edge_attr_encoding", type=str, default=None)  # Edge attribute encoding method
    # baseline encoding
    parser.add_argument("--num_nodes_baseline_encoding", type=str, default="uniform")  # Baseline node count encoding
    parser.add_argument("--num_edges_baseline_encoding", type=str, default="uniform")  # Baseline edge count encoding
    parser.add_argument(
        "--adj_matrix_baseline_encoding", type=str, default="erdos_renyi"
    )  # Baseline adjacency encoding
    # ----------------- isomorphism
    parser.add_argument("--isomorphism_type", type=str, default="exact")  # Graph isomorphism matching type
    # exponential moving average coefficient: used to compute the empirical frequencies of the atoms in order
    # to speed-up the matching between subgraphs and dictionary atoms
    parser.add_argument("--ema_coeff", type=float, default=0.5)  # EMA coefficient for atom frequencies
    # ----------------- Neural Network parameters (neural partitioning)
    parser.add_argument("--input_node_embedding", type=str, default="None")  # Input node embedding type
    parser.add_argument("--d_out_node_embedding", type=int, default=None)  # Node embedding output dimension
    parser.add_argument("--edge_embedding", type=str, default="None")  # Edge embedding type
    parser.add_argument("--d_out_edge_embedding", type=int, default=None)  # Edge embedding output dimension
    parser.add_argument("--inject_edge_features", type=parse.str2bool, default=False)  # Include edge features in messages
    parser.add_argument("--multi_embedding_aggr", type=str, default="sum")  # Multiple embedding aggregation method

    parser.add_argument("--degree_encoding", type=str, default="one_hot_max")  # Node degree encoding method
    parser.add_argument("--degree_embedding", type=str, default="one_hot_encoder")  # Degree embedding type
    parser.add_argument("--d_out_degree_embedding", type=int, default=None)  # Degree embedding output dimension
    parser.add_argument("--degree_as_tag", type=parse.str2bool, default=True)  # Use degree as node tag
    parser.add_argument("--retain_features", type=parse.str2bool, default=False)  # Keep original node features
    parser.add_argument("--inject_degrees", type=parse.str2bool, default=False)  # Add degree info to features

    parser.add_argument("--train_eps", type=parse.str2bool, default=False)  # Train epsilon parameter
    parser.add_argument("--aggr", type=str, default="add")  # Message aggregation function
    parser.add_argument("--flow", type=str, default="source_to_target")  # Message passing direction
    parser.add_argument("--extend_dims", type=parse.str2bool, default=True)  # Extend feature dimensions
    parser.add_argument("--d_msg", type=int, default=None)  # Message dimension
    parser.add_argument("--d_h", type=int, default=None)  # Hidden layer dimension
    parser.add_argument("--num_mlp_layers", type=int, default=2)  # Number of MLP layers
    parser.add_argument("--activation_mlp", type=str, default="relu")  # MLP activation function
    parser.add_argument("--bn_mlp", type=parse.str2bool, default=False)  # Batch normalization in MLP
    parser.add_argument("--bn", type=parse.str2bool, default=False)  # Batch normalization
    parser.add_argument("--activation", type=str, default="relu")  # Activation function
    parser.add_argument("--dropout", type=float, default=0)  # Dropout rate

    parser.add_argument("--num_layers", type=int, default=2)  # Number of neural network layers
    parser.add_argument("--model_name", type=str, default="MPNN_sparse")  # Neural network model name
    parser.add_argument("--aggr_fn", type=str, default="general")  # Aggregation function type
    parser.add_argument("--d_out", type=int, default=16)  # Output feature dimension
    parser.add_argument("--final_projection", type=parse.str2list2bool, default=[False])  # Apply final projection layer
    parser.add_argument("--final_projection_layer", type=str, default="mlp")  # Final projection layer type
    parser.add_argument("--readout", type=str, default="sum")  # Graph readout function
    # ----------------- Neural Partitioning specifications
    parser.add_argument(
        "--partitioning_algorithm", type=str, default="subgraph_selection"
    )  # Graph partitioning algorithm
    parser.add_argument(
        "--sampling_mechanism", type=str, default="sampling_without_replacement"
    )  # Subgraph sampling strategy
    # ----------------- probabilistic model learnable parameters
    parser.add_argument("--b_distribution", type=str, default="learnable")  # Block size distribution type
    parser.add_argument("--delta_distribution", type=str, default="learnable")  # Delta parameter distribution type
    parser.add_argument("--atom_distribution", type=str, default="learnable")  # Atom selection distribution type
    parser.add_argument("--cut_size_distribution", type=str, default="uniform")  # Cut size distribution type
    # ----------------- optimisation and learning parameters
    parser.add_argument("--shuffle", type=parse.str2bool, default=True)  # Shuffle training data
    parser.add_argument("--n_iters_train", type=int, default=None)  # Training iterations per epoch
    parser.add_argument("--n_iters_test", type=int, default=None)  # Testing iterations per epoch
    parser.add_argument("--eval_frequency", type=int, default=1)  # Evaluation frequency in epochs
    parser.add_argument("--regularization", type=float, default=0)  # L2 regularization weight
    parser.add_argument("--scheduler", type=str, default="None")  # Learning rate scheduler type
    parser.add_argument("--scheduler_mode", type=str, default="min")  # Scheduler mode (min/max)
    parser.add_argument("--min_lr", type=float, default=0.0)  # Minimum learning rate
    parser.add_argument("--decay_steps", type=int, default=50)  # Steps between LR decay
    parser.add_argument("--decay_rate", type=float, default=0.9)  # Learning rate decay factor
    parser.add_argument("--patience", type=int, default=20)  # Scheduler patience epochs
    parser.add_argument("--amortisation_param", type=float, default=1)  # Amortization parameter
    parser.add_argument("--evaluation_fn", type=str, default="dataset_space_saving")  # Evaluation function name
    parser.add_argument("--num_epochs", type=int, default=100)  # Total training epochs
    parser.add_argument("--lr_policy", type=float, default=0.001)  # Policy network learning rate
    parser.add_argument("--lr_dict", type=float, default=0.1)  # Dictionary learning rate

    parser.add_argument("--bits_per_parameter", type=int, default=16)  # Bits per model parameter

    # ----------------- Special cases (fixed dictionary and/or candidate subgraphs)
    # initial dictionary (optional):
    parser.add_argument("--atom_types", type=parse.str2list2str, default=[])  # Initial dictionary atom types
    parser.add_argument("--k", type=parse.str2list2int, default=[])  # K-values for atom generation
    parser.add_argument(
        "--custom_edge_lists", type=parse.str2ListOfListsOfLists2int, default=None
    )  # Custom edge lists for atoms
    # subgraph isomorphism (optional):
    parser.add_argument("--candidate_subgraphs", type=parse.str2bool, default=False)  # Use pre-computed candidate subgraphs
    parser.add_argument("--multiprocessing", type=parse.str2bool, default=False)  # Enable multiprocessing
    parser.add_argument("--num_processes", type=int, default=64)  # Number of parallel processes

    args = parser.parse_args()
    print(args)
    main(vars(args))