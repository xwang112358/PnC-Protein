import torch
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import math
import wandb
from utils.conversions import convert_csr_to_nx
# import graph_tool as gt  <- REMOVED
# import graph_tool.stats as gt_stats  <- REMOVED
# import graph_tool.inference as gt_inference  <- REMOVED
from matplotlib import cm
plt.switch_backend("cairo")

colormap = cm.get_cmap("Set1")


def visualise_intermediate(
    visualise_data_dict,
    inds_to_visualise,
    graph,
    mask,
    subgraphs,
    atom_indices=None,
    attr_mapping=None,
    node_attr_dims=None,
    edge_attr_dims=None,
):
    """
    (Original function - no changes needed as it doesn't use graph_tool)
    """
    for ind_to_visualise in inds_to_visualise:
        # Create node mask for current graph index
        n_mask = (graph.batch == ind_to_visualise) & mask
        if n_mask.sum() == 0:
            continue

        # Create edge mask based on node mask
        edge_mask = n_mask[graph.edge_index[0]] & n_mask[graph.edge_index[1]]

        # Extract edges and nodes for visualization
        edges_vis = graph.edge_index[:, edge_mask]
        nodes_vis = torch.where(n_mask)[0]

        # Create NetworkX graph for visualization
        G_vis = nx.Graph()
        node_label_list = nodes_vis.tolist()
        edge_label_list = edges_vis.transpose(1, 0).tolist()
        G_vis.add_nodes_from(node_label_list)
        G_vis.add_edges_from(edge_label_list)

        # Map attributes if provided
        if node_attr_dims is not None or edge_attr_dims is not None:
            node_attrs, edge_attrs = attr_mapping.map(
                graph.x[n_mask],
                graph.edge_features[edge_mask]
                if hasattr(graph, "edge_features")
                else None,
            )

        # Create node labels from attributes
        if node_attr_dims is not None:
            node_labels = {
                node_label: ",".join(
                    [
                        attr_mapping.node_attr_values[i][node_attrs[node_ind, i].item()]
                        for i in range(node_attr_dims)
                    ]
                )
                for node_ind, node_label in enumerate(node_label_list)
            }
        else:
            node_labels = None

        # Create edge labels from attributes
        if edge_attr_dims is not None:
            edge_labels = {
                (edge_label[0], edge_label[1]): ",".join(
                    [
                        attr_mapping.edge_attr_values[i][edge_attrs[edge_ind, i].item()]
                        for i in range(edge_attr_dims)
                    ]
                )
                for edge_ind, edge_label in enumerate(edge_label_list)
            }
        else:
            edge_labels = None

        # Set colors based on subgraph membership
        node_color = [
            "b" if i not in subgraphs[ind_to_visualise] else "r"
            for i in list(G_vis.nodes())
        ]
        edge_color = [
            "k"
            if e[0] not in subgraphs[ind_to_visualise]
            or e[1] not in subgraphs[ind_to_visualise]
            else "r"
            for e in list(G_vis.edges())
        ]
        edge_width = [
            1
            if e[0] not in subgraphs[ind_to_visualise]
            or e[1] not in subgraphs[ind_to_visualise]
            else 2
            for e in list(G_vis.edges())
        ]

        # Store visualization data
        curr_visualisation = {
            "G": G_vis,
            "cluster_labels": subgraphs[ind_to_visualise],
            "node_color": node_color,
            "edge_color": edge_color,
            "edge_width": edge_width,
            "node_labels": node_labels,
            "edge_labels": edge_labels,
            "atom_index": atom_indices[ind_to_visualise],
        }

        # Add to visualization dictionary
        if ind_to_visualise not in visualise_data_dict:
            visualise_data_dict[ind_to_visualise] = [curr_visualisation]
        else:
            visualise_data_dict[ind_to_visualise].append(curr_visualisation)
    return visualise_data_dict


def visualise_partitioning_nx(
    G, cluster_labels_list, mv_fractional, ax, hs_scaling=3, node_size=200
):
    """
    Visualize graph partitioning using NetworkX and Matplotlib.
    This function replaces the original `visualise_partitioning`.

    Args:
        G (nx.Graph): NetworkX graph to visualize.
        cluster_labels_list (list): List of cluster labels for each node.
        mv_fractional (list): Fractional values for vertex sizing.
        ax (matplotlib.axes.Axes): Matplotlib axis to draw on.
        hs_scaling (int): Scaling factor for halo sizes.
        node_size (int): Base size for the graph nodes.
    
    Returns:
        int: The number of clusters.
    """
    # Create a mapping from node to its cluster index for easy lookup
    node_to_cluster = {
        node: i for i, cluster in enumerate(cluster_labels_list) for node in cluster
    }
    
    # Generate colors for each node based on its cluster
    num_clusters = len(cluster_labels_list)
    colors = [colormap(node_to_cluster.get(node, -1) / num_clusters) for node in G.nodes()]

    # Calculate halo sizes based on mv_fractional
    # The halo is a larger circle drawn behind the node
    halo_sizes = [
        node_size * (1 + hs_scaling * (1 - mv_fractional[node_to_cluster.get(node, -1)]))
        for node in G.nodes()
    ]

    # Use a layout that attempts to cluster nodes
    pos = nx.spring_layout(G, iterations=100, seed=42)

    # 1. Draw the edges
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.5, width=0.5, edge_color="k")

    # 2. Draw the halos (larger, semi-transparent nodes)
    nx.draw_networkx_nodes(
        G,
        pos,
        ax=ax,
        node_size=halo_sizes,
        node_color=colors,
        alpha=0.3, # Make halos transparent
    )

    # 3. Draw the actual nodes on top of the halos
    nx.draw_networkx_nodes(
        G,
        pos,
        ax=ax,
        node_size=node_size,
        node_color=colors,
    )
    
    ax.get_figure().set_facecolor('white') # Set a neutral background

    return num_clusters

def visualisation_log(
    visualise_data_dict,
    inds_to_visualise,
    visualise_step,
    total_costs=None,
    baseline=None,
    cost_terms=None,
    x_a_fractional=None,
    partioning=True,
    directed=False,
    subgraphs=None,
    graphs=None,
):
    """
    (This function is updated to use the new `visualise_partitioning_nx`)
    """
    # Handle case where visualization data dict is None but subgraphs exist
    if visualise_data_dict is None and subgraphs is not None:
        for ind_to_visualise in inds_to_visualise:
            fig, ax = plt.subplots(figsize=(10, 10)) # Create figure and axis

            # Extract cluster information
            cluster_labels_list = [
                cluster_labels.tolist()
                for cluster_labels in subgraphs[ind_to_visualise]["cluster_labels"]
            ]

            # Process fractional values
            mv_fractional = x_a_fractional[
                subgraphs[ind_to_visualise]["atom_indices"][0]
            ].tolist()
            if isinstance(mv_fractional, list):
                 for i in range(len(mv_fractional)):
                     if mv_fractional[i] == -1:
                         mv_fractional[i] = 0
            else:
                if mv_fractional == -1:
                    mv_fractional = 0


            # Create visualization graph
            G_vis = nx.Graph()
            G_vis.add_nodes_from(list(range(int(graphs[ind_to_visualise].graph_size))))
            G_vis.add_edges_from(
                graphs[ind_to_visualise].edge_index.transpose(1, 0).tolist()
            )
            # Clean up graph structure
            G_vis.remove_edges_from(nx.selfloop_edges(G_vis))


            # Generate partitioning visualization using the new function
            num_clusters = visualise_partitioning_nx(
                G_vis, cluster_labels_list, mv_fractional, ax, hs_scaling=3
            )

            # Set title with cost information
            baseline_dl = baseline[ind_to_visualise] if baseline is not None else 0
            dl = total_costs[ind_to_visualise]
            if cost_terms is None:
                fig.suptitle(
                    "num clusters: {}, baseline dl: {:.2f}, dl: {:.2f}".format(
                        num_clusters, baseline_dl, dl
                    )
                )
            else:
                print_log = "num clusters: {}, baseline dl: {:.2f}, dl: {:.2f}, b: {:.2f}, H_dict: {:.2f}, H_null: {:.2f}, C: {:.2f}, "
                # Assuming cost_terms are now stored differently or need to be fetched
                # For this example, let's assume cost terms are available directly
                cost_data = [
                     cost_terms["b"][ind_to_visualise].item(),
                     cost_terms["H_dict"][ind_to_visualise].item(),
                     cost_terms["H_null"][ind_to_visualise].item(),
                     cost_terms["C"][ind_to_visualise].item(),
                ]
                print_data = [num_clusters, baseline_dl, dl] + cost_data
                fig.suptitle(print_log.format(*print_data))

            # Log to wandb
            wandb.log(
                {"graph_partitioning/graph_partitioning_" + str(ind_to_visualise): wandb.Image(plt)},
                step=visualise_step,
            )
            plt.close()
    else:

        for ind_to_visualise in inds_to_visualise:
            if ind_to_visualise in visualise_data_dict:
                # Store cost information
                if baseline is not None:
                    visualise_data_dict[ind_to_visualise][0]["dl"] = (
                        baseline[ind_to_visualise].item(),
                        total_costs[ind_to_visualise].item(),
                    )
                else:
                    visualise_data_dict[ind_to_visualise][0]["dl"] = (
                        0,
                        total_costs[ind_to_visualise].item(),
                    )

                # Store detailed cost terms
                if cost_terms is not None:
                    visualise_data_dict[ind_to_visualise][0]["cost_terms"] = [
                        cost_terms["b"][ind_to_visualise].item(),
                        cost_terms["H_dict"][ind_to_visualise].item(),
                        cost_terms["H_null"][ind_to_visualise].item(),
                        cost_terms["C"][ind_to_visualise].item(),
                    ]

                visualise_data_list = visualise_data_dict[ind_to_visualise]

                # Calculate subplot layout
                num_plots = len(visualise_data_list)
                if partioning:
                    num_plots += 1
                num_rows = int(math.ceil(num_plots / 2))

                fig, ax = plt.subplots(num_rows, 2, figsize=(18, num_rows * 9))
                ax = ax.flatten() # Flatten axes array for easy indexing

                # Process fractional values for each visualization
                if x_a_fractional is not None:
                    mv_fractional = [
                        x_a_fractional[visualise_data["atom_index"]].item()
                        if visualise_data["atom_index"] != -1
                        else 0
                        for visualise_data in visualise_data_list
                    ]
                else:
                    mv_fractional = [0 for _ in visualise_data_list]

                # Draw each visualization
                for i, visualise_data in enumerate(visualise_data_list):
                    ax_i = ax[i]
                    pos = nx.spring_layout(visualise_data["G"], scale=2, seed=42)

                    # Draw edge labels if available
                    if visualise_data["edge_labels"] is not None:
                        nx.draw_networkx_edge_labels(
                            visualise_data["G"],
                            pos=pos,
                            edge_labels=visualise_data["edge_labels"],
                            ax=ax_i,
                        )

                    # Draw the graph
                    nx.draw(
                        visualise_data["G"],
                        pos=pos,
                        node_color=visualise_data["node_color"],
                        edge_color=visualise_data["edge_color"],
                        width=visualise_data["edge_width"],
                        with_labels=True,
                        labels=visualise_data["node_labels"],
                        ax=ax_i,
                    )

                    # Set subplot title with cost information
                    if "dl" in visualise_data:
                         print_value = [
                            visualise_data["dl"][0],
                            visualise_data["dl"][1],
                            visualise_data["atom_index"], # No .item() needed if it's already a number
                            mv_fractional[i],
                         ]
                    else:
                         print_value = [0, 0, visualise_data["atom_index"], mv_fractional[i]]

                    ax_i.set_title(
                        "Baseline DL: {:.4f}, DL: {:.4f}, atom_index: {}, mv: {:.4f}".format(*print_value)
                    )

                # Add partitioning visualization if requested
                if partioning:
                    ax_i = ax[len(visualise_data_list)] # Use the next available subplot
                    cluster_labels_list = [
                        visualise_data["cluster_labels"].tolist()
                        for visualise_data in visualise_data_list
                    ]
                    
                    # Call the new visualization function
                    num_clusters = visualise_partitioning_nx(
                        visualise_data_list[0]["G"],
                        cluster_labels_list,
                        mv_fractional,
                        ax_i, # Pass the specific axis
                    )

                    # Set partitioning subplot title
                    if "dl" in visualise_data_list[0]:
                        baseline_dl = visualise_data_list[0]["dl"][0]
                        dl = visualise_data_list[0]["dl"][1]
                        if cost_terms is None:
                            ax_i.set_title(
                                "Num Clusters: {}, Baseline DL: {:.2f}, DL: {:.2f}".format(
                                    num_clusters, baseline_dl, dl
                                )
                            )
                        else:
                            print_log = "Num Clusters: {}, DL: {:.2f}, b: {:.2f}, H_dict: {:.2f}, H_null: {:.2f}, C: {:.2f}"
                            cost_data = visualise_data_dict[ind_to_visualise][0]["cost_terms"]
                            print_data = [num_clusters, dl] + cost_data
                            ax_i.set_title(print_log.format(*print_data))
                    else:
                        ax_i.set_title("Num Clusters: {}".format(num_clusters))
                
                # Hide unused subplots
                for i in range(num_plots, len(ax)):
                    ax[i].set_visible(False)

                plt.tight_layout()
                # Log to wandb
                wandb.log(
                    {"graphs/graph_" + str(ind_to_visualise): wandb.Image(plt)},
                    step=visualise_step,
                )
                plt.close()

    return


def visualise_subgraphs(
    atoms,
    init_i,
    final_i,
    visualise_step=0,
    data_format="nx",
    attr_mapping=None,
    node_attr_dims=None,
    edge_attr_dims=None,
    color_attrs=True,
):
    """
    Visualize individual subgraphs with their attributes and structure.
    
    Args:
        atoms: List of atom/graph objects to visualize
        init_i (int): Starting index for visualization
        final_i (int): Ending index for visualization
        visualise_step (int): Current step for logging
        data_format (str): Format of input data ('nx' or 'csr')
        attr_mapping: Mapping for node and edge attributes
        node_attr_dims: Number of node attribute dimensions
        edge_attr_dims: Number of edge attribute dimensions
        color_attrs (bool): Whether to color nodes based on attributes
    """
    for i in range(init_i, final_i):
        # G_s = nx.Graph()
        # G_s.add_edges_from(edge_lists[i].cpu().numpy().transpose(1,0))
        # G_s.add_nodes_from(range(0,int(num_vertices[i])))
        fig = plt.figure(figsize=(8, 8), dpi=300)
        
        # Convert graph format if necessary
        if data_format == "csr":
            G = convert_csr_to_nx(atoms[i])
        else:
            G = atoms[i]
            
        # Generate layout for visualization
        pos = nx.spring_layout(G, scale=2)
        node_colors = []
        
        # Process node attributes if available
        if node_attr_dims is not None:
            if color_attrs:
                # Handle coloring based on attributes
                if len(attr_mapping.node_attr_values) > 1:
                    print("fix this")  # TODO: Handle multiple attribute types
                    import pdb
                    pdb.set_trace()
                else:
                    pass
                    # groups = set(attr_mapping.node_attr_values[0])
                    # mapping = dict(zip(count(), sorted(groups)))
                    
            # Create node labels from attributes
            node_labels = {}
            for node in G.nodes(data=True):
                node_ind = node[0]
                attr_dict = node[1]
                # print(f'DEBUG: Node {node_ind} attr_dict: {attr_dict}')
                node_labels[node_ind] = ",".join(
                    [
                        attr_mapping.node_attr_values[int(k)][v]
                        for k, v in attr_dict.items()
                    ]
                )
                # Assign colors based on actual attribute values
                # Use the actual attribute value for coloring, not just the first dimension
                if len(attr_dict) > 0:
                    # For single-dimensional attributes, use the attribute value directly
                    if len(attr_dict) == 1:
                        attr_val = attr_dict["0"]
                        # print(f'DEBUG: Node {node_ind} attr_val: {attr_val}, amino acid: {attr_mapping.node_attr_values[0][attr_val]}')
                        node_colors.append(
                            colormap(attr_val / len(attr_mapping.node_attr_values[0]))
                        )
                    else:
                        # For multi-dimensional attributes, use the first non-zero or most significant attribute
                        primary_attr = attr_dict["0"]  # Can be extended to be more sophisticated
                        node_colors.append(
                            colormap(primary_attr / len(attr_mapping.node_attr_values[0]))
                        )
                else:
                    node_colors.append(colormap(0.0))
        else:
            # print("DEBUG: No attr_mapping available, using default colors")
            node_labels = None
            
        # Process edge attributes if available
        if edge_attr_dims is not None:
            edge_labels = {}
            for edge in G.edges(data=True):
                edge_ind = (edge[0], edge[1])
                attr_dict = edge[2]
                edge_labels[edge_ind] = ",".join(
                    [
                        attr_mapping.edge_attr_values[int(k)][v]
                        for k, v in attr_dict.items()
                    ]
                )
            # Draw edge labels
            nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=25)
            
        # Draw the graph with appropriate styling
        if len(node_colors) == 0:
            nx.draw(G, pos=pos, with_labels=False, labels=node_labels)
        else:
            nx.draw(
                G,
                pos,
                nodelist=G.nodes(),
                with_labels=True,
                labels=node_labels,
                node_color=node_colors,
                node_size=1000,
                width=4.0,
                font_size=25,
            )
            # plt.colorbar(nc)
            # plt.axis('off')
        # nx.draw(G, with_labels=True);
        
        # Log to wandb
        wandb.log({"subgraphs/subgraph_" + str(i): wandb.Image(plt)}, step=visualise_step)
        # plt.savefig('./images/MUTAG/subgraph_'+str(i)+'.svg')
        # plt.savefig('./images/MUTAG/subgraph_' + str(i) + '.svg')
        plt.close()

    return
