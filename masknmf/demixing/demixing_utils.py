import numpy as np
import scipy.sparse

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import torch
from typing import *
import networkx as nx
from collections import defaultdict


def construct_graph_from_sparse_tensor(adj_tensor: torch.sparse_coo_tensor) -> nx.Graph:
    """
    Constructs a NetworkX graph from a sparse COO tensor representing an adjacency matrix.

    Args:
        adj_tensor (torch.sparse_coo_tensor): A sparse COO tensor representing the adjacency matrix of the graph.

    Returns:
        nx.Graph: A NetworkX graph constructed from the sparse tensor, including all nodes.
    """
    # Move tensor to CPU if it's on CUDA

    curr_indices = adj_tensor.indices().cpu()
    # Convert indices to NumPy arrays
    rows = curr_indices[0].numpy()
    cols = curr_indices[1].numpy()

    # Determine the number of nodes
    num_nodes = adj_tensor.shape[0]

    # Create the NetworkX graph
    graph = nx.Graph()

    # Add all nodes to the graph
    graph.add_nodes_from(range(num_nodes))

    # Add edges to the graph
    graph.add_edges_from(zip(rows, cols))

    return graph


def color_and_get_tensors(graph: nx.Graph, device: str) -> List[torch.Tensor]:
    """
    Color the nodes of a graph using a greedy coloring algorithm and convert the
    resulting color groups into PyTorch tensors.

    Args:
        graph (nx.Graph): The input graph to be colored.

    Returns:
        List[torch.Tensor]: A list of PyTorch tensors, where each tensor contains
        the nodes corresponding to a specific color. The list is ordered by color.
    """
    # Compute coloring using greedy algorithm
    coloring = nx.coloring.greedy_color(graph, strategy="largest_first")

    # Create a dictionary mapping colors to lists of nodes
    color_to_nodes: Dict[int, List[int]] = defaultdict(list)
    for node, color in coloring.items():
        color_to_nodes[color].append(node)

    # Convert lists of nodes to PyTorch tensors
    color_to_tensors: List[torch.Tensor] = [
        torch.tensor(nodes, dtype=torch.long).to(device)
        for nodes in color_to_nodes.values()
    ]

    return color_to_tensors


def torch_sparse_to_scipy_coo(a):
    data = a.values().cpu().detach().numpy()
    row = a.indices().cpu().detach().numpy()[0, :]
    col = a.indices().cpu().detach().numpy()[1, :]
    return scipy.sparse.coo_matrix((data, (row, col)), a.shape)


def ndarray_to_torch_sparse_coo(my_array: np.ndarray):
    """
    Args:
        my_array (np.ndarray):
    Returns:
        torch.sparse_coo_tensor: A torch.sparse_coo_tensor from the input array
    """
    rows, cols = my_array.nonzero()
    values = my_array[rows, cols]

    # Create the indices tensor (2 x N, where N is the number of non-zero elements)
    indices = torch.tensor(np.vstack([rows, cols]), dtype=torch.long)
    values = torch.tensor(values, dtype=torch.float32)

    # Create the sparse COO tensor
    torch_sparse_representation = torch.sparse_coo_tensor(
        indices, values, my_array.shape
    )
    return torch_sparse_representation


def scipy_sparse_to_torch(scipy_sparse):
    """
    Converts a scipy sparse matrix (any format) to a PyTorch sparse COO tensor.

    Args:
        scipy_sparse (scipy.sparse.spmatrix): The input sparse matrix in any scipy sparse format (CSR, CSC, COO, etc.).

    Returns:
        torch.sparse_coo_tensor: The corresponding PyTorch sparse COO tensor.

    Raises:
        TypeError: If the input is not a scipy.sparse.spmatrix.
    """
    # Type checking
    if not isinstance(scipy_sparse, scipy.sparse.spmatrix):
        raise TypeError(
            f"Expected input to be a scipy.sparse.spmatrix, but got {type(scipy_sparse)}"
        )

    # Convert to COO format if it's not already
    scipy_coo = scipy_sparse.tocoo()

    # Extract row, column indices and values
    row = torch.tensor(scipy_coo.row, dtype=torch.long)
    col = torch.tensor(scipy_coo.col, dtype=torch.long)
    values = torch.tensor(scipy_coo.data, dtype=torch.float32)

    # Stack row and column into a 2 x N indices tensor
    indices = torch.stack([row, col], dim=0)

    # Create the torch sparse COO tensor
    sparse_tensor = torch.sparse_coo_tensor(indices, values, scipy_sparse.shape)

    return sparse_tensor


def torch_dense_to_sparse_coo(dense_tensor):
    """
    Converts a 2D dense PyTorch tensor to a sparse COO tensor.

    Args:
        dense_tensor (torch.Tensor): The input dense tensor (must be 2D).

    Returns:
        torch.sparse_coo_tensor: A PyTorch sparse COO tensor representing the input tensor.

    Raises:
        ValueError: If the input tensor is not 2D.
    """
    # Ensure the input is a 2D tensor
    if dense_tensor.dim() != 2:
        raise ValueError(
            f"Expected a 2D tensor, but got a tensor with {dense_tensor.dim()} dimensions."
        )

    # Get the non-zero indices
    non_zero_indices = dense_tensor.nonzero(as_tuple=True)

    # Stack the indices (rows and columns) to create the indices tensor
    indices = torch.stack(non_zero_indices, dim=0)

    # Get the non-zero values from the dense tensor
    values = dense_tensor[non_zero_indices]

    # Create the sparse COO tensor using the indices, values, and shape of the dense tensor
    sparse_tensor = torch.sparse_coo_tensor(indices, values, dense_tensor.shape)

    return sparse_tensor


def show_img(ax, img):
    # Visualize local correlation, adapt from kelly's code
    im = ax.imshow(img, cmap="jet")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(im, cax=cax, orientation="vertical", spacing="uniform")


def spatial_sum_plot(a, a_fin, patch_size, order="C", num_list_fin=None, text=False):
    scale = np.maximum(1, (patch_size[1] / patch_size[0]))
    fig = plt.figure(figsize=(16 * scale, 8))
    ax = plt.subplot(1, 2, 1)
    ax.imshow(a_fin.sum(axis=1).reshape(patch_size, order=order), cmap="jet")

    if num_list_fin is None:
        num_list_fin = np.arange(a_fin.shape[1])
    if text:
        for ii in range(a_fin.shape[1]):
            temp = a_fin[:, ii].reshape(patch_size, order=order)
            pos0 = np.where(temp == temp.max())[0][0]
            pos1 = np.where(temp == temp.max())[1][0]
            ax.text(
                pos1,
                pos0,
                f"{num_list_fin[ii] + 1}",
                verticalalignment="bottom",
                horizontalalignment="right",
                color="white",
                fontsize=15,
                fontweight="bold",
            )

    ax.set(title="more passes spatial components")
    ax.title.set_fontsize(15)
    ax.title.set_fontweight("bold")

    ax1 = plt.subplot(1, 2, 2)
    ax1.imshow(a.sum(axis=1).reshape(patch_size, order=order), cmap="jet")

    if text:
        for ii in range(a.shape[1]):
            temp = a[:, ii].reshape(patch_size, order=order)
            pos0 = np.where(temp == temp.max())[0][0]
            pos1 = np.where(temp == temp.max())[1][0]
            ax1.text(
                pos1,
                pos0,
                f"{ii + 1}",
                verticalalignment="bottom",
                horizontalalignment="right",
                color="white",
                fontsize=15,
                fontweight="bold",
            )

    ax1.set(title="1 pass spatial components")
    ax1.title.set_fontsize(15)
    ax1.title.set_fontweight("bold")
    plt.tight_layout()
    plt.show()
    return fig


def cosine_similarity(img1, img2):
    """
    Calculates cosine similarity between two 2D images
    Args:
        img1: first image being compared
        img2: second image being compared
    Returns:
        cosine_sim: cosine similarity between these two images
    """
    if np.count_nonzero(img1 != 0) == 0 or np.count_nonzero(img2 != 0) == 0:
        print("one of these arrays is zero!!!")
        if np.count_nonzero(img2 != 0) == 0:
            print("the second one was 0")
        return 0
    img1_flat = img1.flatten()
    img2_flat = img2.flatten()

    img1_normal = img1_flat / np.linalg.norm(img1_flat)
    img2_normal = img2_flat / np.linalg.norm(img2_flat)

    cosine_sim = img1_normal.T.dot(img2_normal)
    return cosine_sim


def normalize_traces(trace1, trace2):
    """
    Normalizes trace
    Args:
        trace1: First trace (provided as an ndarray)
        trace2: Second trace (provided as ndarray)

    Returns:
        trace1_norm: Normalized trace1
        trace2_norm: Normalized trace2
    """

    if np.count_nonzero(trace1 != 0) == 0:
        trace1_norm = np.zeros_like(trace_1)
    else:
        trace1_norm = trace1 / np.linalg.norm(trace1)

    if np.count_nonzero(trace2 != 0) == 0:
        trace2_norm = np.zeros_like(trace_2)
    else:
        trace2_norm = trace2 / np.linalg.norm(trace2)

    return trace1_norm, trace2_norm


def get_box(img):
    """
    For a given frame in the dataset, this function calculates its bounding box
        Args:
            img (np.ndarray): Shape (d1 x d2). The image to analyze

        Returns:
            [height_min, height_max, width_min, width_max]: a list of bounding coordinates which can be used to crop original image

    """

    # If all pixels are 0, there is no need to crop (the image is empty)
    if np.count_nonzero(img) == 0:
        return 0, img.shape[0], 0, img.shape[1]

    # Calculate bounding box by finding minimal elements in the support
    else:
        x, y = np.nonzero(img)
        return int(np.amin(x)), int(np.amax(x)), int(np.amin(y)), int(np.amax(y))



