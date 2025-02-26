from vicentin import Graph
from vicentin.image.utils import get_neighbors


def _compute_edges(H, W, L, img, weight_func, neighborhood):
    """
    Computes the weighted edges of a graph representation of an image.

    This function flattens the image into a 1D array and calculates edges between
    neighboring pixels using a given weight function. The edges are stored as a list
    of tuples representing the source node, target node, and edge weight.

    Args:
        H (int): The height (number of rows) of the image.
        W (int): The width (number of columns) of the image.
        L (int): The depth (number of channels) of the image.
        img (numpy.ndarray or jnp.ndarray): A 1D, 2D, or 3D NumPy array representing the image.
        weight_func (callable): A function `f(u, v) -> float` that computes edge
                                weights based on pixel values.
        neighborhood (int): The type of neighborhood connectivity:
            - `4`: Only direct neighbors.
            - `8`: Includes diagonal neighbors.

    Returns:
        list[tuple[int, int, float]]: A list of tuples representing graph edges.
            Each tuple `(u, v, weight)` contains:
                - `u` (int): The index of the source node.
                - `v` (int): The index of the target node (neighbor).
                - `weight` (float): The weight computed between the two pixels.

    Time Complexity:
        - O(N), where N = H * W * L (each pixel is visited once).

    Space Complexity:
        - O(E), where E is the number of edges (depends on neighborhood type).
    """
    edges = []
    flatten_img = img.flatten()

    for row in range(H):
        for col in range(W):
            for k in range(L):
                u_id = row * (W * L) + col * L + k

                for neighbor in get_neighbors(img, row, col, k, neighborhood):
                    v_id = neighbor[0] * (W * L) + neighbor[1] * L + neighbor[2]

                    weight = weight_func(flatten_img[u_id], flatten_img[v_id])
                    edges.append((u_id, v_id, weight))

    return edges


def img2graph(img, weight_func=None, neighborhood=4):
    """
    Converts a 1D, 2D, or 3D image into a graph representation.

    Each pixel in the image is treated as a node, and edges are created between
    neighboring pixels based on the specified neighborhood type. The edge weights
    are determined by a user-defined function (defaulting to squared intensity difference).

    Args:
        img (numpy.ndarray or jnp.ndarray): An array of shape (N,), (N, M), or (N, M, L)
                             representing a 1D, 2D, or 3D image.
        weight_func (callable, optional): A function `f(u, v) -> float` that computes edge
                                          weights based on pixel values.
                                          Defaults to `(u - v) ** 2`.
        neighborhood (int, optional): The type of neighborhood connectivity:
            - `4`: Only direct neighbors.
            - `8`: Includes diagonal neighbors.
            Defaults to `4`.

    Returns:
        Graph: A `vicentin.Graph` instance where:
            - Each pixel corresponds to a node.
            - Edges connect neighboring pixels with weights computed by `weight_func`.

    Time Complexity:
        - O(N), where N = H * W * L (each pixel is visited once).

    Space Complexity:
        - O(N + E), where E is the number of edges (depends on neighborhood type).

    Raises:
        ValueError: If `img` has more than 3 dimensions.
    """
    if img.ndim > 3:
        raise ValueError("Image must be 1D, 2D, or 3D")

    if img.ndim == 1:
        H, W, L = img.shape[0], 1, 1
    elif img.ndim == 2:
        H, W, L = img.shape[0], img.shape[1], 1
    else:
        H, W, L = img.shape

    if weight_func is None:
        weight_func = lambda u, v: (u - v) ** 2

    graph = Graph(H * W * L)

    for u, v, w in _compute_edges(H, W, L, img, weight_func, neighborhood):
        graph.add_edge(u, v, w)

    return graph
