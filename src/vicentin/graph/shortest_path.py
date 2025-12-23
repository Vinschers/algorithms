from math import inf
from collections import defaultdict

from vicentin.data_structures import Graph, PriorityQueue


def dijkstra(graph: Graph, source: int):
    """
    Computes the shortest paths from a given source vertex to all other vertices in a weighted graph
    using Dijkstra's algorithm.

    This implementation uses a priority queue (min-heap) to efficiently extract the minimum-distance vertex.
    It supports graphs with non-negative edge weights.

    Time Complexity:
        - O((V + E) log V), where V is the number of vertices and E is the number of edges.

    Space Complexity:
        - O(V), for storing distances and predecessors.

    Args:
        graph (Graph): The graph on which to run Dijkstra's algorithm.
        source (int): The ID of the source vertex.

    Returns:
        tuple[dict[int, float], dict[int, int | None]]:
            - A dictionary `dist` mapping each vertex ID to its shortest distance from the source.
            - A dictionary `prev` mapping each vertex ID to its predecessor in the shortest path.
              If a vertex is unreachable from the source, it will have `None` as its predecessor.

    Raises:
        ValueError: If the source vertex is not found in the graph.
    """
    if source not in graph.vertices:
        raise ValueError(f"Vertex {source} not found in the graph.")

    pq = PriorityQueue(max=False)
    dist = defaultdict(lambda: inf)
    prev = defaultdict(lambda: None)

    dist[source] = 0
    pq.enqueue(source, 0)

    while not pq.is_empty():
        _, u = pq.dequeue()
        for _, edge in graph.vertices[u].neighbors:
            v = edge.target
            w = edge.weight

            alt = dist[u] + w
            if alt < dist[v]:
                prev[v] = u
                dist[v] = alt

                if v in pq:
                    pq.update(v, alt)
                else:
                    pq.enqueue(v, alt)

    return dist, prev


def bellman_ford(graph: Graph, source: int):
    """
    Computes the shortest paths from a given source vertex to all other vertices in a weighted graph
    using the Bellman-Ford algorithm.

    Unlike Dijkstra’s algorithm, Bellman-Ford can handle graphs with negative edge weights.
    It iterates over all edges multiple times to relax distances and detect negative weight cycles.

    Time Complexity:
        - O(VE), where V is the number of vertices and E is the number of edges.

    Space Complexity:
        - O(V), for storing distances and predecessors.

    Args:
        graph (Graph): The graph on which to run the Bellman-Ford algorithm.
        source (int): The ID of the source vertex.

    Returns:
        tuple[dict[int, float], dict[int, int | None]]:
            - A dictionary `dist` mapping each vertex ID to its shortest distance from the source.
            - A dictionary `prev` mapping each vertex ID to its predecessor in the shortest path.
              If a vertex is unreachable from the source, it will have `None` as its predecessor.

    Raises:
        ValueError: If the source vertex is not found in the graph.
        ValueError: If a negative weight cycle is detected in the graph.
    """

    if source not in graph.vertices:
        raise ValueError(f"Vertex {source} not found in the graph.")

    dist = defaultdict(lambda: inf)
    prev = defaultdict(lambda: None)

    dist[source] = 0

    for _ in range(graph.n - 1):
        for _, edge in graph.edges:
            u, v, w = edge.source, edge.target, edge.weight

            alt = dist[u] + w
            if alt < dist[v]:
                dist[v] = alt
                prev[v] = u

    for _, edge in graph.edges:
        u, v, w = edge.source, edge.target, edge.weight
        if dist[u] + w < dist[v]:
            raise ValueError("Graph contains a negative weight cycle")

    return dist, prev


def floyd_warshall(graph: Graph):
    """
    Computes the shortest paths between all pairs of vertices in a weighted graph
    using the Floyd–Warshall algorithm.

    Unlike Dijkstra’s and Bellman–Ford algorithms, which compute paths from a single source,
    Floyd–Warshall simultaneously computes the shortest paths between every pair of vertices.
    It can handle graphs with negative edge weights and detect negative weight cycles.

    Time Complexity:
        - O(V^3), where V is the number of vertices.

    Space Complexity:
        - O(V^2), for storing the distance matrix.

    Args:
        graph (Graph): The graph on which to run the Floyd–Warshall algorithm.

    Returns:
        dict[int, dict[int, float]]:
            - A nested dictionary `dists` where `dists[u][v]` represents the shortest distance
              from vertex `u` to vertex `v`. If no path exists between `u` and `v`,
              the value will be infinity.

    Raises:
        ValueError: If a negative weight cycle is detected in the graph.
    """
    dists = defaultdict(lambda: defaultdict(lambda: inf))

    for edge in graph.edges:
        u, v, w = edge.source, edge.target, edge.weight
        dists[u][v] = w

    for v in graph.vertices:
        dists[v][v] = 0

    for k in graph.vertices:
        for i in graph.vertices:
            for j in graph.vertices:
                dists[i][j] = min(dists[i][j], dists[i][k] + dists[k][j])

    for v in graph.vertices:
        if dists[v][v] < 0:
            raise ValueError("Graph contains a negative weight cycle")

    return dists
