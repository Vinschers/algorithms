from vicentin.data_structures import Graph
from vicentin.graph.shortest_path import bellman_ford


def negative_cycle(graph: Graph, source: int):
    """
    Detects and returns a negative weight cycle in a weighted graph, if one exists.

    This function leverages the Bellman-Ford algorithm to compute the shortest paths from the
    given source vertex to all other vertices. After obtaining the distance and predecessor
    mappings, it iterates through all edges to check for further relaxations—an indication of a
    negative weight cycle. If such a cycle is detected, the function reconstructs the cycle by
    tracing back through the predecessor pointers and returns the cycle as an iterator in reversed
    order. If no negative cycle is found, the function returns an empty list.

    Time Complexity:
        - O(VE), where V is the number of vertices and E is the number of edges.

    Space Complexity:
        - O(V), primarily for storing distances, predecessor mappings, and visitation flags.

    Args:
        graph (Graph): The graph in which to search for a negative weight cycle.
        source (int): The ID of the source vertex from which to begin the search.

    Returns:
        Iterator[int] or list[int]:
            - An iterator (or list) of vertex IDs that forms the negative weight cycle,
              if a negative cycle is detected.
            - An empty list if no negative weight cycle exists.

    Raises:
        ValueError: If the source vertex is not found in the graph (as detected by the Bellman-Ford algorithm).
    """
    dist, prev = bellman_ford(graph, source)

    for _, edge in graph.edges:
        u, v, w = edge.source, edge.target, edge.weight

        # check for negative cycle
        if dist[u] + w < dist[v]:
            visited = [False] * graph.n

            visited[v] = True
            while u and not visited[u]:
                visited[u] = True
                u = prev[u]

            cycle = [u]
            v = prev[u]

            while v != u:
                cycle.append(v)
                v = prev[u]

            return reversed(cycle)

    return []
