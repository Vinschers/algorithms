from ..data_structures import Graph

def dijkstra(graph: Graph, source: int):
    if source not in graph.vertices:
        raise ValueError(f"Vertex {source} not found in the graph.")


def bellman_ford():
    pass


def floyd_warshall():
    pass
