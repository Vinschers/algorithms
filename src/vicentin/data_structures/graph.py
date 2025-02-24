from collections import defaultdict

import networkx as nx
import matplotlib.pyplot as plt


class Vertex:
    """
    A simple wrapper to represent a graph vertex.

    Stores:
    - The vertex's unique identifier.
    - Optional arbitrary data.
    - A mapping of neighbor_id to edge_id.
    - Degree information:
        - For directed graphs: in_neighbors, in_degree, and out_degree.
        - For undirected graphs: degree.

    Time Complexity for vertex operations: O(1)
    Space Complexity per vertex: O(1)
    """

    def __init__(self, v_id: int, data=None, directed: bool = False):
        """
        Initialize a Vertex instance.

        Time Complexity: O(1)
        Space Complexity: O(1)

        Args:
            v_id (int): The unique identifier for the vertex.
            data (Any, optional): Optional data associated with the vertex.
            directed (bool): If True, initializes directed graph attributes (in_neighbors, in_degree, out_degree);
                             otherwise, initializes undirected graph attribute (degree).
        """
        self.id = v_id
        self.data = data
        self.neighbors = {}  # neighbor_id: edge_id

        if directed:
            self.in_neighbors = {}
            self.in_degree = 0
            self.out_degree = 0
        else:
            self.degree = 0

    def __repr__(self):
        """
        Return a string representation of the Vertex.

        Time Complexity: O(1)
        Space Complexity: O(1)

        Returns:
            str: A string showing the vertex ID and its degree information.
        """

        if hasattr(self, "in_degree"):
            return f"<Vertex {self.id}: in_degree={self.in_degree}, out_degree={self.out_degree}>"
        else:
            return f"<Vertex {self.id}: degree={self.degree}>"


class Edge:
    """
    A simple wrapper to represent an edge in a graph.

    Stores:
    - The edge's unique identifier.
    - The source vertex ID.
    - The target vertex ID.
    - Optional arbitrary data.
    - A weight (default is 1).

    Time Complexity for edge operations: O(1)
    Space Complexity per edge: O(1)
    """

    def __init__(self, e_id: int, source: int, target: int, data=None, weight: float = 1):
        """
        Initialize an Edge instance.

        Time Complexity: O(1)
        Space Complexity: O(1)

        Args:
            e_id (int): The unique identifier for the edge.
            source (int): The source vertex ID.
            target (int): The target vertex ID.
            data (Any, optional): Optional data associated with the edge.
            weight (float, optional): The weight of the edge (default is 1).
        """

        self.id = e_id
        self.source = source
        self.target = target
        self.data = data
        self.weight = weight

    def __repr__(self):
        """
        Return a string representation of the Edge.

        Time Complexity: O(1)
        Space Complexity: O(1)

        Returns:
            str: A string in the format "<Edge {id}: {source} -> {target}, weight={weight}>".
        """

        return f"<Edge {self.id}: {self.source} -> {self.target}, weight={self.weight}>"


class Graph:
    """
    A graph data structure supporting both directed and undirected graphs.

    Manages vertices and edges:
    - Vertices are stored in a dictionary mapping vertex IDs to Vertex objects.
    - Edges are stored in a dictionary mapping edge IDs to Edge objects.
    - Each vertex maintains its own adjacency map and degree information.

    Overall operations vary:
    - Most vertex and edge operations run in O(1) time.
    - Removing a vertex takes O(degree(v)) time in the worst case.
    - Specialized graph constructions (e.g., complete graph) may require higher time complexity.

    Space Complexity: O(V + E)
    Time Complexity: Varies by operation.
    """

    def __init__(self, n: int = 0, directed: bool = False):
        """
        Initialize a Graph instance.

        Creates an empty graph with optional initial vertices. If 'n' is provided,
        n vertices are added to the graph using the add_vertex method.

        Time Complexity: O(n) for adding n vertices.
        Space Complexity: O(n)

        Args:
            n (int, optional): The initial number of vertices to add (default is 0).
            directed (bool, optional): If True, creates a directed graph; otherwise, an undirected graph (default is False).
        """

        self.directed = directed

        self._next_vertex_id = 0
        self._next_edge_id = 0

        self.vertices = {}  # vertex_id: Vertex
        self.edges = {}  # edge_id: Edge

        self.n = 0  # number of vertices
        self.m = 0  # number of edges

        self.adj_matrix = defaultdict(lambda: defaultdict(float))  # dict of dicts

        for _ in range(n):
            self.add_vertex()

    def add_vertex(self, data=None):
        """
        Add a new vertex to the graph.

        Creates a new Vertex with a unique ID and optional associated data,
        stores it in the graph, and returns its ID.

        Time Complexity: O(1)
        Space Complexity: O(1)

        Args:
            data (Any, optional): Optional data to store with the vertex.

        Returns:
            int: The unique ID of the newly added vertex.
        """

        v_id = self._next_vertex_id
        self._next_vertex_id += 1

        vertex_obj = Vertex(v_id, data, directed=self.directed)
        self.vertices[v_id] = vertex_obj

        self.n += 1
        return v_id

    def add_edge(self, u_id: int, v_id: int, weight: float = 1, data=None):
        """
        Add an edge to the graph between vertices u and v with an optional weight and data.

        For directed graphs, the edge is added from u to v.
        For undirected graphs, the edge is added in both directions (u to v and v to u).

        Time Complexity: O(1) on average.
        Space Complexity: O(1)

        Args:
            u_id (int): The source vertex ID.
            v_id (int): The target vertex ID.
            weight (float, optional): The weight of the edge (default is 1).
            data (Any, optional): Optional data associated with the edge.

        Returns:
            int: The unique ID of the newly added edge.

        Raises:
            ValueError: If either the source or target vertex does not exist.
        """

        if u_id not in self.vertices or v_id not in self.vertices:
            raise ValueError("Source or target vertex does not exist.")

        e_id = self._next_edge_id
        self._next_edge_id += 1

        edge_obj = Edge(e_id, u_id, v_id, data, weight)
        self.edges[e_id] = edge_obj

        self.vertices[u_id].neighbors[v_id] = e_id
        self.adj_matrix[u_id][v_id] = weight

        if self.directed:
            self.vertices[v_id].in_neighbors[u_id] = e_id

            self.vertices[u_id].out_degree += 1
            self.vertices[v_id].in_degree += 1
        else:
            self.vertices[v_id].neighbors[u_id] = e_id
            self.adj_matrix[v_id][u_id] = weight

            self.vertices[u_id].degree += 1
            self.vertices[v_id].degree += 1

        self.m += 1
        return e_id

    def remove_edge(self, e_id: int):
        """
        Remove an edge from the graph given its edge ID.

        Updates the adjacency maps and degree information of the incident vertices.

        Time Complexity: O(1) on average.
        Space Complexity: O(1)

        Args:
            e_id (int): The unique ID of the edge to be removed.

        Raises:
            ValueError: If the edge does not exist.
        """

        if e_id not in self.edges:
            raise ValueError("Edge does not exist.")

        edge_obj = self.edges[e_id]
        u, v = edge_obj.source, edge_obj.target

        if v in self.vertices[u].neighbors and self.vertices[u].neighbors[v] == e_id:
            del self.vertices[u].neighbors[v]
        del self.adj_matrix[u][v]

        if self.directed:
            if u in self.vertices[v].in_neighbors and self.vertices[v].in_neighbors[u] == e_id:
                del self.vertices[v].in_neighbors[u]

            self.vertices[u].out_degree -= 1
            self.vertices[v].in_degree -= 1
        else:
            if u in self.vertices[v].neighbors and self.vertices[v].neighbors[u] == e_id:
                del self.vertices[v].neighbors[u]
            del self.adj_matrix[v][u]

            self.vertices[u].degree -= 1
            self.vertices[v].degree -= 1

        del self.edges[e_id]
        self.m -= 1

    def remove_vertex(self, v_id: int):
        """
        Remove a vertex from the graph given its vertex ID.

        Removes all incident edges (both outgoing and, for directed graphs, incoming) and then deletes the vertex.

        Time Complexity: O(degree(v)), where degree(v) is the number of incident edges.
        Space Complexity: O(1) auxiliary space.

        Args:
            v_id (int): The unique ID of the vertex to be removed.

        Raises:
            ValueError: If the vertex does not exist.
        """

        if v_id not in self.vertices:
            raise ValueError("Vertex does not exist.")

        v_obj = self.vertices[v_id]
        incident_edges = set()

        # Outgoing edges (present in neighbors)
        incident_edges.update(v_obj.neighbors.values())

        if self.directed:
            incident_edges.update(v_obj.in_neighbors.values())

        for e_id in incident_edges:
            self.remove_edge(e_id)

        del self.vertices[v_id]
        del self.adj_matrix[v_id]
        self.n -= 1

    def draw(self):
        """
        Visualizes the graph using `networkx` and `matplotlib`.

        Nodes represent vertices, and edges are drawn based on adjacency.
        If the graph is weighted, edge weights are displayed.

        Time Complexity: O(V + E)
        Space Complexity: O(V + E)
        """
        G = nx.DiGraph() if self.directed else nx.Graph()
        for v in self.vertices:
            G.add_node(v)
        for edge in self.edges.values():
            G.add_edge(edge.source, edge.target, weight=edge.weight)

        pos = nx.spring_layout(G)
        labels = nx.get_edge_attributes(G, "weight")
        nx.draw(G, pos, with_labels=True)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
        plt.show()

    def __repr__(self):
        """
        Return a string representation of the graph with basic information.

        Time Complexity: O(1)
        Space Complexity: O(1)

        Returns:
            str: A formatted string summarizing the graph type (directed/undirected) and the counts of vertices and edges.
        """

        graph_type = "Directed" if self.directed else "Undirected"
        return f"<{graph_type} Graph: |V|={self.n}, |E|={self.m}>"

    @staticmethod
    def line(n: int, directed: bool = False):
        """
        Create and return a line graph with n vertices.

        A line graph is a graph where vertices are connected sequentially in a single line.
        Edges are added between successive vertices.

        Time Complexity: O(n)
        Space Complexity: O(n)

        Args:
            n (int): The number of vertices in the graph.
            directed (bool, optional): If True, creates a directed line graph; otherwise, an undirected line graph (default is False).

        Returns:
            Graph: A Graph instance representing the line graph.
        """

        graph = Graph(n, directed)
        vertices = list(graph.vertices.keys())

        for i in range(n - 1):
            graph.add_edge(vertices[i], vertices[i + 1])

        return graph

    @staticmethod
    def cycle(n: int, directed: bool = False):
        """
        Create and return a cycle graph with n vertices.

        A cycle graph is a graph where vertices are connected in a closed loop.
        Edges are added between successive vertices, and an edge from the last to the first vertex completes the cycle.

        Time Complexity: O(n)
        Space Complexity: O(n)

        Args:
            n (int): The number of vertices in the graph.
            directed (bool, optional): If True, creates a directed cycle graph; otherwise, an undirected cycle graph (default is False).

        Returns:
            Graph: A Graph instance representing the cycle graph.
        """

        graph = Graph(n, directed)
        vertices = list(graph.vertices.keys())

        for i in range(n):
            graph.add_edge(vertices[i], vertices[(i + 1) % n])

        return graph

    @staticmethod
    def complete(n: int, directed: bool = False):
        """
        Create and return a complete graph with n vertices.

        In a complete graph, every pair of distinct vertices is connected by an edge.

        Time Complexity: O(n^2)
        Space Complexity: O(n^2)

        Args:
            n (int): The number of vertices in the graph.
            directed (bool, optional): If True, creates a directed complete graph; otherwise, an undirected complete graph (default is False).

        Returns:
            Graph: A Graph instance representing the complete graph.
        """

        graph = Graph(n, directed)
        vertices = list(graph.vertices.keys())

        for i in range(n):
            for j in range(n):
                if i != j:
                    graph.add_edge(vertices[i], vertices[j])

        return graph
