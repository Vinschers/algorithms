class _UFNode:
    """
    Internal node for UnionFind (disjoint-set) structure.

    Each node stores:
    - its parent pointer (initially itself),
    - the size of the tree for which it is a root,
    - and the actual element value.
    """

    def __init__(self, x):
        """
        Initialize a new set node for element x.

        Args:
            x: The element represented by this node.

        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        self.parent = self
        self.size = 1
        self.value = x


class UnionFind:
    """
    Disjoint-set (Union-Find) data structure with **path splitting** and union by size.

    This structure supports:
    - make_set(x): create singleton sets,
    - find(x): find the set representative/root,
    - union(x, y): merge two sets,
    - connected(x, y): test whether two elements are in the same set,
    - `x in uf`: test whether an element exists in the structure,
    - len(uf): total number of elements tracked.

    **Path Splitting**:
        This implementation uses *path splitting*: during traversal, each visited
        node is repointed to its grandparent. This flattens trees similarly to
        path compression but is done iteratively within the find loop.
    """

    def __init__(self) -> None:
        """
        Initialize an empty UnionFind.

        No elements exist initially; use `make_set` to add them.

        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        self.sets = {}

    def make_set(self, x):
        """
        Create a new set containing the element x.

        Args:
            x: The element to create a set for.

        Raises:
            ValueError: If x is already in a set.

        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        if x in self.sets:
            raise ValueError(f"Element {x} already exists")
        self.sets[x] = _UFNode(x)

    def _find_node(self, x):
        """
        Internal helper to return the root _UFNode object for element x.

        Uses **path splitting** to compress paths:
        each visited node points to its grandparent during traversal,
        effectively flattening the tree and improving future queries.

        Args:
            x: The element whose root node to find.

        Returns:
            _UFNode: The root node object of x’s set.

        Raises:
            KeyError: If x is not in any set.

        Time Complexity: amortized O(α(n))
        Space Complexity: O(1)
        """
        if x not in self.sets:
            raise KeyError(f"{x} not found")

        node = self.sets[x]
        while node.parent != node:
            parent = node.parent
            node.parent = node.parent.parent  # path splitting
            node = parent
        return node

    def find(self, x):
        """
        Find the representative (root) element for the set containing x.

        Internally calls `_find_node` to locate the root and then returns its value.

        Args:
            x: The element to find.

        Returns:
            The representative element (value) of x’s set.

        Raises:
            KeyError: If x is not in any set.

        Time Complexity: amortized O(α(n))
        Space Complexity: O(1)
        """
        return self._find_node(x).value

    def union(self, x, y):
        """
        Merge the sets containing x and y.

        Uses the union-by-size heuristic to attach the smaller tree under
        the larger one. If x and y are already in the same set, does nothing.

        Args:
            x: First element.
            y: Second element.

        Raises:
            KeyError: If either x or y is not in any set.

        Time Complexity: amortized O(α(n))
        Space Complexity: O(1)
        """
        if x not in self.sets or y not in self.sets:
            raise KeyError("One or both elements are not found")

        root_x = self._find_node(x)
        root_y = self._find_node(y)

        if root_x == root_y:
            return

        if root_x.size < root_y.size:
            root_x, root_y = root_y, root_x

        root_y.parent = root_x
        root_x.size += root_y.size

    def connected(self, x, y):
        """
        Check if x and y are in the same set.

        Compares the root nodes returned by `_find_node`.

        Args:
            x: First element.
            y: Second element.

        Returns:
            bool: True if x and y are in same set, False otherwise.

        Raises:
            KeyError: If either x or y is not in any set.

        Time Complexity: amortized O(α(n))
        Space Complexity: O(1)
        """
        return self._find_node(x) is self._find_node(y)

    def __contains__(self, x):
        """
        Check whether an element x exists in this UnionFind.

        Enables the use of the `in` operator, e.g.:

            if x in uf:
                ...

        Args:
            x: Element to check.

        Returns:
            bool: True if x is tracked by this UnionFind, False otherwise.

        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        return x in self.sets

    def __len__(self):
        """
        Return the total number of elements currently tracked by the UnionFind.

        This is simply the count of all elements added via `make_set`.

        Returns:
            int: The total number of elements.

        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        return len(self.sets)
