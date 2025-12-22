from typing import Optional


class _TreeNode:
    """
    A generic tree node used as the base for different tree implementations.

    Each node stores:
    - A value (the key or data stored in the node).
    - References to parent, left child, and right child nodes.
    """

    def __init__(self, value):
        """
        Initialize a generic tree node.

        Time Complexity: O(1)
        Space Complexity: O(1)

        Args:
            value: The data or key to store in the node.
        """
        self.value = value

        self.parent: Optional[_RedBlackTreeNode] = None
        self.left: Optional[_RedBlackTreeNode] = None
        self.right: Optional[_RedBlackTreeNode] = None


class _BST:
    """
    A simple (unbalanced) Binary Search Tree (BST) implementation.

    This implementation supports:
    - Inserting new values.
    - Deleting values.
    - Maintaining the BST ordering property.
    """

    def __init__(self):
        """
        Initialize an empty Binary Search Tree.

        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        self.root = None

    def insert(self, value):
        """
        Insert a value into the BST.

        If the tree is empty, the value becomes the root node.
        Otherwise, it recursively finds the correct position maintaining the BST property.

        Time Complexity: O(h), where h = tree height.
        Space Complexity: O(h) for recursive stack.

        Args:
            value: The value to insert.
        """
        if self.root is None:
            self.root = _TreeNode(value)
        else:
            self._insert_node(self.root, value)

    def _insert_node(self, node, value):
        """
        Helper method to insert a value starting from a given node.

        Time Complexity: O(h)
        Space Complexity: O(h) for recursion.
        """
        if value < node.value:
            if node.left is None:
                node.left = _TreeNode(value)
                node.left.parent = node
            else:
                self._insert_node(node.left, value)
        elif value > node.value:
            if node.right is None:
                node.right = _TreeNode(value)
                node.right.parent = node
            else:
                self._insert_node(node.right, value)

    def delete(self, value):
        """
        Delete a value from the BST.

        If the value is not found, no action is taken.

        Time Complexity: O(h), where h = tree height.
        Space Complexity: O(h) for recursive stack.

        Args:
            value: The value to remove from the tree.
        """
        self.root = self._delete_node(self.root, value)

    def _delete_node(self, node, value):
        """
        Helper method to recursively delete a value from the tree.

        Time Complexity: O(h)
        Space Complexity: O(h) for recursion.
        """
        if node is None:
            return node

        if value < node.value:
            node.left = self._delete_node(node.left, value)
        elif value > node.value:
            node.right = self._delete_node(node.right, value)
        else:
            if node.left is None:
                temp = node.right
                if temp:
                    temp.parent = node.parent
                return temp
            elif node.right is None:
                temp = node.left
                if temp:
                    temp.parent = node.parent
                return temp

            successor = self._min_value_node(node.right)
            node.value = successor.value
            node.right = self._delete_node(node.right, successor.value)
        return node

    def _min_value_node(self, node):
        """
        Return the node with the minimum value in the subtree rooted at the given node.

        Time Complexity: O(h)
        Space Complexity: O(1)

        Args:
            node: The subtree root to search from.

        Returns:
            _TreeNode: The node with the smallest value.
        """
        current = node
        while current.left:
            current = current.left
        return current


class _AVLTreeNode(_TreeNode):
    """
    A node in an AVL (self-balancing) tree.

    In addition to value and child pointers, it stores:
    - The height of the node for balancing calculations.
    """

    def __init__(self, value):
        """
        Initialize an AVL tree node with height 1.

        Time Complexity: O(1)
        Space Complexity: O(1)

        Args:
            value: The value to store in the node.
        """
        super().__init__(value)
        self.height = 1


class _AVLTree:
    """
    An AVL Tree: a self-balancing Binary Search Tree.

    This implementation supports:
    - Inserting values with automatic balancing.
    - Deleting values with automatic balancing.
    - Maintaining node heights and rotations.
    """

    def __init__(self):
        """
        Initialize an empty AVL Tree.

        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        self.root = None

    def insert(self, value):
        """
        Insert a value into the AVL Tree with automatic rebalancing.

        Time Complexity: O(log n)
        Space Complexity: O(log n) for recursion.

        Args:
            value: The value to insert.
        """
        self.root = self._insert_node_avl(self.root, value)

    def _insert_node_avl(self, node, value):
        """
        Helper to recursively insert a value while maintaining AVL balance.

        Time Complexity: O(log n)
        Space Complexity: O(log n)
        """
        if node is None:
            return _AVLTreeNode(value)

        if value < node.value:
            node.left = self._insert_node_avl(node.left, value)
            node.left.parent = node
        elif value > node.value:
            node.right = self._insert_node_avl(node.right, value)
            node.right.parent = node
        else:
            return node

        node.height = 1 + max(self._height(node.left), self._height(node.right))
        return self._rebalance(node)

    def delete(self, value):
        """
        Delete a value from the AVL Tree, rebalancing as needed.

        Time Complexity: O(log n)
        Space Complexity: O(log n)

        Args:
            value: The value to delete.
        """
        self.root = self._delete_node_avl(self.root, value)

    def _delete_node_avl(self, node, value):
        """
        Helper to delete a value from the AVL Tree and rebalance.

        Time Complexity: O(log n)
        Space Complexity: O(log n)
        """
        if node is None:
            return node

        if value < node.value:
            node.left = self._delete_node_avl(node.left, value)
        elif value > node.value:
            node.right = self._delete_node_avl(node.right, value)
        else:
            if node.left is None:
                temp = node.right
                if temp:
                    temp.parent = node.parent
                return temp
            elif node.right is None:
                temp = node.left
                if temp:
                    temp.parent = node.parent
                return temp

            succ = self._min_value_node(node.right)
            node.value = succ.value
            node.right = self._delete_node_avl(node.right, succ.value)

        node.height = 1 + max(self._height(node.left), self._height(node.right))
        return self._rebalance(node)

    def _height(self, node):
        """
        Return the height of a node.

        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        return node.height if node else 0

    def _balance_factor(self, node):
        """
        Compute the balance factor of a node.

        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        return self._height(node.left) - self._height(node.right)

    def _rebalance(self, node):
        """
        Rebalance the subtree rooted at the given node if it is unbalanced.

        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        balance = self._balance_factor(node)
        if balance > 1:
            if self._balance_factor(node.left) < 0:
                node.left = self._rotate_left(node.left)
            return self._rotate_right(node)
        if balance < -1:
            if self._balance_factor(node.right) > 0:
                node.right = self._rotate_right(node.right)
            return self._rotate_left(node)
        return node

    def _rotate_left(self, z):
        """
        Perform a left rotation around the given node.

        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        y = z.right
        T2 = y.left

        y.left = z
        z.right = T2

        y.parent = z.parent
        z.parent = y
        if T2:
            T2.parent = z

        z.height = 1 + max(self._height(z.left), self._height(z.right))
        y.height = 1 + max(self._height(y.left), self._height(y.right))
        return y

    def _rotate_right(self, z):
        """
        Perform a right rotation around the given node.

        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        y = z.left
        T3 = y.right

        y.right = z
        z.left = T3

        y.parent = z.parent
        z.parent = y
        if T3:
            T3.parent = z

        z.height = 1 + max(self._height(z.left), self._height(z.right))
        y.height = 1 + max(self._height(y.left), self._height(y.right))
        return y

    def _min_value_node(self, node):
        """
        Return the node with the minimum value in the subtree rooted at the given node.

        Time Complexity: O(log n)
        Space Complexity: O(1)
        """
        current = node
        while current.left:
            current = current.left
        return current


class _RedBlackTreeNode(_TreeNode):
    """
    A node within a Red-Black Tree.

    In addition to standard BST attributes, it stores:
    - Color: 'RED' or 'BLACK'.
    """

    def __init__(self, value, color="RED"):
        """
        Initialize a Red-Black Tree node.

        Time Complexity: O(1)
        Space Complexity: O(1)

        Args:
            value: The value to store.
            color (str): The initial color of the node ('RED' or 'BLACK').
        """
        super().__init__(value)
        self.color = color


class _RedBlackTree:
    """
    A Red-Black Tree — a self-balancing Binary Search Tree.

    This implementation supports:
    - Inserting and deleting values with balancing via rotations and recoloring.
    - Maintaining the Red-Black Tree properties.
    """

    def __init__(self):
        """
        Initialize an empty Red-Black Tree.

        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        self.root = None

    def insert(self, value):
        """
        Insert a value into the Red-Black Tree.

        Uses standard BST insertion followed by fix-up to restore Red-Black properties.

        Time Complexity: O(log n)
        Space Complexity: O(log n)

        Args:
            value: The value to insert.
        """
        z = _RedBlackTreeNode(value, "RED")
        y = None
        x = self.root
        while x:
            y = x
            if z.value < x.value:
                x = x.left
            elif z.value > x.value:
                x = x.right
            else:
                return  # ignore duplicates

        z.parent = y
        if y is None:
            self.root = z
        elif z.value < y.value:
            y.left = z
        else:
            y.right = z

        self._fix_insert(z)

    def delete(self, value):
        """
        Delete a value from the Red-Black Tree.

        Uses standard BST deletion followed by fix-up to restore Red-Black properties.

        Time Complexity: O(log n)
        Space Complexity: O(log n)

        Args:
            value: The value to remove.
        """
        z = self._search_node(self.root, value)
        if z is None:
            return

        y = z
        y_original_color = y.color
        if z.left is None:
            x = z.right
            parent = z.parent
            self._transplant(z, z.right)
        elif z.right is None:
            x = z.left
            parent = z.parent
            self._transplant(z, z.left)
        else:
            y = self._min_value_node(z.right)
            y_original_color = y.color
            x = y.right
            parent = y.parent if y.parent != z else y
            if y.parent != z:
                self._transplant(y, y.right)
                y.right = z.right
                if y.right:
                    y.right.parent = y
            self._transplant(z, y)
            y.left = z.left
            if y.left:
                y.left.parent = y
            y.color = z.color

        if y_original_color == "BLACK":
            self._fix_delete(x, parent)

        if self.root:
            self.root.color = "BLACK"

    # ------- Helpers --------
    def _search_node(self, node, value):
        """
        Search for a node with the given value in the subtree rooted at `node`.

        Time Complexity: O(log n)
        Space Complexity: O(1)

        Args:
            node: The root of the subtree to search.
            value: The value to look for.

        Returns:
            _RedBlackTreeNode or None: The matching node, or None if not found.
        """
        while node:
            if value == node.value:
                return node
            node = node.left if value < node.value else node.right
        return None

    def _is_red(self, node):
        """
        Return True if the node is RED.

        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        return node is not None and node.color == "RED"

    def _is_black(self, node):
        """
        Return True if the node is BLACK (or None, treated as black).

        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        return node is None or node.color == "BLACK"

    def _left_rotate(self, x):
        """
        Perform a left rotation around node x.

        Time Complexity: O(1)
        Space Complexity: O(1)

        Args:
            x: The pivot node to rotate around.
        """
        y = x.right
        if y is None:
            return
        x.right = y.left
        if y.left:
            y.left.parent = x
        y.parent = x.parent
        if x.parent is None:
            self.root = y
        elif x == x.parent.left:
            x.parent.left = y
        else:
            x.parent.right = y
        y.left = x
        x.parent = y

    def _right_rotate(self, x):
        """
        Perform a right rotation around node x.

        Time Complexity: O(1)
        Space Complexity: O(1)

        Args:
            x: The pivot node to rotate around.
        """
        y = x.left
        if y is None:
            return
        x.left = y.right
        if y.right:
            y.right.parent = x
        y.parent = x.parent
        if x.parent is None:
            self.root = y
        elif x == x.parent.right:
            x.parent.right = y
        else:
            x.parent.left = y
        y.right = x
        x.parent = y

    def _transplant(self, u, v):
        """
        Replace the subtree rooted at node u with the subtree rooted at node v.

        This is a standard helper for BST deletion.

        Time Complexity: O(1)
        Space Complexity: O(1)

        Args:
            u: The node to be replaced.
            v: The new subtree root.
        """
        if u.parent is None:
            self.root = v
        elif u == u.parent.left:
            u.parent.left = v
        else:
            u.parent.right = v
        if v:
            v.parent = u.parent

    def _fix_insert(self, z):
        """
        Restore Red-Black Tree properties after inserting a red node.

        Uses recoloring and rotations to maintain the five Red-Black properties.

        Time Complexity: O(log n)
        Space Complexity: O(1)

        Args:
            z: The newly inserted node.
        """
        while z.parent and z.parent.color == "RED":
            if z.parent == z.parent.parent.left:
                y = z.parent.parent.right  # uncle
                if self._is_red(y):
                    z.parent.color = "BLACK"
                    y.color = "BLACK"
                    z.parent.parent.color = "RED"
                    z = z.parent.parent
                else:
                    if z == z.parent.right:
                        z = z.parent
                        self._left_rotate(z)
                    z.parent.color = "BLACK"
                    z.parent.parent.color = "RED"
                    self._right_rotate(z.parent.parent)
            else:
                y = z.parent.parent.left  # uncle
                if self._is_red(y):
                    z.parent.color = "BLACK"
                    y.color = "BLACK"
                    z.parent.parent.color = "RED"
                    z = z.parent.parent
                else:
                    if z == z.parent.left:
                        z = z.parent
                        self._right_rotate(z)
                    z.parent.color = "BLACK"
                    z.parent.parent.color = "RED"
                    self._left_rotate(z.parent.parent)

        if self.root:
            self.root.color = "BLACK"

    def _fix_delete(self, x, parent):
        """
        Restore Red-Black Tree properties after deleting a node.

        Uses rotations and recoloring to handle 'double-black' situations.

        Time Complexity: O(log n)
        Space Complexity: O(1)

        Args:
            x: The child node replacing the deleted node (may be None).
            parent: The parent of x.
        """
        while (x != self.root) and self._is_black(x):
            if x == (parent.left if parent else None):
                w = parent.right if parent else None
                if self._is_red(w) and w is not None:
                    w.color = "BLACK"
                    if parent:
                        parent.color = "RED"
                    self._left_rotate(parent)
                    if parent:
                        w = parent.right
                if self._is_black(w.left if w else None) and self._is_black(
                    w.right if w else None
                ):
                    if w:
                        w.color = "RED"
                    x = parent
                    parent = x.parent if x else None
                else:
                    if self._is_black(w.right if w else None):
                        if w and w.left:
                            w.left.color = "BLACK"
                        if w:
                            w.color = "RED"
                            self._right_rotate(w)
                        if parent:
                            w = parent.right
                    if w and parent:
                        w.color = parent.color
                    if parent:
                        parent.color = "BLACK"
                    if w and w.right:
                        w.right.color = "BLACK"
                    self._left_rotate(parent)
                    x = self.root
                    parent = None
            else:
                w = parent.left if parent else None
                if self._is_red(w) and w and parent:
                    w.color = "BLACK"
                    parent.color = "RED"
                    self._right_rotate(parent)
                    w = parent.left
                if self._is_black(w.left if w else None) and self._is_black(
                    w.right if w else None
                ):
                    if w:
                        w.color = "RED"
                    x = parent
                    parent = x.parent if x else None
                else:
                    if self._is_black(w.left if w else None):
                        if w and w.right:
                            w.right.color = "BLACK"
                        if w:
                            w.color = "RED"
                            self._left_rotate(w)
                        if parent:
                            w = parent.left
                    if w and parent:
                        w.color = parent.color
                    if parent:
                        parent.color = "BLACK"
                    if w and w.left:
                        w.left.color = "BLACK"
                    self._right_rotate(parent)
                    x = self.root
                    parent = None
        if x:
            x.color = "BLACK"

    def _min_value_node(self, node):
        """
        Return the node with the minimum value in the subtree rooted at the given node.

        Time Complexity: O(log n)
        Space Complexity: O(1)

        Args:
            node: The subtree root to search from.

        Returns:
            _RedBlackTreeNode: The node with the smallest value.
        """
        current = node
        while current.left:
            current = current.left
        return current


class Tree:
    """
    A unified interface for different types of binary trees.

    This class acts as a wrapper around:
    - _BST (regular binary search tree).
    - _AVLTree (self-balancing AVL tree).
    - _RedBlackTree (self-balancing Red-Black tree).

    The desired tree type is selected via the `balanced` parameter.
    """

    def __init__(self, balanced: str = ""):
        """
        Initialize a Tree wrapper with the specified balancing technique.

        Time Complexity: O(1)
        Space Complexity: O(1)

        Args:
            balanced (str): Type of tree to create.
                - "" (empty): regular BST.
                - "AVL": AVL Tree.
                - "RED-BLACK": Red-Black Tree.

        Raises:
            TypeError: If an unknown balancing technique is provided.
        """
        balanced = balanced.strip().upper()

        if balanced == "":
            self.tree = _BST()
        elif balanced == "AVL":
            self.tree = _AVLTree()
        elif balanced == "RED-BLACK":
            self.tree = _RedBlackTree()
        else:
            raise TypeError("Unknown balancing technique")

    def insert(self, value):
        """
        Insert a value into the underlying tree.

        Time Complexity:
            - BST: O(h)
            - AVL / Red-Black: O(log n)

        Args:
            value: The value to insert.
        """
        self.tree.insert(value)

    def delete(self, value):
        """
        Delete a value from the underlying tree.

        Time Complexity:
            - BST: O(h)
            - AVL / Red-Black: O(log n)

        Args:
            value: The value to delete.
        """
        self.tree.delete(value)

    def search(self, value):
        """
        Search for a value in the underlying tree.

        Time Complexity:
            - BST: O(h)
            - AVL / Red-Black: O(log n)

        Args:
            value: The value to search for.

        Returns:
            _TreeNode or None: The node containing the value, or None if not found.
        """
        node = self.tree.root

        while node:
            if value == node.value:
                return node
            elif value < node.value:
                node = node.left
            else:
                node = node.right
        return None

    @property
    def root(self):
        """
        Get the root node of the underlying tree.

        Returns:
            _TreeNode or None: The root node.
        """
        return self.tree.root
