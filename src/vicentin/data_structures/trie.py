from collections import defaultdict
from .stack import Stack


class _TrieNode:
    """
    A node within the Trie data structure.

    Each node stores:
    - An optional value (associated with the full word ending at this node).
    - A mapping of child nodes keyed by characters.
    - A terminal flag to indicate if the node marks the end of a valid word.
    """

    def __init__(self, value=None):
        """
        Initialize a Trie node.

        Time Complexity: O(1)
        Space Complexity: O(1)

        Args:
            value (optional): The value associated with the node if it represents a terminal word.
        """
        self.value = value
        self.children = defaultdict(_TrieNode)
        self.terminal = False


class Trie:
    """
    A Trie (prefix tree) for storing and retrieving strings with optional values.

    This implementation supports:
    - Adding words with optional values.
    - Deleting words (with automatic pruning of unused nodes).
    - Representing the contents of the trie in a human-readable form.
    """

    def __init__(self) -> None:
        """
        Initialize an empty Trie.

        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        self.root = _TrieNode()

    def add(self, word: str, value=None):
        """
        Insert a word into the trie with an optional value.

        If the word already exists, its value is updated. Intermediate nodes are created
        automatically for missing characters.

        Time Complexity: O(m), where m = len(word)
        Space Complexity: O(m) for potentially new nodes.

        Args:
            word (str): The word to insert.
            value (optional): The value to associate with the word.
        """
        node = self.root

        for char in word:
            node = node.children[char]

        node.value = value
        node.terminal = True

    def delete(self, word: str):
        """
        Remove a word from the trie, pruning unused nodes bottom-up.

        This method:
        - Traverses the trie to find the word.
        - Unmarks the terminal node.
        - Removes any non-terminal nodes with no children along the path.

        Time Complexity: O(m), where m = len(word)
        Space Complexity: O(m) auxiliary space for the stack.

        Args:
            word (str): The word to remove from the trie.

        Returns:
            None. (Does nothing if the word does not exist.)
        """
        node = self.root
        stack = Stack()

        for char in word:
            if char not in node.children:
                return  # Not found
            stack.push((node, char))
            node = node.children[char]

        if not node.terminal:
            return  # Not found

        node.terminal = False
        node.value = None

        for parent, char in stack:
            child = parent.children[char]

            if not child.terminal and len(child.children) == 0:
                del parent.children[char]
            else:
                break

    def __repr__(self) -> str:
        """
        Return a string representation of the Trie showing all stored words and their values.

        Time Complexity: O(n), where n = total number of characters stored in the trie.
        Space Complexity: O(n) auxiliary space for building the string list.

        Returns:
            str: A string representation of the trie contents.
        """

        def print_rec(node: _TrieNode, strings: list[str], text: str):
            """
            Helper to recursively traverse the trie and collect word-value pairs.

            Time Complexity: O(n) for all nodes.
            Space Complexity: O(depth) call stack.
            """
            if node.terminal:
                strings.append(f"{text}: {node.value}")

            for char, child_node in node.children.items():
                print_rec(child_node, strings, text + char)

        strings = []
        print_rec(self.root, strings, "")

        return f"Trie([{', '.join(map(str, strings))}])"
