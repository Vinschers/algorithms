from collections.abc import Sequence
from .heap import Heap


class _PriorityQueueNode:
    def __init__(self, priority: int, value):
        """
        Initialize a priority queue node with the given priority and value.

        Time Complexity: O(1)
        Space Complexity: O(1)

        Args:
            priority: The priority of the node.
            value: The value associated with the node.
        """
        self.priority = priority
        self.value = value

    def __lt__(self, other):
        """
        Compare this node with another for less-than based on priority.

        Time Complexity: O(1)
        Space Complexity: O(1)

        Args:
            other (_PriorityQueueNode): The node to compare against.

        Returns:
            bool: True if this node's priority is less than the other's; otherwise, False.
        """

        return self.priority < other.priority

    def __gt__(self, other):
        """
        Compare this node with another for greater-than based on priority.

        Time Complexity: O(1)
        Space Complexity: O(1)

        Args:
            other (_PriorityQueueNode): The node to compare against.

        Returns:
            bool: True if this node's priority is greater than the other's; otherwise, False.
        """

        return self.priority > other.priority

    def __repr__(self):
        """
        Return the string representation of the node.

        Time Complexity: O(1)
        Space Complexity: O(1)

        Returns:
            str: The string representation in the format "(priority, value)".
        """

        return f"({self.priority}, {self.value})"


class PriorityQueue:
    """
    A priority queue built atop a Heap, allowing duplicates.

    For duplicates, each 'value' can appear multiple times with different priorities.
    We track node objects in a dictionary: value -> set of _PriorityQueueNode.
    Operations remain O(log n) on average, leveraging the heap's index map internally.
    """

    def __init__(self, values=None, priorities=None, max=True):
        """
        Initialize the PriorityQueue.

        This constructor supports multiple initialization modes:
        - If both 'values' and 'priorities' are None, the queue starts empty.
        - If 'values' is a sequence and 'priorities' is None, 'values' is treated as a list of (value, priority) pairs.
        - If both 'values' and 'priorities' are sequences, they are paired together.
        - The 'max' parameter determines whether the queue is a max-priority queue (default True) or a min-priority queue.

        Time Complexity: O(n) for heap construction.
        Space Complexity: O(n)

        Args:
            values (optional): A sequence of (value, priority) pairs or a sequence of values.
            priorities (optional): A sequence of priorities corresponding to the values.
            max (bool): If True, creates a max-priority queue; if False, creates a min-priority queue.

        Raises:
            ValueError: If 'values' and 'priorities' are provided but their lengths differ, or if the input format is invalid.
        """
        self.max = max
        self.node_map = {}

        if values is None and priorities is None:
            nodes = []
        elif isinstance(values, Sequence) and priorities is None:
            nodes = [_PriorityQueueNode(p, v) for (v, p) in values]
        elif isinstance(values, Sequence) and isinstance(priorities, Sequence):
            if len(values) != len(priorities):
                raise ValueError("Values and priorities must have the same length.")
            nodes = [_PriorityQueueNode(p, v) for (v, p) in zip(values, priorities)]
        else:
            raise ValueError("Provide either (values, priorities) or a list of (value, priority) pairs.")

        self.heap = Heap(nodes, max)

        for node in nodes:
            if node.value not in self.node_map:
                self.node_map[node.value] = {node}
            else:
                self.node_map[node.value].add(node)

    def __len__(self):
        """
        Return the number of elements in the priority queue.

        Time Complexity: O(1)
        Space Complexity: O(1)

        Returns:
            int: The number of elements in the queue.
        """

        return len(self.heap)

    def __contains__(self, item):
        return item in self.node_map

    def is_empty(self):
        """
        Determine whether the priority queue is empty.

        Time Complexity: O(1)
        Space Complexity: O(1)

        Returns:
            bool: True if the queue is empty, otherwise False.
        """

        return len(self) == 0

    def peek(self):
        """
        Retrieve the (priority, value) tuple of the top node without removing it.

        Time Complexity: O(1)
        Space Complexity: O(1)

        Returns:
            tuple: A tuple (priority, value) of the top node in the queue.

        Raises:
            KeyError: If the priority queue is empty.
        """

        if self.is_empty():
            raise KeyError("Priority queue is empty")

        node = self.heap.peek()
        return node.priority, node.value

    def enqueue(self, value, priority):
        """
        Insert a new (priority, value) item into the priority queue.

        A new _PriorityQueueNode is created and inserted into the underlying heap.
        The node is also recorded in the node_map to support duplicate management.

        Time Complexity: O(log n) on average.
        Space Complexity: O(1) auxiliary space.

        Args:
            value: The value to be enqueued.
            priority: The priority associated with the value.
        """
        node = _PriorityQueueNode(priority, value)
        self.heap.insert(node)  # O(log n)

        if value not in self.node_map:
            self.node_map[value] = {node}
        else:
            self.node_map[value].add(node)

    def dequeue(self):
        """
        Remove and return the (priority, value) tuple of the top node from the queue.

        The top node is extracted from the heap, and the corresponding entry in the node_map is updated.
        This operation rebalances the heap to maintain the heap property.

        Time Complexity: O(log n) on average.
        Space Complexity: O(1) auxiliary space.

        Returns:
            tuple: A tuple (priority, value) representing the removed node.

        Raises:
            KeyError: If the priority queue is empty.
        """

        if self.is_empty():
            raise KeyError("Priority queue is empty")

        node = self.heap.extract()  # O(log n)

        node_set = self.node_map[node.value]
        node_set.remove(node)
        if not node_set:
            del self.node_map[node.value]

        return node.priority, node.value

    def update(self, value, new_priority, all=False):
        """
        Update the priority of one or all occurrences of a given value in the priority queue.

        If 'all' is False (default), only one occurrence is updated.
        If 'all' is True, all occurrences of the specified value are updated.
        For each update, if the current priority differs from new_priority, the node is replaced in the heap.

        Time Complexity:
            - O(log n) on average if updating one occurrence.
            - O(k log n) on average if updating k occurrences.

        Space Complexity: O(1) auxiliary space.

        Args:
            value: The value whose priority is to be updated.
            new_priority: The new priority to assign to the value.
            all (bool): If True, update all occurrences of the value; otherwise, update only one occurrence.

        Raises:
            KeyError: If the specified value is not found in the priority queue.
        """
        if value not in self.node_map or not self.node_map[value]:
            raise KeyError(f"Value {value} not found in priority queue")

        nodes_to_update = list(self.node_map[value]) if all else [next(iter(self.node_map[value]))]

        for old_node in nodes_to_update:
            if old_node.priority != new_priority:
                new_node = _PriorityQueueNode(new_priority, value)
                self.heap.replace(old_node, new_node)  # O(log n)

                self.node_map[value].remove(old_node)
                self.node_map[value].add(new_node)
