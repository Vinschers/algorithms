from collections import deque


class Queue:
    """
    A FIFO (First-In-First-Out) queue built atop a deque.

    This queue provides efficient O(1) enqueue and dequeue operations at opposite ends.
    It supports peeking, iteration from front to back, and O(1) integer indexing.
    """

    def __init__(self) -> None:
        """
        Initialize an empty queue.

        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        self._queue = deque()

    def enqueue(self, item):
        """
        Add an item to the back of the queue.

        Time Complexity: O(1)
        Space Complexity: O(1) auxiliary space.

        Args:
            item: The item to add to the queue.
        """
        self._queue.append(item)

    def dequeue(self):
        """
        Remove and return the item from the front of the queue.

        Time Complexity: O(1)
        Space Complexity: O(1)

        Returns:
            The item removed from the front of the queue.

        Raises:
            IndexError: If the queue is empty.
        """
        return self._queue.popleft()

    def peek(self):
        """
        Retrieve the item at the front of the queue without removing it.

        Time Complexity: O(1)
        Space Complexity: O(1)

        Returns:
            The item at the front of the queue.

        Raises:
            IndexError: If the queue is empty.
        """
        if not self._queue:
            raise IndexError("Queue is empty")
        return self._queue[0]

    def clear(self):
        """
        Remove all items from the queue.

        Time Complexity: O(n) where n = len(queue)
        Space Complexity: O(1) auxiliary space.
        """
        self._queue.clear()

    def __len__(self):
        """
        Return the number of items in the queue.

        Time Complexity: O(1)
        Space Complexity: O(1)

        Returns:
            int: The number of items in the queue.
        """
        return len(self._queue)

    def __bool__(self):
        """
        Determine whether the queue is non-empty.

        Time Complexity: O(1)
        Space Complexity: O(1)

        Returns:
            bool: True if the queue has at least one item, otherwise False.
        """
        return bool(self._queue)

    def __contains__(self, item):
        """
        Determine whether an item exists in the queue.

        Time Complexity: O(n) where n = len(queue)
        Space Complexity: O(1)

        Args:
            item: The item to search for.

        Returns:
            bool: True if the item exists in the queue, otherwise False.
        """
        return item in self._queue

    def __iter__(self):
        """
        Iterate through the queue from front to back.

        Time Complexity: O(n)
        Space Complexity: O(1)

        Yields:
            Each item in the queue starting from the front.
        """
        return iter(self._queue)

    def __getitem__(self, index):
        """
        Retrieve an item or a slice from the queue.

        When given an integer index, returns the item at that position
        starting from the front (0 = front, 1 = next, etc.).
        When given a slice, returns a list of items in front-to-back order.

        Time Complexity:
            - O(1) for integer index.
            - O(k) for a slice of length k.

        Space Complexity:
            - O(1) for integer index.
            - O(k) for a slice of length k.

        Args:
            index (int or slice): The index or slice of items to retrieve.

        Returns:
            object or list: The item at the given index or a list of items for the slice.

        Raises:
            IndexError: If the integer index is out of range.
            TypeError: If the index is neither int nor slice.
        """
        n = len(self._queue)

        if isinstance(index, int):
            # Handle negative indices
            if index < 0:
                index = n + index
            if index < 0 or index >= n:
                raise IndexError("Queue index out of range")
            return self._queue[index]

        elif isinstance(index, slice):
            # Convert deque to list once and slice it
            return list(self._queue)[index]  # O(k)

        else:
            raise TypeError("Invalid argument type.")

    @property
    def front(self):
        """
        The front item of the queue as a property (read-only).

        Time Complexity: O(1)
        Space Complexity: O(1)

        Returns:
            The item at the front of the queue.

        Raises:
            IndexError: If the queue is empty.
        """
        return self.peek()

    def __repr__(self):
        """
        Return the string representation of the queue from front to back.

        Time Complexity: O(n)
        Space Complexity: O(1)

        Returns:
            str: A string representation of the queue.
        """
        return f"Queue([{', '.join(map(repr, self._queue))}])"
