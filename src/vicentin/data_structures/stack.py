from collections import deque


class Stack:
    """
    A LIFO (Last-In-First-Out) stack data structure built atop a deque.

    This stack provides efficient O(1) push and pop operations at the top.
    It supports direct O(1) integer indexing from top to bottom and slicing,
    which returns a list of elements in top-to-bottom order.
    """

    def __init__(self):
        """
        Initialize an empty stack.

        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        self._stack = deque()

    def push(self, item):
        """
        Push an item onto the top of the stack.

        Time Complexity: O(1)
        Space Complexity: O(1) auxiliary space.

        Args:
            item: The item to push onto the stack.
        """
        self._stack.append(item)

    def pop(self):
        """
        Pop and return the item at the top of the stack.

        Time Complexity: O(1)
        Space Complexity: O(1)

        Returns:
            The item removed from the top of the stack.

        Raises:
            IndexError: If the stack is empty.
        """
        return self._stack.pop()

    def clear(self):
        """
        Remove all items from the stack.

        Time Complexity: O(n) where n = len(stack)
        Space Complexity: O(1) auxiliary space.
        """
        self._stack.clear()

    def __contains__(self, item):
        """
        Determine whether an item exists in the stack.

        Time Complexity: O(n) where n = len(stack)
        Space Complexity: O(1)

        Args:
            item: The item to search for.

        Returns:
            bool: True if the item exists in the stack, otherwise False.
        """
        return item in self._stack

    def __iter__(self):
        """
        Iterate through the stack from top to bottom.

        Time Complexity: O(n)
        Space Complexity: O(1)

        Yields:
            Each item in the stack starting from the top.
        """
        return reversed(self._stack)

    def __getitem__(self, index):
        """
        Retrieve an item or a slice from the stack.

        When given an integer index, returns the item at that position
        starting from the top (0 = top, 1 = one below top, etc.).
        When given a slice, returns a list of items in top-to-bottom order.

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
        n = len(self._stack)

        if isinstance(index, int):
            if index < 0:
                index = n + index
            real_index = n - 1 - index
            if real_index < 0 or real_index >= n:
                raise IndexError("Stack index out of range")
            return self._stack[real_index]

        elif isinstance(index, slice):
            rev_list = list(reversed(self._stack))
            return rev_list[index]

        else:
            raise TypeError("Invalid argument type.")

    def __len__(self):
        """
        Return the number of items in the stack.

        Time Complexity: O(1)
        Space Complexity: O(1)

        Returns:
            int: The number of items in the stack.
        """
        return len(self._stack)

    def __bool__(self):
        """
        Determine whether the stack is non-empty.

        Time Complexity: O(1)
        Space Complexity: O(1)

        Returns:
            bool: True if the stack has at least one item, otherwise False.
        """
        return bool(self._stack)

    def peek(self):
        """
        Retrieve the item at the top of the stack without removing it.

        Time Complexity: O(1)
        Space Complexity: O(1)

        Returns:
            The item at the top of the stack.

        Raises:
            IndexError: If the stack is empty.
        """
        if not self._stack:
            raise IndexError("Stack is empty")
        return self._stack[-1]

    @property
    def top(self):
        """
        The top item of the stack as a property (read-only).

        Time Complexity: O(1)
        Space Complexity: O(1)

        Returns:
            The item at the top of the stack.

        Raises:
            IndexError: If the stack is empty.
        """
        return self.peek()

    def __repr__(self):
        """
        Return the string representation of the stack from top to bottom.

        Time Complexity: O(n)
        Space Complexity: O(1)

        Returns:
            str: A string representation of the stack.
        """
        return f"Stack([{', '.join(map(repr, reversed(self._stack)))}])"
