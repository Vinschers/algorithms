class Heap:
    """
    A mutable heap data structure that supports duplicate keys.

    This heap uses an array to store elements and an auxiliary dictionary (index_map)
    to keep track of the indices of each key, allowing for efficient updates and duplicate management.
    The primary operations (insert, remove_value, replace, and extract) run in O(log n) time on average,
    with a space complexity of O(n) for storing the heap and its associated index mapping.
    """

    def __init__(self, array=None, max=True):
        """
        Initialize a Heap instance.

        If an initial array is provided, the heap is built from the array using
        the bottom-up heap construction approach (heapify), which runs in O(n) time.
        The auxiliary index_map is built alongside to manage duplicate keys.

        Time Complexity: O(n)
        Space Complexity: O(n)

        Args:
            array (iterable, optional): An optional iterable of initial elements.
            max (bool): If True, creates a max-heap; if False, creates a min-heap.
        """
        self.max = max
        self.index_map = {}

        if array is None:
            self.array = []
        else:
            self.array = list(array)

        self.n = len(self.array)

        for i in range(self.n // 2 - 1, -1, -1):  # From n // 2 - 1 down to 0
            self.heapify(i)

        for i, key in enumerate(self.array):
            if key in self.index_map:
                self.index_map[key].add(i)
            else:
                self.index_map[key] = {i}

    def __len__(self):
        """
        Return the number of elements in the heap.

        Time Complexity: O(1)
        Space Complexity: O(1)

        Returns:
            int: The number of elements in the heap.
        """
        return self.n

    def __getitem__(self, index):
        """
        Retrieve the element at the specified index in the heap array.

        Time Complexity: O(1)
        Space Complexity: O(1)

        Args:
            index (int): The index of the element to retrieve.

        Returns:
            The element at the specified index.

        Raises:
            IndexError: If the index is out of the range of the heap.
        """
        if 0 <= index < self.n:
            return self.array[index]
        raise IndexError("Heap index out of range")

    def peek(self):
        """
        Return the root element of the heap without removing it.

        Time Complexity: O(1)
        Space Complexity: O(1)

        Returns:
            The root element of the heap.

        Raises:
            KeyError: If the heap is empty.
        """
        if self.n == 0:
            raise KeyError("Heap is empty")
        return self.array[0]

    def parent(self, i):
        """
        Compute the parent index for the given index in the heap.

        Time Complexity: O(1)
        Space Complexity: O(1)

        Args:
            i (int): The index of the current node.

        Returns:
            int: The index of the parent node.
        """
        return (i - 1) // 2

    def left(self, i):
        """
        Compute the index of the left child for the given node index.

        Time Complexity: O(1)
        Space Complexity: O(1)

        Args:
            i (int): The index of the current node.

        Returns:
            int: The index of the left child node.
        """
        return 2 * i + 1

    def right(self, i):
        """
        Compute the index of the right child for the given node index.

        Time Complexity: O(1)
        Space Complexity: O(1)

        Args:
            i (int): The index of the current node.

        Returns:
            int: The index of the right child node.
        """
        return 2 * i + 2

    def swap(self, i, j):
        """
        Swap the elements at indices i and j in the heap array and update the index_map accordingly.

        Time Complexity: O(1)
        Space Complexity: O(1)

        Args:
            i (int): The index of the first element.
            j (int): The index of the second element.
        """
        if i == j:
            return

        val_i, val_j = self.array[i], self.array[j]

        self.index_map[val_i].remove(i)
        self.index_map[val_j].remove(j)

        self.array[i], self.array[j] = self.array[j], self.array[i]

        self.index_map[val_i].add(j)
        self.index_map[val_j].add(i)

    def _heap_condition(self, a, b):
        """
        Check whether the heap property is satisfied between two values based on the heap type.

        For a max-heap, returns True if a > b.
        For a min-heap, returns True if a < b.

        Time Complexity: O(1)
        Space Complexity: O(1)

        Args:
            a: The first value.
            b: The second value.

        Returns:
            bool: True if the heap condition holds, False otherwise.
        """
        if self.max:
            return a > b
        else:
            return a < b

    def heapify(self, i):
        """
        Restore the heap property starting from index i by recursively moving the element down.

        Time Complexity: O(log n) in the worst case.
        Space Complexity: O(log n) due to the recursion stack (worst-case), O(1) auxiliary space otherwise.

        Args:
            i (int): The index from which to start heapifying.
        """
        index = i
        l = self.left(i)
        r = self.right(i)

        if l < self.n and self._heap_condition(self.array[l], self.array[index]):
            index = l

        if r < self.n and self._heap_condition(self.array[r], self.array[index]):
            index = r

        if index != i:
            self.swap(i, index)
            self.heapify(index)

    def bubble(self, i):
        """
        Restore the heap property by moving the element at index i upward until the heap property is restored.

        Time Complexity: O(log n) in the worst case.
        Space Complexity: O(1)

        Args:
            i (int): The index of the element to bubble up.
        """
        while i > 0 and not self._heap_condition(self.array[self.parent(i)], self.array[i]):
            self.swap(i, self.parent(i))
            i = self.parent(i)

    def index(self, key):
        """
        Return the smallest index of the given key from the index_map.

        Time Complexity: O(d) where d is the number of duplicate keys (worst-case O(n) if all elements are duplicates).
        Space Complexity: O(1)

        Args:
            key: The key to search for in the heap.

        Returns:
            int: The smallest index at which the key appears.

        Raises:
            KeyError: If the key is not found in the heap.
        """
        if key in self.index_map and self.index_map[key]:
            return min(self.index_map[key])
        else:
            raise KeyError("Key not found in the heap")

    def insert(self, key):
        """
        Inserts a new key into the heap.

        The key is appended to the heap array and its index is recorded in the index_map.
        After insertion, the heap property is restored by "bubbling" the key upward.

        Time Complexity: O(log n) on average.
        Space Complexity: O(1) auxiliary space.

        Args:
            key: The key to be inserted into the heap.

        Returns:
            self: The heap instance (allowing method chaining).
        """
        self.array.append(key)
        self.n += 1

        if key in self.index_map:
            self.index_map[key].add(self.n - 1)
        else:
            self.index_map[key] = {self.n - 1}

        self.bubble(self.n - 1)

        return self

    def remove_value(self, key):
        """
        Removes an instance of the specified key from the heap.

        The method locates the key using the index_map and swaps it with the last element in the heap.
        It then removes the last element and updates the index_map accordingly.
        Finally, the heap property is restored by either bubbling up or heapifying down the swapped element.

        Time Complexity: O(log n) on average.
        Space Complexity: O(1) auxiliary space.

        Args:
            key: The key to be removed from the heap.

        Raises:
            KeyError: If the heap is empty.
            ValueError: If the key is not found in the heap.
        """
        if self.n == 0:
            raise KeyError("Heap is empty")

        try:
            index = self.index(key)
        except KeyError:
            raise ValueError("Key not found in the heap")

        last_index = self.n - 1
        self.swap(index, last_index)
        self.index_map[key].remove(last_index)

        if not self.index_map[key]:
            del self.index_map[key]

        self.array.pop()
        self.n -= 1

        if index < self.n:
            parent_value = self.array[self.parent(index)]

            if index > 0 and self._heap_condition(self.array[index], parent_value):
                self.bubble(index)
            else:
                self.heapify(index)

    def replace(self, old, new):
        """
        Replaces an occurrence of an old key with a new key in the heap.

        The element with the old key is located and replaced by the new key.
        The index_map is updated accordingly and the heap property is restored
        by either bubbling up or heapifying down from the replacement index.

        Time Complexity: O(log n) on average.
        Space Complexity: O(1) auxiliary space.

        Args:
            old: The key to be replaced.
            new: The new key to replace the old key.

        Raises:
            ValueError: If the old key is not found in the heap.
        """
        if old == new:
            return

        try:
            index = self.index(old)
        except KeyError:
            raise ValueError("Key not found in the heap")

        self.array[index] = new

        self.index_map[old].remove(index)
        if not self.index_map[old]:
            del self.index_map[old]

        if new in self.index_map:
            self.index_map[new].add(index)
        else:
            self.index_map[new] = {index}

        if index > 0 and self._heap_condition(new, self.array[self.parent(index)]):
            self.bubble(index)
        else:
            self.heapify(index)

    def extract(self):
        """
        Removes and returns the root element of the heap.

        The root element (which is either the minimum or maximum element, depending on the heap type)
        is removed. The last element in the array is moved to the root position, and the heap property
        is restored by heapifying from the root.

        Time Complexity: O(log n) on average.
        Space Complexity: O(1) auxiliary space.

        Returns:
            The root element of the heap.

        Raises:
            KeyError: If the heap is empty.
        """
        if self.n == 0:
            raise KeyError("Heap is empty")

        root = self.array[0]

        self.index_map[root].remove(0)
        if not self.index_map[root]:
            del self.index_map[root]

        self.n -= 1

        if self.n == 0:
            self.array.pop()
            return root

        last_val = self.array.pop()
        self.array[0] = last_val

        self.index_map[last_val].remove(self.n)
        self.index_map[last_val].add(0)

        self.heapify(0)

        return root
