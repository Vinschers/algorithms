from vicentin.data_structures import MaxHeap


def heapsort(array):
    heap = MaxHeap(array)

    for i in range(heap.n, 2, -1):
        heap.swap(1, i)
        heap.n -= 1
        heap.max_heapify(1)

    return heap.array
