from vicentin.data_structures import Heap


def heapsort(array):
    heap = Heap(array)

    for i in range(heap.n, 2, -1):
        heap.swap(1, i)
        heap.n -= 1
        heap.heapify(1)

    return heap.array
