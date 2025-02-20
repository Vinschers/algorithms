def _max_sum_subsequence_aux(x, n):
    """
    A helper function that recursively finds the maximum sum subsequence in an array.

    This function computes the indices and values associated with the maximum sum 
    subsequence by iterating recursively through the array.

    Parameters:
    - x (list of int/float): The input list of numbers.
    - n (int): The current size of the sublist being considered.

    Returns:
    - i (int): The starting index of the maximum sum subsequence.
    - j (int): The ending index of the maximum sum subsequence.
    - k (int): The starting index of the current suffix sum.
    - max_seq (int/float): The maximum sum of a subsequence found so far.
    - max_suf (int/float): The maximum suffix sum ending at index `n`.
    """

    if n == 1 and x[0] < 0:
        i = j = k = 0
        max_seq = max_suf = 0
    elif n == 1 and x[0] >= 0:
        i = j = k = 1
        max_seq = max_suf = x[0]
    else:
        i, j, k, max_seq, max_suf = _max_sum_subsequence_aux(x, n - 1)

        if k == 0:
            k = n

        max_suf += x[-1]

        if max_suf > max_seq:
            i = k
            j = n
            max_seq = max_suf
        elif max_suf < 0:
            max_suf = 0
            k = 0

    return i, j, k, max_seq, max_suf

def max_sum_subsequence(x):
    """
    Finds the contiguous subsequence with the maximum sum in an array.

    Parameters:
    - x (list of int/float): The input list of numbers.

    Returns:
    - i (int): The starting index of the maximum sum subsequence.
    - j (int): The ending index of the maximum sum subsequence.
    - max_seq (int/float): The maximum sum of the subsequence.
    """

    i, j, _, max_seq, _ = _max_sum_subsequence_aux(x, len(x))
    return i, j, max_seq
