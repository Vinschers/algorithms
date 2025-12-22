from collections import defaultdict


def rod_cut(L: int, cuts: dict):
    if L == 0:
        return 0

    r = defaultdict(float)
    r[0] = 0

    cut_sizes = sorted(cuts)

    for j in range(1, L + 1):
        q = -float("inf")

        for i in cut_sizes:
            if i > j:
                break

            price = cuts[i]
            q = max(q, price + r[j - i])

        r[j] = q

    return r[L]
