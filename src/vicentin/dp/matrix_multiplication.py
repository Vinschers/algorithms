from functools import lru_cache


def matrix_chain_order(dims: list[int]) -> int:
    @lru_cache
    def partition_cost(i, j):
        partition_costs = []

        for k in range(i + 1, j):
            cost = (
                partition_cost(i, k)
                + dims[i] * dims[k] * dims[j]
                + partition_cost(k, j)
            )
            partition_costs.append(cost)

        return min(partition_costs, default=0)

    return partition_cost(0, len(dims) - 1)
