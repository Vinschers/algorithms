def edit_distance(s1: str, s2: str) -> int:
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i

    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            deletion_cost = dp[i - 1][j] + 1
            insertion_cost = dp[i][j - 1] + 1
            substitution_cost = dp[i - 1][j - 1] + int(s1[i - 1] != s2[j - 1])

            dp[i][j] = min(insertion_cost, deletion_cost, substitution_cost)

    return dp[-1][-1]
