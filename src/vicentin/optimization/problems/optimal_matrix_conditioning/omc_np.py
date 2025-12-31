import numpy as np
from numpy import eye, zeros, stack
from numpy.linalg import eigvalsh

from vicentin.optimization.problems import SDP_dual


def optimal_matrix_conditioning(A: np.ndarray, epsilon: float = 1e-3):
    n = A.shape[0]
    m = n * (n + 1) // 2

    I = eye(n)
    zeros_matrix = zeros((n, n))

    b = zeros(m + 1)
    b[0] = -1

    lambdas_A = eigvalsh(A)

    if epsilon < 1 - lambdas_A.min():
        raise ValueError(f"epsilon must be greater than {1 - lambdas_A.min()}.")

    def E(k):
        M = zeros((n, n))

        i = 0
        while k >= (n - i):
            k -= n - i
            i += 1
        j = i + k

        M[i, j] = M[j, i] = 1
        return M

    E_basis = stack([E(k) for k in range(m)])

    # I <= X  =>  sum(y_i * -A_i) <= -I
    lmi_lower = [zeros_matrix] + (-E_basis).tolist() + [-I]

    # X <= tI => sum(y_i * A_i) - tI <= 0
    lmi_upper = [-I] + E_basis.tolist() + [zeros_matrix]

    # X - A <= epsilon*I => X <= A + epsilon*I
    lmi_near_pos = [zeros_matrix] + E_basis.tolist() + [A + epsilon * I]

    # A - X <= epsilon*I => -X <= epsilon*I - A
    lmi_near_neg = [zeros_matrix] + (-E_basis).tolist() + [epsilon * I - A]

    LMIs = [lmi_lower, lmi_upper, lmi_near_pos, lmi_near_neg]

    shift = (epsilon + max(lambdas_A.min(), 0)) / 2
    shift = min(epsilon / 2, shift)

    y0 = zeros(m + 1)
    y0[0] = lambdas_A.max() + 1 + shift
    X0 = A + shift * I

    k = 1
    for i in range(n):
        for j in range(i, n):
            y0[k] = X0[i, j]
            k += 1

    y_star = SDP_dual(b, LMIs, y0)

    X = zeros((n, n))
    k = 1
    for i in range(n):
        for j in range(i, n):
            val = y_star[k]
            X[i, j] = X[j, i] = val
            k += 1

    return X
