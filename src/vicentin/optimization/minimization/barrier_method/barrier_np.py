from typing import Callable, Optional, Sequence
import numpy as np

from vicentin.optimization.minimization import newton_method

STANDARD_INEQUALITY = 0
LOG_INEQUALITY = 1


def barrier_functions(I: Sequence[Sequence], x0: np.ndarray):
    m = len(I)
    ineq_type = [STANDARD_INEQUALITY] * m
    inequalities = []

    for i, ineq in enumerate(I):
        if len(ineq) == 4:
            ineq_type[i] = ineq[3]
            inequalities.append(ineq[:3])
        else:
            inequalities.append(ineq)

    for i in range(m):
        f_i = inequalities[i][0](x0)
        if ineq_type[i] == STANDARD_INEQUALITY:
            if (f_i.ndim == 0 and f_i >= 0) or (
                f_i.ndim > 0 and np.any(f_i >= 0)
            ):
                raise ValueError("Initial point is not feasible.")

    def phi(x):
        y = 0

        for i in range(m):
            f, _, _ = inequalities[i]
            f_x = f(x)

            if ineq_type[i] == STANDARD_INEQUALITY:
                if np.any(f_x >= 0):
                    return np.inf

                y -= np.log(-f_x).sum()
            elif ineq_type[i] == LOG_INEQUALITY:
                if np.any(np.isnan(f_x)) or np.any(np.isinf(f_x)):
                    return np.inf

                y += f_x.sum()

        return y

    def grad_phi(x):
        grad = np.zeros_like(x)

        for i in range(m):
            f, grad_f, _ = inequalities[i]

            if ineq_type[i] == STANDARD_INEQUALITY:
                y = np.atleast_1d(f(x))
                J = np.atleast_2d(grad_f(x))

                grad -= J.T @ (1 / y)
            elif ineq_type[i] == LOG_INEQUALITY:
                g = np.atleast_1d(grad_f(x))
                grad += np.sum(g, axis=0) if g.ndim > x.ndim else g

        return grad

    def hess_phi(x):
        n = x.shape[0]
        hess = np.zeros((n, n))

        for i in range(m):
            f, grad_f, hess_f = inequalities[i]
            H = hess_f(x)

            if ineq_type[i] == STANDARD_INEQUALITY:
                y = np.atleast_1d(f(x))
                J = np.atleast_2d(grad_f(x))

                hess += (J.T * (1.0 / y**2)) @ J

                if H.ndim == 2:
                    H = H[None, ...]

                if H.ndim >= 2:
                    hess -= np.tensordot(1 / y, H, axes=([0], [0]))
            elif ineq_type[i] == LOG_INEQUALITY:
                if H.ndim > 2:
                    hess += np.sum(H, axis=0)
                else:
                    hess += H

        return hess

    return phi, grad_phi, hess_phi, inequalities, ineq_type


def barrier_method(
    F: Sequence[Callable],
    G: Sequence[Sequence[Callable]],
    x0: np.ndarray,
    equality: Optional[tuple] = None,
    max_iter: int = 100,
    epsilon: float = 1e-4,
    mu: float = 6,
    linear_solver: Optional[Callable] = None,
    return_dual: bool = False,
    return_loss: bool = False,
    return_t: bool = False,
):
    f, grad_f, hess_f = F

    phi, grad_phi, hess_phi, ineqs, ineq_types = barrier_functions(G, x0)

    x = x0.copy()
    t = 1
    m = len(ineqs)
    loss = []
    i = 1

    while True:
        F_phi = lambda z: t * f(z) + phi(z)
        grad_F_phi = lambda z: t * grad_f(z) + grad_phi(z)
        hess_F_phi = lambda z: t * hess_f(z) + hess_phi(z)

        x, w = newton_method(
            (F_phi, grad_F_phi, hess_F_phi),
            x,
            equality,
            linear_solver=linear_solver,
            return_dual=True,
        )

        loss.append(float(f(x)))

        duality_gap = m / t
        if duality_gap < epsilon:
            break

        t *= mu
        i += 1

        if i >= max_iter:
            break

    lambdas = []
    for i in range(m):
        lambda_ = 0

        if ineq_types[i] == STANDARD_INEQUALITY:
            f_i = ineqs[i][0](x)
            lambda_ = -1 / (t * f_i)
        elif ineq_types[i] == LOG_INEQUALITY:
            lambda_ = -ineqs[i][1](x) / t

        lambdas.append(lambda_)

    mu = w / t

    output = [x]

    if return_dual:
        output.append((lambdas, mu))

    if return_loss:
        output.append(loss)

    if return_t:
        output.append(t)

    if len(output) == 1:
        return output[0]

    return output
