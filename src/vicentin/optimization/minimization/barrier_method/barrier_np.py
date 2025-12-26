from typing import Callable, Sequence
import numpy as np

from vicentin.optimization.minimization import newton_method


def barrier_functions(
    inequalities: list[Callable],
    grad_inequalities: list[Callable],
    hess_inequalities: list[Callable],
):
    def phi(x):
        y_arr = np.concatenate([np.atleast_1d(f(x)) for f in inequalities])

        if np.any(y_arr >= 0):
            return np.inf

        return -np.log(-y_arr).sum()

    def grad_phi(x):
        grad = np.zeros_like(x)

        for f, grad_f in zip(inequalities, grad_inequalities):
            y = np.atleast_1d(f(x))
            J = np.atleast_2d(grad_f(x))

            grad -= J.T @ (1 / y)

        return grad

    def hess_phi(x):
        n = x.shape[0]
        hess = np.zeros((n, n))

        for f, grad_f, hess_f in zip(
            inequalities, grad_inequalities, hess_inequalities
        ):
            y = np.atleast_1d(f(x))
            J = np.atleast_2d(grad_f(x))
            H = hess_f(x)

            hess += (J.T * (1.0 / y**2)) @ J

            if H.ndim == 2:
                H = H[None, ...]

            if H.ndim >= 2:
                hess -= np.tensordot(1 / y, H, axes=([0], [0]))

        return hess

    return phi, grad_phi, hess_phi


def barrier_method(
    F: Sequence[Callable],
    G: Sequence[list[Callable]],
    x0: np.ndarray,
    max_iter: int = 100,
    tol: float = 1e-5,
    epsilon: float = 1e-4,
    mu: float = 36,
    return_loss: bool = False,
):
    f, grad_f, hess_f = F
    inequalities, grad_inequalities, hess_inequalities = G

    phi, grad_phi, hess_phi = barrier_functions(
        inequalities, grad_inequalities, hess_inequalities
    )

    x = x0.copy()
    t = 1
    m = len(inequalities)
    loss = []
    i = 1

    while True:
        F_phi = lambda z: t * f(z) + phi(z)
        grad_F_phi = lambda z: t * grad_f(z) + grad_phi(z)
        hess_F_phi = lambda z: t * hess_f(z) + hess_phi(z)

        x = newton_method((F_phi, grad_F_phi, hess_F_phi), x)
        f_x = f(x)

        loss.append(f_x)
        if np.abs(f_x) < tol:
            break

        duality_gap = m / t
        if duality_gap < epsilon:
            break

        t *= mu
        i += 1

        if i >= max_iter:
            break

    return (x, loss) if return_loss else x
