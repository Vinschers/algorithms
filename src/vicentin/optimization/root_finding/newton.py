from vicentin.utils import norm, solve, copy, scalar


def newton_raphson(f, Jf, x0, max_iter=100, tol=1e-6):
    """
    Solves a system of nonlinear equations using the Newton-Raphson method.

    The method iteratively updates the solution estimate `x` by solving the
    linear system:

        Jf(x) dx = f(x)

    where Jf(x) is the Jacobian matrix of f(x). The iteration stops when
    ||f(x)|| < tol or when the maximum number of iterations (`max_iter`)
    is reached.

    **Complexity Analysis:**
    - **Time Complexity:** O(N^3) per iteration (due to solving the linear system).
    - **Space Complexity:** O(N^2) (for storing the Jacobian matrix and function values).

    **Note:**
    - If `f` is a **single-variable function**, then `Jf` is its **derivative**.
    - If `f` is a **multi-variable function (but not a system)**, then `Jf` is the **transpose of its gradient**.
    - If `Jf(x)` is singular at any step, the method may fail to converge.
    - A good initial guess (`x0`) is crucial for ensuring convergence.

    Parameters:
    ----------
    f : callable
        Function that returns the system of nonlinear equations evaluated at `x`.
    Jf : callable
        Function that returns the Jacobian matrix of `f` at `x`.
    x0 : ndarray | Tensor
        Initial guess for the solution.
    max_iter : int, optional (default=100)
        Maximum number of iterations.
    tol : float, optional (default=1e-6)
        Tolerance for convergence. The algorithm stops when ||f(x)|| < tol.

    Returns:
    -------
    x : ndarray | Tensor
        The approximate solution to f(x) = 0.
    loss : list of float
        A list of ||f(x)|| at each iteration, useful for analyzing convergence.
    """

    x = copy(x0)

    loss = []

    for _ in range(max_iter):
        dx = solve(Jf(x), f(x))  # Solve Jf(x) dx = f(x)
        x = x - dx  # Update x

        f_norm = scalar(norm(f(x)))

        loss.append(f_norm)

        if f_norm < tol:
            break

    return x, loss
