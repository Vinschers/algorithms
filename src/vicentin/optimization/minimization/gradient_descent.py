from vicentin.utils import norm, dot, sum, soft_threshold


def proximal_gradient_descent(F, dF, prox, x0, step_size, line_search=False, max_iter=100, tol=1e-6):
    """
    Performs proximal gradient descent to minimize an objective function with a non-smooth term.

    This algorithm minimizes functions of the form:
        F(x) = f(x) + g(x),
    where:
    - `f(x)` is a smooth, differentiable function.
    - `g(x)` is a possibly non-smooth function, handled using the proximal operator.

    Parameters:
    ----------
    F : callable
        The objective function to be minimized.
    dF : callable
        The gradient of the smooth part `f(x)`, returning a gradient vector at a given `x`.
    prox : callable
        The proximal operator for the non-smooth function `g(x)`, with the signature:
        `prox(x, gamma)`, where:
            - `x` is the input vector.
            - `gamma` is the step size.
    x0 : array-like
        The initial point for the optimization.
    step_size : float
        The initial step size (learning rate) for gradient descent.
    line_search : bool, optional (default=False)
        If True, applies backtracking line search to adaptively adjust `step_size`.
    max_iter : int, optional (default=100)
        The maximum number of iterations.
    tol : float, optional (default=1e-6)
        Convergence tolerance. The algorithm stops when `norm(grad) < tol`.

    Returns:
    -------
    x : array-like
        The optimal solution found by the algorithm.
    loss : list of floats
        A list of function values `F(x)` at each iteration.
    """

    x = x0.copy()
    loss = []

    for _ in range(max_iter):
        grad = dF(x)
        next_x = prox(x - step_size * grad)

        if line_search:
            gamma = step_size

            # Test Lipschitz condition
            while F(next_x) > F(x) + dot(grad, next_x - x) + sum((next_x - x) ** 2) / (2 * gamma):
                gamma /= 2
                next_x = prox(x - gamma * grad, gamma)

            step_size = 2 * gamma

        x = next_x

        loss.append(F(x))

        if norm(grad) < tol:
            break

    return x, loss


def gradient_descent(F, dF, x0, step_size, line_search=False, max_iter=100, tol=1e-6):
    """
    Performs gradient descent optimization to minimize a given function.

    Parameters:
    ----------
    F : callable
        The objective function to be minimized. It should take a vector `x` as input and return a scalar value.
    dF : callable
        The gradient of `F`, returning the derivative (gradient vector) at a given `x`.
    x0 : array-like
        The starting point for the optimization.
    step_size : float
        The initial step size (learning rate) for the gradient descent updates. Theoretically, the step size should be chosen based on the Lipschitz constant of the gradient.
    line_search : bool, optional (default=False)
        If True, uses a backtracking line search to adaptively adjust the step size.
    max_iter : int, optional (default=100)
        The maximum number of iterations before stopping.
    tol : float, optional (default=1e-6)
        The tolerance for stopping. If the norm of the gradient falls below `tol`, the algorithm stops.

    Returns:
    -------
    x : array-like
        The optimized value of `x` that minimizes `F`.
    loss : list
        A list of function values `F(x)` at each iteration.
    """
    return proximal_gradient_descent(F, dF, lambda x, gamma: x, x0, step_size, line_search, max_iter, tol)


def lasso_gradient_descent(F, dF, x0, step_size, lamb, line_search=False, max_iter=100, tol=1e-6):
    """
    Solves Lasso regression using proximal gradient descent.

    This function minimizes the Lasso objective:

        F(x) = f(x) + lambda * ||x||_1

    where:
    - `f(x)` is a smooth differentiable function (e.g., least squares loss).
    - `||x||_1` is the L1 regularization term, promoting sparsity.

    The optimization is performed using **proximal gradient descent**, where the
    non-smooth part `lambda * ||x||_1` is handled via the **soft-thresholding operator**.

    Parameters:
    ----------
    F : callable
        The objective function to be minimized (should include the squared loss term).
    dF : callable
        The gradient of the smooth part `f(x)`, returning a gradient vector.
    x0 : array-like
        The starting point for the optimization.
    step_size : float
        The initial step size (learning rate).
    lamb : float
        The regularization parameter for L1 penalty.
    line_search : bool, optional (default=False)
        If True, applies backtracking line search to adjust the step size.
    max_iter : int, optional (default=100)
        The maximum number of iterations.
    tol : float, optional (default=1e-6)
        Convergence tolerance. The algorithm stops when `norm(grad) < tol`.

    Returns:
    -------
    x : array-like
        The optimal solution found by the algorithm.
    loss : list of floats
        A list of function values `F(x)` at each iteration.
    """

    return proximal_gradient_descent(F, dF, lambda x, gamma: soft_threshold(x, gamma * lamb), x0, step_size, line_search, max_iter, tol)
