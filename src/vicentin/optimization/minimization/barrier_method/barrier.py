from typing import Any, Callable, Optional, Sequence

from vicentin.utils import Dispatcher

dispatcher = Dispatcher()


try:
    from .barrier_np import barrier_method as barrier_np

    dispatcher.register("numpy", barrier_np)
except ModuleNotFoundError:
    pass

try:
    from .barrier_torch import barrier_method as barrier_torch

    dispatcher.register("torch", barrier_torch)
except ModuleNotFoundError:
    pass


def barrier_method(
    F: Sequence[Callable] | Callable,
    G: Sequence[list[Callable]] | Sequence[Callable],
    x0: Any,
    max_iter: int = 100,
    tol: float = 1e-5,
    epsilon: float = 1e-4,
    mu: float = 2,
    return_loss: bool = True,
    backend: Optional[str] = None,
):
    dispatcher.detect_backend(x0, backend)
    x0 = dispatcher.cast_values(x0)

    return dispatcher(F, G, x0, max_iter, tol, epsilon, mu, return_loss)


#
# import numpy as np
# import torch
#
# def run_comprehensive_tests(barrier_method_func):
#     """
#     barrier_method_func: The high-level barrier_method that uses the dispatcher.
#     """
#
#     # --- HELPER: Problem A (NumPy Setup) ---
#     def get_problem_a_numpy():
#         f = lambda x: np.sum(x**2)
#         grad_f = lambda x: 2 * x
#         hess_f = lambda x: 2 * np.eye(len(x))
#
#         # g(x) = 2 - x1 - x2 <= 0
#         g = lambda x: 2.0 - np.sum(x)
#         grad_g = lambda x: -np.ones_like(x)
#         hess_g = lambda x: np.zeros((len(x), len(x)))
#
#         F = (f, grad_f, hess_f)
#         G = [(g, grad_g, hess_g)]
#         x0 = np.array([5.0, 5.0]) # Must be strictly feasible
#         return F, G, x0
#
#     # --- HELPER: Problem C (Torch Setup) ---
#     def get_problem_c_torch():
#         # Objective: (x1-2)^2 + (x2-2)^2
#         f = lambda x: torch.sum((x - 2.0)**2)
#         # Constraint: x1^2 + x2^2 - 1 <= 0
#         g = lambda x: torch.sum(x**2) - 1.0
#
#         F = f
#         G = [g]
#         x0 = torch.tensor([0.1, 0.1], dtype=torch.float64) # Inside the disk
#         return F, G, x0
#
#     print("--- 🧪 Starting Physical Tests ---")
#
#     # TEST 1: NumPy Quadratic with Linear Inequality
#     print("\n[TEST 1] NumPy: QP (Minimize x^2 + y^2 s.t. x+y >= 2)")
#     F_a, G_a, x0_a = get_problem_a_numpy()
#     res_a = barrier_method_func(F_a, G_a, x0_a, mu=10, epsilon=1e-6)
#     x_opt_a = res_a[0] if isinstance(res_a, tuple) else res_a
#     print(f"Result: {x_opt_a}")
#     assert np.allclose(x_opt_a, [1.0, 1.0], atol=1e-3), f"Failed QP: {x_opt_a}"
#     print("✅ Test 1 Passed.")
#
#     # TEST 2: PyTorch Non-linear (Point to Disk)
#     print("\n[TEST 2] Torch: Non-linear (Minimize dist to (2,2) s.t. x^2 + y^2 <= 1)")
#     F_c, G_c, x0_c = get_problem_c_torch()
#     res_c = barrier_method_func(F_c, G_c, x0_c, mu=20, epsilon=1e-6)
#     x_opt_c = res_c[0] if isinstance(res_c, tuple) else res_c
#     print(f"Result: {x_opt_c}")
#     expected_c = 1.0 / np.sqrt(2)
#     assert np.allclose(x_opt_c.numpy(), [expected_c, expected_c], atol=1e-3), f"Failed Disk: {x_opt_c}"
#     print("✅ Test 2 Passed.")
#
#     # TEST 3: Mixed Argument Types (Keyword G)
#     print("\n[TEST 3] Mixed Arguments: Passing G as keyword")
#     # Using Problem A again
#     res_3 = barrier_method_func(F_a, G=G_a, x0=x0_a, mu=5)
#     x_opt_3 = res_3[0] if isinstance(res_3, tuple) else res_3
#     assert np.allclose(x_opt_3, [1.0, 1.0], atol=1e-2)
#     print("✅ Test 3 Passed.")
#
#     # TEST 4: Multiple Constraints (Box Constraints)
#     print("\n[TEST 4] NumPy: Multiple constraints (1 <= x <= 2)")
#     # f = (x-1.5)^2 -> min at 1.5
#     f_d = lambda x: (x[0] - 1.5)**2
#     df_d = lambda x: np.array([2 * (x[0] - 1.5)])
#     hf_d = lambda x: np.array([[2.0]])
#     # g1: 1 - x <= 0, g2: x - 2 <= 0
#     G_d = [
#         (lambda x: 1.0 - x[0], lambda x: np.array([-1.0]), lambda x: np.array([[0.0]])),
#         (lambda x: x[0] - 2.0, lambda x: np.array([1.0]), lambda x: np.array([[0.0]]))
#     ]
#     res_4 = barrier_method_func((f_d, df_d, hf_d), G_d, np.array([1.1]))
#     x_opt_4 = res_4[0] if isinstance(res_4, tuple) else res_4
#     print(f"Result: {x_opt_4}")
#     assert np.allclose(x_opt_4, [1.5], atol=1e-2)
#     print("✅ Test 4 Passed.")
#
#     print("\n🎉 ALL FUNCTIONAL TESTS PASSED SUCCESSFULLY!")
#
# # Usage:
# run_comprehensive_tests(barrier_method)
