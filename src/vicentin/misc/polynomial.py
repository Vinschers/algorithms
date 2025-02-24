from typing import List, Union


def horner(coefficients: List[Union[int, float]], x: Union[int, float]) -> Union[int, float]:
    """
    Evaluates a polynomial using Horner's method.

    Horner's method efficiently computes the value of a polynomial given
    its coefficients and a value for the variable x.

    The polynomial is assumed to be in the form:
        P(x) = a_n * x^n + a_(n-1) * x^(n-1) + ... + a_1 * x + a_0

    The coefficients list should be ordered from the highest degree term
    to the lowest, i.e., [a_n, a_(n-1), ..., a_1, a_0].

    Parameters:
    - coefficients (list of float/int): List of polynomial coefficients in descending order of powers.
    - x (float/int): The value at which the polynomial is to be evaluated.

    Returns:
    - float/int: The result of evaluating the polynomial at x.
    """
    if len(coefficients) == 0:
        return 0

    p = coefficients[0]

    for ai in coefficients[1:]:
        p = ai + x * p

    return p
