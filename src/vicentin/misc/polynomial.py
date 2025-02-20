def horner(coefficients, x):
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
    if len(coefficients) == 1:
        return coefficients[0]
    else:
        return coefficients[-1] + x * horner(coefficients[:-1], x)
