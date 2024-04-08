from colors import bcolors


def lagrange_interpolation(x_data, y_data, x):
    """
    Lagrange Interpolation

    Parameters:
    x_data (list): List of x-values for data points.
    y_data (list): List of y-values for data points.
    x (float): The x-value where you want to evaluate the interpolated polynomial.

    Returns:
    tuple: Coefficients of the interpolated polynomial.
    float: The interpolated y-value at the given x.
    """
    # Input validation
    if len(x_data) != len(y_data) or len(x_data) == 0:
        raise ValueError("Invalid input data. x_data and y_data must have the same non-zero length.")

    n = len(x_data)
    result = 0.0
    polynomial_coefficients = []

    # Lagrange interpolation formula
    for i in range(n):
        term = y_data[i]
        for j in range(n):
            if i != j:
                term *= (x - x_data[j]) / (x_data[i] - x_data[j])
        result += term
        polynomial_coefficients.append(term)

    return tuple(polynomial_coefficients), result


if __name__ == '__main__':
    x_data = [1, 2, 4]
    y_data = [1, 0, 1.5]
    x_interpolate = 3  # The x-value where you want to interpolate

    try:
        polynomial_coefficients, y_interpolate = lagrange_interpolation(x_data, y_data, x_interpolate)
        print(bcolors.OKBLUE, "\nInterpolated value at x =", x_interpolate, "is y =", y_interpolate, bcolors.ENDC)
        print("Polynomial Coefficients:", polynomial_coefficients)
    except ValueError as e:
        print(bcolors.FAIL, "Error:", e, bcolors.ENDC)