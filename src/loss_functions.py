import numpy as np
from numba import njit
from scipy.optimize import root_scalar
from src.root_finding import brent_root_finder, find_first_greather_than_zero

PROXIMAL_TOL = 1e-15
RTOL = 1e-10

BRAKETS_SCALAR_ROOT = 1e20
DIVISION_POINTS = 200
MULTIPLIER_NEAR_BUMP = 15
MULTIPLIER_NEAR_PARABOLA = 2


# ---


@njit(error_model="numpy", fastmath=True)
def loss_l2(y, z):
    return 0.5 * (y - z) ** 2


@njit(error_model="numpy", fastmath=True)
def Dloss_l2(y, z):
    return y - z


@njit(error_model="numpy", fastmath=True)
def DDloss_l2(y, z):
    return 1


# ----


@njit(error_model="numpy", fastmath=True)
def loss_double_quad(y, z, width):
    return 0.5 * (y - z) ** 2 + (y - z) ** 2 / ((y - z) ** 2 + width)


@njit(error_model="numpy", fastmath=True)
def Dloss_double_quad(y, z, width):
    return y - z + (2 * width * (y - z)) / (((y - z) ** 2 + width) ** 2)


@njit(error_model="numpy", fastmath=True)
def DDloss_double_quad(y, z, width):
    return 1 + 2 * width * (width - 3 * (y - z) ** 2) / (((y - z) ** 2 + width) ** 3)


@njit(error_model="numpy", fastmath=True)
def _proximal_argument_loss_double_quad(z, y, omega, V, width):
    return (
        0.5 * (z - omega) ** 2 / V
        + 0.5 * (y - z) ** 2
        + (y - z) ** 2 / ((y - z) ** 2 + width)
    )


@njit(error_model="numpy", fastmath=True)
def _proximal_argument_derivative_loss_double_quad(z, y, omega, V, width):
    return (z - omega) / V - y + z - (2 * width * (y - z)) / (((y - z) ** 2 + width) ** 2)


@njit(error_model="numpy", fastmath=True)
def _proximal_argument_second_derivative_loss_double_quad(z, y, omega, V, width):
    return (
        -1
        - 2 * width * (width - 3 * (y - z) ** 2) / (((y - z) ** 2 + width) ** 3)
        - 1 / V
    )


# @njit(error_model="numpy", fastmath=True)
# def _proximal_argument_both_derivatives_loss_double_quad(z, y, omega, V, width):
#     return (
#         (z - omega) / V - y + z - (2 * width * (y - z)) / (((y - z) ** 2 + width) ** 2),
#         (
#             -1
#             - 2 * width * (width - 3 * (y - z) ** 2) / (((y - z) ** 2 + width) ** 3)
#             - 1 / V
#         ),
#     )


# -----------------------------------

# -----------------------------------


@njit(error_model="numpy", fastmath=True)
def proximal_loss_double_quad(y, omega, V, width):
    # width = loss_args["width"]
    sgn = np.sign((V * y + omega) / (1 + V) - y)

    points_near_bump = np.linspace(
        y - MULTIPLIER_NEAR_BUMP * width / V,
        y + MULTIPLIER_NEAR_BUMP * width / V,
        DIVISION_POINTS,
    )
    fun_evals_near_bump = _proximal_argument_derivative_loss_double_quad(
        points_near_bump, y, omega, V, width
    )

    points_near_parabola = np.linspace(
        (V * y + omega) / (1 + V) - MULTIPLIER_NEAR_PARABOLA * V / (V + 1),
        (V * y + omega) / (1 + V) + MULTIPLIER_NEAR_PARABOLA * V / (V + 1),
        DIVISION_POINTS,
    )
    fun_evals_near_parabola = _proximal_argument_derivative_loss_double_quad(
        points_near_parabola, y, omega, V, width
    )

    proximal_val = y if sgn == 0 else None
    function_proximal_val = y if sgn == 0 else None

    if not np.all(sgn * fun_evals_near_bump <= 0):
        if sgn >= 0:
            first_index = 0
            second_index = find_first_greather_than_zero(fun_evals_near_bump, False)
        else:
            first_index = find_first_greather_than_zero(-fun_evals_near_bump, True)
            second_index = -1

        proximal_val = brent_root_finder(
            _proximal_argument_derivative_loss_double_quad,
            points_near_bump[first_index],
            points_near_bump[second_index],
            PROXIMAL_TOL,
            RTOL,
            10000,
            (y, omega, V, width),
        )
        function_proximal_val = _proximal_argument_loss_double_quad(
            proximal_val, y, omega, V, width
        )

    if not np.all(sgn * fun_evals_near_parabola >= 0):
        if sgn >= 0:
            first_index = find_first_greather_than_zero(-fun_evals_near_parabola, True)
            second_index = -1
        else:
            first_index = 0
            second_index = find_first_greather_than_zero(fun_evals_near_parabola, False)

        tmp_proximal = brent_root_finder(
            _proximal_argument_derivative_loss_double_quad,
            points_near_parabola[first_index],
            points_near_parabola[second_index],
            PROXIMAL_TOL,
            RTOL,
            10000,
            (y, omega, V, width),
        )
        tmp_function_proximal_val = _proximal_argument_loss_double_quad(
            tmp_proximal, y, omega, V, width
        )

        if proximal_val is None and function_proximal_val is None:
            proximal_val = tmp_proximal
            function_proximal_val = tmp_function_proximal_val

        if function_proximal_val > tmp_function_proximal_val:
            proximal_val = tmp_proximal
            function_proximal_val = tmp_function_proximal_val

    proximal_derivative = 1 / (1 + V * DDloss_double_quad(y, proximal_val, width))

    return proximal_val, proximal_derivative
