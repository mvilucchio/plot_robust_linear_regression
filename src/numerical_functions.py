import numpy as np
from scipy.integrate import dblquad
from numba import njit, vectorize
from src.loss_functions import proximal_loss_double_quad
from src.integration_utils import (
    find_integration_borders_square,
    divide_integration_borders_grid,
    domains_double_line_constraint,
    domains_double_line_constraint_only_inside,
    double_romb_integration,
    precompute_values_double_romb_integration,
    romberg_linspace,
    domains_line_constraint,
    divide_integration_borders_multiple_grid
)

MULT_INTEGRAL = 9
EPSABS = 1e-12
EPSREL = 1e-12


@njit(error_model="numpy", fastmath=True)
def _gaussian_function(y, x):
    return np.exp(-(x ** 2) / 2) / np.sqrt(2 * np.pi)


def precompute_proximals_loss_double_quad_grid(
    x_range, y_range, m, q, sigma, width,
):
    # create the array if it is not there

    shape = (len(x_range), len(y_range))
    proximals = np.empty(shape)
    proximal_derivatives = np.empty(shape)

    with np.nditer(
        [proximals, proximal_derivatives], flags=["multi_index"], op_flags=["readwrite"]
    ) as it:
        for prox, prox_der in it:
            prox[...], prox_der[...] = proximal_loss_double_quad(
                y_range[it.multi_index[0]],
                np.sqrt(q) * x_range[it.multi_index[1]],
                sigma,
                width,
            )

    return proximals, proximal_derivatives


# ------------------------------


@njit(error_model="numpy", fastmath=True)
def _ZoutBayes_single_noise_erm(y, xi, m, q, sigma, delta):
    eta = m ** 2 / q
    return np.exp(-((y - np.sqrt(eta) * xi) ** 2) / (2 * (1 - eta + delta))) / np.sqrt(
        2 * np.pi * (1 - eta + delta)
    )


@njit(error_model="numpy", fastmath=True)
def ZoutBayes_single_noise(y, omega, V, delta):
    return np.exp(-((y - omega) ** 2) / (2 * (V + delta))) / np.sqrt(
        2 * np.pi * (V + delta)
    )


@njit(error_model="numpy", fastmath=True)
def _foutBayes_single_noise_erm(y, xi, m, q, sigma, delta):
    eta = m ** 2 / q
    return (y - np.sqrt(eta) * xi) / (1 - eta + delta)


@njit(error_model="numpy", fastmath=True)
def foutBayes_single_noise(y, omega, V, delta):
    return (y - omega) / (V + delta)


# ----


@njit(error_model="numpy", fastmath=True)
def _ZoutBayes_double_noise_erm(y, xi, m, q, sigma, delta_small, delta_large, eps):
    eta = m ** 2 / q
    return (1 - eps) * np.exp(
        -((y - np.sqrt(eta) * xi) ** 2) / (2 * (1 - eta + delta_small))
    ) / np.sqrt(2 * np.pi * (1 - eta + delta_small)) + eps * np.exp(
        -((y - np.sqrt(eta) * xi) ** 2) / (2 * (1 - eta + delta_large))
    ) / np.sqrt(
        2 * np.pi * (1 - eta + delta_large)
    )


@njit(error_model="numpy", fastmath=True)
def ZoutBayes_double_noise(y, omega, V, delta_small, delta_large, eps):
    return np.exp(-((y - omega) ** 2) / (2 * (V + delta_large))) * (
        (1 - eps)
        * np.exp(
            ((y - omega) ** 2)
            * (1 / (2 * (V + delta_large)) - 1 / (2 * (V + delta_small)))
        )
        / np.sqrt(2 * np.pi * (V + delta_small))
        + eps / np.sqrt(2 * np.pi * (V + delta_large))
    )


@njit(error_model="numpy", fastmath=True)
def DZoutBayes_double_noise(y, omega, V, delta_small, delta_large, eps):
    return (y - omega) * (
        (1 - eps)
        * np.exp(-((y - omega) ** 2) / (2 * (V + delta_small)))
        / np.sqrt(2 * np.pi * (V + delta_small) ** 3)
        + eps
        * np.exp(-((y - omega) ** 2) / (2 * (V + delta_large)))
        / np.sqrt(2 * np.pi * (V + delta_large) ** 3)
    )


@njit(error_model="numpy", fastmath=True)
def _foutBayes_double_noise_erm(y, xi, m, q, sigma, delta_small, delta_large, eps):
    eta = m ** 2 / q
    small_exponential = np.exp(
        -((y - np.sqrt(eta) * xi) ** 2) / (2 * (1 - eta + delta_small))
    )
    large_exponential = np.exp(
        -((y - np.sqrt(eta) * xi) ** 2) / (2 * (1 - eta + delta_large))
    )
    return (
        (y - np.sqrt(eta) * xi)
        * (
            (1 - eps) * small_exponential / np.power(1 - eta + delta_small, 3 / 2)
            + eps * large_exponential / np.power(1 - eta + delta_large, 3 / 2)
        )
        / (
            (1 - eps) * small_exponential / np.power(1 - eta + delta_small, 1 / 2)
            + eps * large_exponential / np.power(1 - eta + delta_large, 1 / 2)
        )
    )


@njit(error_model="numpy", fastmath=True)
def foutBayes_double_noise(y, omega, V, delta_small, delta_large, eps):
    # small_exponential = np.exp(-((y - omega) ** 2) / (2 * (V + delta_small)))
    # large_exponential = np.exp(-((y - omega) ** 2) / (2 * (V + delta_large)))
    # return (
    #     (y - omega)
    #     * (
    #         (1 - eps) * small_exponential / np.sqrt((V + delta_small) ** 3)
    #         + eps * large_exponential / np.sqrt((V + delta_large) ** 3)
    #     )
    #     / (
    #         (1 - eps) * small_exponential / np.sqrt(V + delta_small)
    #         + eps * large_exponential / np.sqrt(V + delta_large)
    #     )
    # )
    exponential = np.exp(
        ((y - omega) ** 2) * (1 / (2 * (V + delta_large)) - 1 / (2 * (V + delta_small)))
    )
    return (
        (y - omega)
        * (
            (1 - eps) * exponential / np.sqrt((V + delta_small) ** 3)
            + eps / np.sqrt((V + delta_large) ** 3)
        )
        / (
            (1 - eps) * exponential / np.sqrt(V + delta_small)
            + eps / np.sqrt(V + delta_large)
        )
    )


# ----


@njit(error_model="numpy", fastmath=True)
def _ZoutBayes_decorrelated_noise_erm(
    y, xi, m, q, sigma, delta_small, delta_large, eps, beta
):
    eta = m ** 2 / q
    return (1 - eps) * np.exp(
        -((y - np.sqrt(eta) * xi) ** 2) / (2 * (1 - eta + delta_small))
    ) / np.sqrt(2 * np.pi * (1 - eta + delta_small)) + eps * np.exp(
        -((y - beta * np.sqrt(eta) * xi) ** 2)
        / (2 * (beta ** 2 * (1 - eta) + delta_large))
    ) / np.sqrt(
        2 * np.pi * (beta ** 2 * (1 - eta) + delta_large)
    )


@njit(error_model="numpy", fastmath=True)
def ZoutBayes_decorrelated_noise(y, omega, V, delta_small, delta_large, eps, beta):
    return (1 - eps) * np.exp(-((y - omega) ** 2) / (2 * (V + delta_small))) / np.sqrt(
        2 * np.pi * (V + delta_small)
    ) + eps * np.exp(
        -((y - beta * omega) ** 2) / (2 * (beta ** 2 * V + delta_large))
    ) / np.sqrt(
        2 * np.pi * (beta ** 2 * V + delta_large)
    )


@njit(error_model="numpy", fastmath=True)
def DZoutBayes_decorrelated_noise(y, omega, V, delta_small, delta_large, eps, beta):
    small_exponential = np.exp(-((y - omega) ** 2) / (2 * (V + delta_small))) / np.sqrt(
        2 * np.pi
    )
    large_exponential = np.exp(
        -((y - beta * omega) ** 2) / (2 * (beta ** 2 * V + delta_large))
    ) / np.sqrt(2 * np.pi)

    return (1 - eps) * small_exponential * (y - omega) / np.power(
        V + delta_small, 3 / 2
    ) + eps * beta * large_exponential * (y - beta * omega) / np.power(
        beta ** 2 * V + delta_large, 3 / 2
    )


@njit(error_model="numpy", fastmath=True)
def _foutBayes_decorrelated_noise_erm(
    y, xi, m, q, sigma, delta_small, delta_large, eps, beta
):
    eta = m ** 2 / q
    small_exponential = np.exp(
        -((y - np.sqrt(eta) * xi) ** 2) / (2 * (1 - eta + delta_small))
    )
    large_exponential = np.exp(
        -((y - beta * np.sqrt(eta) * xi) ** 2)
        / (2 * (beta ** 2 * (1 - eta) + delta_large))
    )
    return (
        (y - np.sqrt(eta) * xi)
        * (1 - eps)
        * small_exponential
        / np.power(1 - eta + delta_small, 3 / 2)
        + eps
        * beta
        * (y - beta * np.sqrt(eta) * xi)
        * large_exponential
        / np.power(beta ** 2 * (1 - eta) + delta_large, 3 / 2)
    ) / (
        (1 - eps) * small_exponential / np.power(1 - eta + delta_small, 1 / 2)
        + eps * large_exponential / np.power(beta ** 2 * (1 - eta) + delta_large, 1 / 2)
    )


@njit(error_model="numpy", fastmath=True)
def foutBayes_decorrelated_noise(y, omega, V, delta_small, delta_large, eps, beta):
    small_exponential = np.exp(-((y - omega) ** 2) / (2 * (V + delta_small)))
    large_exponential = np.exp(
        -((y - beta * omega) ** 2) / (2 * (beta ** 2 * V + delta_large))
    )
    return (
        (y - omega) * (1 - eps) * small_exponential / np.power(V + delta_small, 3 / 2)
        + eps
        * beta
        * (y - beta * omega)
        * large_exponential
        / np.power(beta ** 2 * V + delta_large, 3 / 2)
    ) / (
        (1 - eps) * small_exponential / np.power(V + delta_small, 1 / 2)
        + eps * large_exponential / np.power(beta ** 2 * V + delta_large, 1 / 2)
    )
    # exponential = np.exp(
    #     ((y - omega) ** 2) / (2 * (V + delta_small))
    #     - ((y - beta * omega) ** 2) / (2 * ((beta ** 2) * V + delta_large))
    # )
    # return (
    #     (y - omega)
    #     * (
    #         (1 - eps) / np.sqrt((V + delta_small) ** 3)
    #         + eps * exponential / np.sqrt(((beta ** 2) * V + delta_large) ** 3)
    #     )
    #     / (
    #         (1 - eps) / np.sqrt(V + delta_small)
    #         + eps * exponential / np.sqrt((beta ** 2) * V + delta_large)
    #     )
    # )


# -------


@njit(error_model="numpy", fastmath=True)
def foutL2(y, omega, V):
    return (y - omega) / (1 + V)


@njit(error_model="numpy", fastmath=True)
def DfoutL2(y, omega, V):
    return -1.0 / (1 + V)


@njit(error_model="numpy", fastmath=True)
def foutL1(y, omega, V):
    return (y - omega + np.sign(omega - y) * np.maximum(np.abs(omega - y) - V, 0.0)) / V


@vectorize  # @njit(error_model="numpy", fastmath=True)
def DfoutL1(y, omega, V):
    if np.abs(omega - y) > V:
        return 0.0
    else:
        return -1.0 / V


@vectorize
def foutHuber(y, omega, V, a):
    if a + a * V + omega < y:
        return a
    elif np.abs(y - omega) <= a + a * V:
        return (y - omega) / (1 + V)
    elif omega > a + a * V + y:
        return -a
    else:
        return 0.0


@vectorize
def DfoutHuber(y, omega, V, a):
    if (y < omega and a + a * V + y < omega) or (a + a * V + omega < y):
        return 0.0
    else:
        return -1.0 / (1 + V)


# --------------
# Functions to integrate - Single Noise
# --------------


@njit(error_model="numpy", fastmath=True)
def q_integral_BO_single_noise(y, xi, q, m, sigma, delta):
    return (
        np.exp(-(xi ** 2) / 2)
        / np.sqrt(2 * np.pi)
        * ZoutBayes_single_noise(y, np.sqrt(q) * xi, 1 - q, delta)
        * (foutBayes_single_noise(y, np.sqrt(q) * xi, 1 - q, delta) ** 2)
    )


# ----


@njit(error_model="numpy", fastmath=True)
def m_integral_L2_single_noise(y, xi, q, m, sigma, delta):
    eta = m ** 2 / q
    return (
        np.exp(-(xi ** 2) / 2)
        / np.sqrt(2 * np.pi)
        * ZoutBayes_single_noise(y, np.sqrt(eta) * xi, (1 - eta), delta)
        * foutBayes_single_noise(y, np.sqrt(eta) * xi, (1 - eta), delta)
        * foutL2(y, np.sqrt(q) * xi, sigma)
    )


@njit(error_model="numpy", fastmath=True)
def q_integral_L2_single_noise(y, xi, q, m, sigma, delta):
    eta = m ** 2 / q
    return (
        np.exp(-(xi ** 2) / 2)
        / np.sqrt(2 * np.pi)
        * ZoutBayes_single_noise(y, np.sqrt(eta) * xi, (1 - eta), delta)
        * (foutL2(y, np.sqrt(q) * xi, sigma) ** 2)
    )


@njit(error_model="numpy", fastmath=True)
def sigma_integral_L2_single_noise(y, xi, q, m, sigma, delta):
    eta = m ** 2 / q
    return (
        np.exp(-(xi ** 2) / 2)
        / np.sqrt(2 * np.pi)
        * ZoutBayes_single_noise(y, np.sqrt(eta) * xi, (1 - eta), delta)
        * DfoutL2(y, np.sqrt(q) * xi, sigma)
    )


# ----


@njit(error_model="numpy", fastmath=True)
def m_integral_L1_single_noise(y, xi, q, m, sigma, delta):
    eta = m ** 2 / q
    return (
        np.exp(-(xi ** 2) / 2)
        / np.sqrt(2 * np.pi)
        * ZoutBayes_single_noise(y, np.sqrt(eta) * xi, (1 - eta), delta)
        * foutBayes_single_noise(y, np.sqrt(eta) * xi, (1 - eta), delta)
        * foutL1(y, np.sqrt(q) * xi, sigma)
    )


@njit(error_model="numpy", fastmath=True)
def q_integral_L1_single_noise(y, xi, q, m, sigma, delta):
    eta = m ** 2 / q
    return (
        np.exp(-(xi ** 2) / 2)
        / np.sqrt(2 * np.pi)
        * ZoutBayes_single_noise(y, np.sqrt(eta) * xi, (1 - eta), delta)
        * (foutL1(y, np.sqrt(q) * xi, sigma) ** 2)
    )


@njit(error_model="numpy", fastmath=True)
def sigma_integral_L1_single_noise(y, xi, q, m, sigma, delta):
    eta = m ** 2 / q
    return (
        np.exp(-(xi ** 2) / 2)
        / np.sqrt(2 * np.pi)
        * ZoutBayes_single_noise(y, np.sqrt(eta) * xi, (1 - eta), delta)
        * DfoutL1(y, np.sqrt(q) * xi, sigma)
    )


# ----


@njit(error_model="numpy", fastmath=True)
def m_integral_Huber_single_noise(y, xi, q, m, sigma, delta, a):
    eta = m ** 2 / q
    return (
        np.exp(-(xi ** 2) / 2)
        / np.sqrt(2 * np.pi)
        * ZoutBayes_single_noise(y, np.sqrt(eta) * xi, (1 - eta), delta)
        * foutBayes_single_noise(y, np.sqrt(eta) * xi, (1 - eta), delta)
        * foutHuber(y, np.sqrt(q) * xi, sigma, a)
    )


@njit(error_model="numpy", fastmath=True)
def q_integral_Huber_single_noise(y, xi, q, m, sigma, delta, a):
    eta = m ** 2 / q
    return (
        np.exp(-(xi ** 2) / 2)
        / np.sqrt(2 * np.pi)
        * ZoutBayes_single_noise(y, np.sqrt(eta) * xi, (1 - eta), delta)
        * (foutHuber(y, np.sqrt(q) * xi, sigma, a) ** 2)
    )


@njit(error_model="numpy", fastmath=True)
def sigma_integral_Huber_single_noise(y, xi, q, m, sigma, delta, a):
    eta = m ** 2 / q
    return (
        np.exp(-(xi ** 2) / 2)
        / np.sqrt(2 * np.pi)
        * ZoutBayes_single_noise(y, np.sqrt(eta) * xi, (1 - eta), delta)
        * DfoutHuber(y, np.sqrt(q) * xi, sigma, a)
    )


# --------------
# Functions to integrate - Double Noise
# --------------


@njit(error_model="numpy", fastmath=True)
def q_integral_BO_double_noise(y, xi, q, m, sigma, delta_small, delta_large, eps):
    return (
        np.exp(-(xi ** 2) / 2)
        / np.sqrt(2 * np.pi)
        * ZoutBayes_double_noise(y, np.sqrt(q) * xi, 1 - q, delta_small, delta_large, eps)
        * (
            foutBayes_double_noise(
                y, np.sqrt(q) * xi, 1 - q, delta_small, delta_large, eps
            )
            ** 2
        )
    )


#  ----


@njit(error_model="numpy", fastmath=True)
def m_integral_L2_double_noise(y, xi, q, m, sigma, delta_small, delta_large, eps):
    eta = m ** 2 / q
    return (
        np.exp(-(xi ** 2) / 2)
        / np.sqrt(2 * np.pi)
        * ZoutBayes_double_noise(
            y, np.sqrt(eta) * xi, (1 - eta), delta_small, delta_large, eps
        )
        * foutBayes_double_noise(
            y, np.sqrt(eta) * xi, (1 - eta), delta_small, delta_large, eps
        )
        * foutL2(y, np.sqrt(q) * xi, sigma)
    )


@njit(error_model="numpy", fastmath=True)
def q_integral_L2_double_noise(y, xi, q, m, sigma, delta_small, delta_large, eps):
    eta = m ** 2 / q
    return (
        np.exp(-(xi ** 2) / 2)
        / np.sqrt(2 * np.pi)
        * ZoutBayes_double_noise(
            y, np.sqrt(eta) * xi, (1 - eta), delta_small, delta_large, eps
        )
        * (foutL2(y, np.sqrt(q) * xi, sigma) ** 2)
    )


@njit(error_model="numpy", fastmath=True)
def sigma_integral_L2_double_noise(y, xi, q, m, sigma, delta_small, delta_large, eps):
    eta = m ** 2 / q
    return (
        np.exp(-(xi ** 2) / 2)
        / np.sqrt(2 * np.pi)
        * ZoutBayes_double_noise(
            y, np.sqrt(eta) * xi, (1 - eta), delta_small, delta_large, eps
        )
        * DfoutL2(y, np.sqrt(q) * xi, sigma)
    )


# ----


@njit(error_model="numpy", fastmath=True)
def m_integral_L1_double_noise(y, xi, q, m, sigma, delta_small, delta_large, eps):
    eta = m ** 2 / q
    return (
        np.exp(-(xi ** 2) / 2)
        / np.sqrt(2 * np.pi)
        * ZoutBayes_double_noise(
            y, np.sqrt(eta) * xi, (1 - eta), delta_small, delta_large, eps
        )
        * foutBayes_double_noise(
            y, np.sqrt(eta) * xi, (1 - eta), delta_small, delta_large, eps
        )
        * foutL1(y, np.sqrt(q) * xi, sigma)
    )


@njit(error_model="numpy", fastmath=True)
def q_integral_L1_double_noise(y, xi, q, m, sigma, delta_small, delta_large, eps):
    eta = m ** 2 / q
    return (
        np.exp(-(xi ** 2) / 2)
        / np.sqrt(2 * np.pi)
        * ZoutBayes_double_noise(
            y, np.sqrt(eta) * xi, (1 - eta), delta_small, delta_large, eps
        )
        * (foutL1(y, np.sqrt(q) * xi, sigma) ** 2)
    )


@njit(error_model="numpy", fastmath=True)
def sigma_integral_L1_double_noise(y, xi, q, m, sigma, delta_small, delta_large, eps):
    eta = m ** 2 / q
    return (
        np.exp(-(xi ** 2) / 2)
        / np.sqrt(2 * np.pi)
        * ZoutBayes_double_noise(
            y, np.sqrt(eta) * xi, (1 - eta), delta_small, delta_large, eps
        )
        * DfoutL1(y, np.sqrt(q) * xi, sigma)
    )


# ----


@njit(error_model="numpy", fastmath=True)
def m_integral_Huber_double_noise(y, xi, q, m, sigma, delta_small, delta_large, eps, a):
    eta = m ** 2 / q
    return (
        np.exp(-(xi ** 2) / 2)
        / np.sqrt(2 * np.pi)
        * ZoutBayes_double_noise(
            y, np.sqrt(eta) * xi, (1 - eta), delta_small, delta_large, eps
        )
        * foutBayes_double_noise(
            y, np.sqrt(eta) * xi, (1 - eta), delta_small, delta_large, eps
        )
        * foutHuber(y, np.sqrt(q) * xi, sigma, a)
    )


@njit(error_model="numpy", fastmath=True)
def q_integral_Huber_double_noise(y, xi, q, m, sigma, delta_small, delta_large, eps, a):
    eta = m ** 2 / q
    return (
        np.exp(-(xi ** 2) / 2)
        / np.sqrt(2 * np.pi)
        * ZoutBayes_double_noise(
            y, np.sqrt(eta) * xi, (1 - eta), delta_small, delta_large, eps
        )
        * (foutHuber(y, np.sqrt(q) * xi, sigma, a) ** 2)
    )


@njit(error_model="numpy", fastmath=True)
def sigma_integral_Huber_double_noise(
    y, xi, q, m, sigma, delta_small, delta_large, eps, a
):
    eta = m ** 2 / q
    return (
        np.exp(-(xi ** 2) / 2)
        / np.sqrt(2 * np.pi)
        * ZoutBayes_double_noise(
            y, np.sqrt(eta) * xi, (1 - eta), delta_small, delta_large, eps
        )
        * DfoutHuber(y, np.sqrt(q) * xi, sigma, a)
    )


# ----


@njit(error_model="numpy", fastmath=True)
def m_integral_numerical_loss_double_noise(
    y, xi, q, m, sigma, delta_small, delta_large, eps, a
):
    eta = m ** 2 / q
    return (
        np.exp(-(xi ** 2) / 2)
        / np.sqrt(2 * np.pi)
        * ZoutBayes_double_noise(
            y, np.sqrt(eta) * xi, (1 - eta), delta_small, delta_large, eps
        )
        * foutBayes_double_noise(
            y, np.sqrt(eta) * xi, (1 - eta), delta_small, delta_large, eps
        )
        * foutHuber(y, np.sqrt(q) * xi, sigma, a)
    )


@njit(error_model="numpy", fastmath=True)
def q_integral_numerical_loss_double_noise(
    y, xi, q, m, sigma, delta_small, delta_large, eps, a
):
    eta = m ** 2 / q
    return (
        np.exp(-(xi ** 2) / 2)
        / np.sqrt(2 * np.pi)
        * ZoutBayes_double_noise(
            y, np.sqrt(eta) * xi, (1 - eta), delta_small, delta_large, eps
        )
        * (foutHuber(y, np.sqrt(q) * xi, sigma, a) ** 2)
    )


@njit(error_model="numpy", fastmath=True)
def sigma_integral_numerical_loss_double_noise(
    y, xi, q, m, sigma, delta_small, delta_large, eps, a
):
    eta = m ** 2 / q
    return (
        np.exp(-(xi ** 2) / 2)
        / np.sqrt(2 * np.pi)
        * ZoutBayes_double_noise(
            y, np.sqrt(eta) * xi, (1 - eta), delta_small, delta_large, eps
        )
        * DfoutHuber(y, np.sqrt(q) * xi, sigma, a)
    )


# --------------
# Functions to integrate - Decorrelated Noise
# --------------


@njit(error_model="numpy", fastmath=True)
def q_integral_BO_decorrelated_noise(
    y, xi, q, m, sigma, delta_small, delta_large, eps, beta
):
    return (
        np.exp(-(xi ** 2) / 2)
        / np.sqrt(2 * np.pi)
        * ZoutBayes_decorrelated_noise(
            y, np.sqrt(q) * xi, 1 - q, delta_small, delta_large, eps, beta
        )
        * (
            foutBayes_decorrelated_noise(
                y, np.sqrt(q) * xi, 1 - q, delta_small, delta_large, eps, beta
            )
            ** 2
        )
    )


#  ----


@njit(error_model="numpy", fastmath=True)
def m_integral_L2_decorrelated_noise(
    y, xi, q, m, sigma, delta_small, delta_large, eps, beta
):
    eta = m ** 2 / q
    return (
        np.exp(-(xi ** 2) / 2)
        / np.sqrt(2 * np.pi)
        * ZoutBayes_decorrelated_noise(
            y, np.sqrt(eta) * xi, (1 - eta), delta_small, delta_large, eps, beta
        )
        * foutBayes_decorrelated_noise(
            y, np.sqrt(eta) * xi, (1 - eta), delta_small, delta_large, eps, beta
        )
        * foutL2(y, np.sqrt(q) * xi, sigma)
    )


@njit(error_model="numpy", fastmath=True)
def q_integral_L2_decorrelated_noise(
    y, xi, q, m, sigma, delta_small, delta_large, eps, beta
):
    eta = m ** 2 / q
    return (
        np.exp(-(xi ** 2) / 2)
        / np.sqrt(2 * np.pi)
        * ZoutBayes_decorrelated_noise(
            y, np.sqrt(eta) * xi, (1 - eta), delta_small, delta_large, eps, beta
        )
        * (foutL2(y, np.sqrt(q) * xi, sigma) ** 2)
    )


@njit(error_model="numpy", fastmath=True)
def sigma_integral_L2_decorrelated_noise(
    y, xi, q, m, sigma, delta_small, delta_large, eps, beta
):
    eta = m ** 2 / q
    return (
        np.exp(-(xi ** 2) / 2)
        / np.sqrt(2 * np.pi)
        * ZoutBayes_decorrelated_noise(
            y, np.sqrt(eta) * xi, (1 - eta), delta_small, delta_large, eps, beta
        )
        * DfoutL2(y, np.sqrt(q) * xi, sigma)
    )


# ----


@njit(error_model="numpy", fastmath=True)
def m_integral_L1_decorrelated_noise(
    y, xi, q, m, sigma, delta_small, delta_large, eps, beta
):
    eta = m ** 2 / q
    return (
        np.exp(-(xi ** 2) / 2)
        / np.sqrt(2 * np.pi)
        * ZoutBayes_decorrelated_noise(
            y, np.sqrt(eta) * xi, (1 - eta), delta_small, delta_large, eps, beta
        )
        * foutBayes_decorrelated_noise(
            y, np.sqrt(eta) * xi, (1 - eta), delta_small, delta_large, eps, beta
        )
        * foutL1(y, np.sqrt(q) * xi, sigma)
    )


@njit(error_model="numpy", fastmath=True)
def q_integral_L1_decorrelated_noise(
    y, xi, q, m, sigma, delta_small, delta_large, eps, beta
):
    eta = m ** 2 / q
    return (
        np.exp(-(xi ** 2) / 2)
        / np.sqrt(2 * np.pi)
        * ZoutBayes_decorrelated_noise(
            y, np.sqrt(eta) * xi, (1 - eta), delta_small, delta_large, eps, beta
        )
        * (foutL1(y, np.sqrt(q) * xi, sigma) ** 2)
    )


@njit(error_model="numpy", fastmath=True)
def sigma_integral_L1_decorrelated_noise(
    y, xi, q, m, sigma, delta_small, delta_large, eps, beta
):
    eta = m ** 2 / q
    return (
        np.exp(-(xi ** 2) / 2)
        / np.sqrt(2 * np.pi)
        * ZoutBayes_decorrelated_noise(
            y, np.sqrt(eta) * xi, (1 - eta), delta_small, delta_large, eps, beta
        )
        * DfoutL1(y, np.sqrt(q) * xi, sigma)
    )


# ----


@njit(error_model="numpy", fastmath=True)
def m_integral_Huber_decorrelated_noise(
    y, xi, q, m, sigma, delta_small, delta_large, eps, beta, a
):
    eta = m ** 2 / q
    return (
        np.exp(-(xi ** 2) / 2)
        / np.sqrt(2 * np.pi)
        * DZoutBayes_decorrelated_noise(
            y, np.sqrt(eta) * xi, (1 - eta), delta_small, delta_large, eps, beta
        )
        * foutHuber(y, np.sqrt(q) * xi, sigma, a)
    )


@njit(error_model="numpy", fastmath=True)
def q_integral_Huber_decorrelated_noise(
    y, xi, q, m, sigma, delta_small, delta_large, eps, beta, a
):
    eta = m ** 2 / q
    return (
        np.exp(-(xi ** 2) / 2)
        / np.sqrt(2 * np.pi)
        * ZoutBayes_decorrelated_noise(
            y, np.sqrt(eta) * xi, (1 - eta), delta_small, delta_large, eps, beta
        )
        * (foutHuber(y, np.sqrt(q) * xi, sigma, a) ** 2)
    )


@njit(error_model="numpy", fastmath=True)
def sigma_integral_Huber_decorrelated_noise(
    y, xi, q, m, sigma, delta_small, delta_large, eps, beta, a
):
    eta = m ** 2 / q
    return (
        np.exp(-(xi ** 2) / 2)
        / np.sqrt(2 * np.pi)
        * ZoutBayes_decorrelated_noise(
            y, np.sqrt(eta) * xi, (1 - eta), delta_small, delta_large, eps, beta
        )
        * DfoutHuber(y, np.sqrt(q) * xi, sigma, a)
    )


# -----------


def border_plus_L1(xi, m, q, sigma):
    return np.sqrt(q) * xi + sigma


def border_minus_L1(xi, m, q, sigma):
    return np.sqrt(q) * xi - sigma


def test_fun_upper_L1(y, m, q, sigma):
    return (y - sigma) / np.sqrt(q)


def test_fun_down_L1(y, m, q, sigma):
    return (y + sigma) / np.sqrt(q)


def border_plus_Huber(xi, m, q, sigma, a):
    return np.sqrt(q) * xi + a * (sigma + 1)


def border_minus_Huber(xi, m, q, sigma, a):
    return np.sqrt(q) * xi - a * (sigma + 1)


def test_fun_upper_Huber(y, m, q, sigma, a):
    return 1 / np.sqrt(q) * (-a * (sigma + 1) + y)


def test_fun_down_Huber(y, m, q, sigma, a):
    return 1 / np.sqrt(q) * (a * (sigma + 1) + y)


def border_BO(xi, m, q, sigma):
    return np.sqrt(q) * xi


def test_fun_BO(y, m, q, sigma):
    return y / np.sqrt(q)


# ------------------
# BayesOpt equations single noise
# ------------------


def q_hat_equation_BO_single_noise(m, q, sigma, delta):
    borders = find_integration_borders_square(
        lambda y, xi: q_integral_BO_single_noise(y, xi, q, m, sigma, delta),
        np.sqrt((1 + delta)),
        1.0,
    )
    return dblquad(
        q_integral_BO_single_noise,
        borders[0][0],
        borders[0][1],
        borders[1][0],
        borders[1][1],
        args=(q, m, sigma, delta),
        epsabs=EPSABS,
        epsrel=EPSREL,
    )[0]


# ------------------
# L2 equations single noise
# ------------------


def m_hat_equation_L2_single_noise(m, q, sigma, delta):
    borders = find_integration_borders_square(
        lambda y, xi: m_integral_L2_single_noise(y, xi, q, m, sigma, delta),
        np.sqrt((1 + delta)),
        1.0,
    )
    return dblquad(
        m_integral_L2_single_noise,
        borders[0][0],
        borders[0][1],
        borders[1][0],
        borders[1][1],
        args=(q, m, sigma, delta),
        epsabs=EPSABS,
        epsrel=EPSREL,
    )[0]


def q_hat_equation_L2_single_noise(m, q, sigma, delta):
    borders = find_integration_borders_square(
        lambda y, xi: q_integral_L2_single_noise(y, xi, q, m, sigma, delta),
        np.sqrt((1 + delta)),
        1.0,
    )
    return dblquad(
        q_integral_L2_single_noise,
        borders[0][0],
        borders[0][1],
        borders[1][0],
        borders[1][1],
        args=(q, m, sigma, delta),
        epsabs=EPSABS,
        epsrel=EPSREL,
    )[0]


def sigma_hat_equation_L2_single_noise(m, q, sigma, delta):
    borders = find_integration_borders_square(
        lambda y, xi: sigma_integral_L2_single_noise(y, xi, q, m, sigma, delta),
        np.sqrt((1 + delta)),
        1.0,
    )
    return dblquad(
        sigma_integral_L2_single_noise,
        borders[0][0],
        borders[0][1],
        borders[1][0],
        borders[1][1],
        args=(q, m, sigma, delta),
        epsabs=EPSABS,
        epsrel=EPSREL,
    )[0]


# ------------------
# L1 equations single noise
# ------------------


def m_hat_equation_L1_single_noise(m, q, sigma, delta):
    borders = find_integration_borders_square(
        lambda y, xi: m_integral_L1_single_noise(y, xi, q, m, sigma, delta),
        np.sqrt((1 + delta)),
        1.0,
    )

    args = {"m": m, "q": q, "sigma": sigma}
    domain_xi, domain_y = domains_double_line_constraint(
        borders, border_plus_L1, border_minus_L1, test_fun_upper_L1, args, args, args,
    )

    integral_value = 0.0
    for xi_funs, y_funs in zip(domain_xi, domain_y):
        integral_value += dblquad(
            m_integral_L1_single_noise,
            xi_funs[0],
            xi_funs[1],
            y_funs[0],
            y_funs[1],
            args=(q, m, sigma, delta),
            epsabs=EPSABS,
            epsrel=EPSREL,
        )[0]

    return integral_value


def q_hat_equation_L1_single_noise(m, q, sigma, delta):
    borders = find_integration_borders_square(
        lambda y, xi: q_integral_L1_single_noise(y, xi, q, m, sigma, delta),
        np.sqrt((1 + delta)),
        1.0,
    )

    args = {"m": m, "q": q, "sigma": sigma}
    domain_xi, domain_y = domains_double_line_constraint(
        borders, border_plus_L1, border_minus_L1, test_fun_upper_L1, args, args, args,
    )

    integral_value = 0.0
    for xi_funs, y_funs in zip(domain_xi, domain_y):
        integral_value += dblquad(
            q_integral_L1_single_noise,
            xi_funs[0],
            xi_funs[1],
            y_funs[0],
            y_funs[1],
            args=(q, m, sigma, delta),
            epsabs=EPSABS,
            epsrel=EPSREL,
        )[0]

    return integral_value


def sigma_hat_equation_L1_single_noise(m, q, sigma, delta):
    borders = find_integration_borders_square(
        lambda y, xi: q_integral_L1_single_noise(y, xi, q, m, sigma, delta),
        np.sqrt((1 + delta)),
        1.0,
    )

    args = {"m": m, "q": q, "sigma": sigma}
    domain_xi, domain_y = domains_double_line_constraint(
        borders, border_plus_L1, border_minus_L1, test_fun_upper_L1, args, args, args,
    )

    integral_value = 0.0
    for xi_funs, y_funs in zip(domain_xi, domain_y):
        integral_value += dblquad(
            sigma_integral_L1_single_noise,
            xi_funs[0],
            xi_funs[1],
            y_funs[0],
            y_funs[1],
            args=(q, m, sigma, delta),
            epsabs=EPSABS,
            epsrel=EPSREL,
        )[0]

    return integral_value


# ------------------
# Huber equations single noise
# ------------------


def m_hat_equation_Huber_single_noise(m, q, sigma, delta, a):
    borders = find_integration_borders_square(
        lambda y, xi: m_integral_Huber_single_noise(y, xi, q, m, sigma, delta, a),
        np.sqrt((1 + delta)),
        1.0,
    )

    args = {"m": m, "q": q, "sigma": sigma, "a": a}
    domain_xi, domain_y = domains_double_line_constraint(
        borders,
        border_plus_Huber,
        border_minus_Huber,
        test_fun_upper_Huber,
        args,
        args,
        args,
    )

    integral_value = 0.0
    for xi_funs, y_funs in zip(domain_xi, domain_y):
        integral_value += dblquad(
            m_integral_Huber_single_noise,
            xi_funs[0],
            xi_funs[1],
            y_funs[0],
            y_funs[1],
            args=(q, m, sigma, delta, a),
            epsabs=EPSABS,
            epsrel=EPSREL,
        )[0]

    return integral_value


def q_hat_equation_Huber_single_noise(m, q, sigma, delta, a):
    borders = find_integration_borders_square(
        lambda y, xi: q_integral_Huber_single_noise(y, xi, q, m, sigma, delta, a),
        np.sqrt((1 + delta)),
        1.0,
    )

    args = {"m": m, "q": q, "sigma": sigma, "a": a}
    domain_xi, domain_y = domains_double_line_constraint(
        borders,
        border_plus_Huber,
        border_minus_Huber,
        test_fun_upper_Huber,
        args,
        args,
        args,
    )

    integral_value = 0.0
    for xi_funs, y_funs in zip(domain_xi, domain_y):
        integral_value += dblquad(
            q_integral_Huber_single_noise,
            xi_funs[0],
            xi_funs[1],
            y_funs[0],
            y_funs[1],
            args=(q, m, sigma, delta, a),
            epsabs=EPSABS,
            epsrel=EPSREL,
        )[0]

    return integral_value


def sigma_hat_equation_Huber_single_noise(m, q, sigma, delta, a):
    borders = find_integration_borders_square(
        lambda y, xi: sigma_integral_Huber_single_noise(y, xi, q, m, sigma, delta, a),
        np.sqrt((1 + delta)),
        1.0,
    )

    args = {"m": m, "q": q, "sigma": sigma, "a": a}
    domain_xi, domain_y = domains_double_line_constraint_only_inside(
        borders,
        border_plus_Huber,
        border_minus_Huber,
        test_fun_upper_Huber,
        args,
        args,
        args,
    )

    integral_value = 0.0
    for xi_funs, y_funs in zip(domain_xi, domain_y):
        integral_value += dblquad(
            sigma_integral_Huber_single_noise,
            xi_funs[0],
            xi_funs[1],
            y_funs[0],
            y_funs[1],
            args=(q, m, sigma, delta, a),
            epsabs=EPSABS,
            epsrel=EPSREL,
        )[0]

    return integral_value


# ------------------
# Numerical Loss equations single noise
# ------------------


def hat_equations_numerical_loss_single_noise(
    m, q, sigma, delta, precompute_proximal_func, loss_args
):
    x_range = romberg_linspace(
        -MULT_INTEGRAL * np.sqrt((1 + delta)), MULT_INTEGRAL * np.sqrt((1 + delta))
    )  # np.linspace(borders[0][0], borders[0][1], 2 ** K_ROMBERG + 1)
    y_range = romberg_linspace(
        -MULT_INTEGRAL * np.sqrt((1 + delta)), MULT_INTEGRAL * np.sqrt((1 + delta))
    )

    dx = x_range[1] - x_range[0]
    dy = y_range[1] - y_range[0]

    Xi, _ = np.meshgrid(x_range, y_range)

    precom_gaussian = precompute_values_double_romb_integration(
        _gaussian_function, x_range, y_range
    )

    precom_proximal, precom_Dproximal = precompute_proximal_func(
        x_range, y_range, m, q, sigma, **loss_args,
    )

    precom_fout = (precom_proximal - np.sqrt(q) * Xi) / sigma
    precom_Dfout = (precom_Dproximal - 1) / sigma

    precom_Zout = precompute_values_double_romb_integration(
        _ZoutBayes_single_noise_erm, x_range, y_range, args=[m, q, sigma, delta]
    )
    precom_foutBayes = precompute_values_double_romb_integration(
        _foutBayes_single_noise_erm, x_range, y_range, args=[m, q, sigma, delta]
    )

    m_hat_values = precom_gaussian * precom_Zout * precom_foutBayes * precom_fout
    q_hat_values = precom_gaussian * precom_Zout * precom_fout * precom_fout
    sigma_hat_values = precom_gaussian * precom_Zout * precom_Dfout

    m_hat_integral = double_romb_integration(m_hat_values, dx, dy)
    q_hat_integral = double_romb_integration(q_hat_values, dx, dy)
    sigma_hat_integral = double_romb_integration(sigma_hat_values, dx, dy)

    return m_hat_integral, q_hat_integral, sigma_hat_integral


# ------------------
# BayesOpt equations double noise
# ------------------


def q_hat_equation_BO_double_noise(m, q, sigma, delta_small, delta_large, eps):
    borders = find_integration_borders_square(
        lambda y, xi: q_integral_BO_double_noise(
            y, xi, q, m, sigma, delta_small, delta_large, eps
        ),
        np.sqrt((1 + max(delta_small, delta_large))),
        1.0,
    )

    args = {"m": m, "q": q, "sigma": sigma}
    domain_xi, domain_y = domains_line_constraint(
        borders, border_BO, test_fun_BO, args, args
    )

    integral_value = 0.0
    for xi_funs, y_funs in zip(domain_xi, domain_y):
        integral_value += dblquad(
            q_integral_BO_double_noise,
            xi_funs[0],
            xi_funs[1],
            y_funs[0],
            y_funs[1],
            args=(q, m, sigma, delta_small, delta_large, eps),
            epsabs=EPSABS,
            epsrel=EPSREL,
        )[0]

    return integral_value

    # return dblquad(
    #     q_integral_BO_double_noise,
    #     -np.inf,
    #     np.inf,
    #     -np.inf,
    #     np.inf,
    #     args=(q, m, sigma, delta_small, delta_large, eps),
    #     epsabs=EPSABS,
    #     epsrel=EPSREL,
    # )[0]


# ------------------
# L2 equations double noise
# ------------------


def m_hat_equation_L2_double_noise(m, q, sigma, delta_small, delta_large, eps):
    borders = find_integration_borders_square(
        lambda y, xi: m_integral_L2_double_noise(
            y, xi, q, m, sigma, delta_small, delta_large, eps
        ),
        np.sqrt((1 + max(delta_small, delta_large))),
        1.0,
    )

    domain_xi, domain_y = divide_integration_borders_grid(borders)

    integral_value = 0.0
    for xi_funs, y_funs in zip(domain_xi, domain_y):
        integral_value += dblquad(
            m_integral_L2_double_noise,
            xi_funs[0],
            xi_funs[1],
            y_funs[0],
            y_funs[1],
            args=(q, m, sigma, delta_small, delta_large, eps),
            epsabs=EPSABS,
            epsrel=EPSREL,
        )[0]

    return integral_value


def q_hat_equation_L2_double_noise(m, q, sigma, delta_small, delta_large, eps):
    borders = find_integration_borders_square(
        lambda y, xi: q_integral_L2_double_noise(
            y, xi, q, m, sigma, delta_small, delta_large, eps
        ),
        np.sqrt((1 + max(delta_small, delta_large))),
        1.0,
    )

    domain_xi, domain_y = divide_integration_borders_grid(borders)

    integral_value = 0.0
    for xi_funs, y_funs in zip(domain_xi, domain_y):
        integral_value += dblquad(
            q_integral_L2_double_noise,
            xi_funs[0],
            xi_funs[1],
            y_funs[0],
            y_funs[1],
            args=(q, m, sigma, delta_small, delta_large, eps),
            epsabs=EPSABS,
            epsrel=EPSREL,
        )[0]

    return integral_value


def sigma_hat_equation_L2_double_noise(m, q, sigma, delta_small, delta_large, eps):
    borders = find_integration_borders_square(
        lambda y, xi: sigma_integral_L2_double_noise(
            y, xi, q, m, sigma, delta_small, delta_large, eps
        ),
        np.sqrt((1 + max(delta_small, delta_large))),
        1.0,
    )

    domain_xi, domain_y = divide_integration_borders_grid(borders)

    integral_value = 0.0
    for xi_funs, y_funs in zip(domain_xi, domain_y):
        integral_value += dblquad(
            sigma_integral_L2_double_noise,
            xi_funs[0],
            xi_funs[1],
            y_funs[0],
            y_funs[1],
            args=(q, m, sigma, delta_small, delta_large, eps),
            epsabs=EPSABS,
            epsrel=EPSREL,
        )[0]

    return integral_value


# ------------------
# L1 equations double noise
# ------------------


def m_hat_equation_L1_double_noise(m, q, sigma, delta_small, delta_large, eps):
    borders = find_integration_borders_square(
        lambda y, xi: m_integral_L1_double_noise(
            y, xi, q, m, sigma, delta_small, delta_large, eps
        ),
        np.sqrt((1 + delta_large)),
        1.0,
    )

    args = {"m": m, "q": q, "sigma": sigma}
    domain_xi, domain_y = domains_double_line_constraint(
        borders, border_plus_L1, border_minus_L1, test_fun_upper_L1, args, args, args,
    )

    integral_value = 0.0
    for xi_funs, y_funs in zip(domain_xi, domain_y):
        integral_value += dblquad(
            m_integral_L1_double_noise,
            xi_funs[0],
            xi_funs[1],
            y_funs[0],
            y_funs[1],
            args=(q, m, sigma, delta_small, delta_large, eps),
            epsabs=EPSABS,
            epsrel=EPSREL,
        )[0]

    return integral_value


def q_hat_equation_L1_double_noise(m, q, sigma, delta_small, delta_large, eps):
    borders = find_integration_borders_square(
        lambda y, xi: q_integral_L1_double_noise(
            y, xi, q, m, sigma, delta_small, delta_large, eps
        ),
        np.sqrt((1 + delta_large)),
        1.0,
    )

    args = {"m": m, "q": q, "sigma": sigma}
    domain_xi, domain_y = domains_double_line_constraint(
        borders, border_plus_L1, border_minus_L1, test_fun_upper_L1, args, args, args,
    )

    integral_value = 0.0
    for xi_funs, y_funs in zip(domain_xi, domain_y):
        integral_value += dblquad(
            q_integral_L1_double_noise,
            xi_funs[0],
            xi_funs[1],
            y_funs[0],
            y_funs[1],
            args=(q, m, sigma, delta_small, delta_large, eps),
            epsabs=EPSABS,
            epsrel=EPSREL,
        )[0]

    return integral_value


def sigma_hat_equation_L1_double_noise(m, q, sigma, delta_small, delta_large, eps):
    borders = find_integration_borders_square(
        lambda y, xi: q_integral_L1_double_noise(
            y, xi, q, m, sigma, delta_small, delta_large, eps
        ),
        np.sqrt((1 + delta_large)),
        1.0,
    )

    args = {"m": m, "q": q, "sigma": sigma}
    domain_xi, domain_y = domains_double_line_constraint(
        borders, border_plus_L1, border_minus_L1, test_fun_upper_L1, args, args, args,
    )

    integral_value = 0.0
    for xi_funs, y_funs in zip(domain_xi, domain_y):
        integral_value += dblquad(
            sigma_integral_L1_double_noise,
            xi_funs[0],
            xi_funs[1],
            y_funs[0],
            y_funs[1],
            args=(q, m, sigma, delta_small, delta_large, eps),
            epsabs=EPSABS,
            epsrel=EPSREL,
        )[0]

    return integral_value


# ------------------
#  Huber equations double noise
# ------------------


def m_hat_equation_Huber_double_noise(m, q, sigma, delta_small, delta_large, eps, a):
    borders = find_integration_borders_square(
        lambda y, xi: m_integral_Huber_double_noise(
            y, xi, q, m, sigma, delta_small, delta_large, eps, a
        ),
        np.sqrt((1 + max(delta_small, delta_large))),
        1.0,
    )

    args = {"m": m, "q": q, "sigma": sigma, "a": a}
    domain_xi, domain_y = domains_double_line_constraint(
        borders,
        border_plus_Huber,
        border_minus_Huber,
        test_fun_upper_Huber,
        args,
        args,
        args,
    )

    integral_value = 0.0
    for xi_funs, y_funs in zip(domain_xi, domain_y):
        integral_value += dblquad(
            m_integral_Huber_double_noise,
            xi_funs[0],
            xi_funs[1],
            y_funs[0],
            y_funs[1],
            args=(q, m, sigma, delta_small, delta_large, eps, a),
            epsabs=EPSABS,
            epsrel=EPSREL,
        )[0]

    return integral_value


def q_hat_equation_Huber_double_noise(m, q, sigma, delta_small, delta_large, eps, a):
    borders = find_integration_borders_square(
        lambda y, xi: q_integral_Huber_double_noise(
            y, xi, q, m, sigma, delta_small, delta_large, eps, a
        ),
        np.sqrt((1 + max(delta_small, delta_large))),
        1.0,
    )

    args = {"m": m, "q": q, "sigma": sigma, "a": a}
    domain_xi, domain_y = domains_double_line_constraint(
        borders,
        border_plus_Huber,
        border_minus_Huber,
        test_fun_upper_Huber,
        args,
        args,
        args,
    )

    integral_value = 0.0
    for xi_funs, y_funs in zip(domain_xi, domain_y):
        integral_value += dblquad(
            q_integral_Huber_double_noise,
            xi_funs[0],
            xi_funs[1],
            y_funs[0],
            y_funs[1],
            args=(q, m, sigma, delta_small, delta_large, eps, a),
            epsabs=EPSABS,
            epsrel=EPSREL,
        )[0]

    return integral_value


def sigma_hat_equation_Huber_double_noise(m, q, sigma, delta_small, delta_large, eps, a):
    borders = find_integration_borders_square(
        lambda y, xi: sigma_integral_Huber_double_noise(
            y, xi, q, m, sigma, delta_small, delta_large, eps, a
        ),
        np.sqrt((1 + max(delta_small, delta_large))),
        1.0,
    )

    args = {"m": m, "q": q, "sigma": sigma, "a": a}
    domain_xi, domain_y = domains_double_line_constraint_only_inside(
        borders,
        border_plus_Huber,
        border_minus_Huber,
        test_fun_upper_Huber,
        args,
        args,
        args,
    )

    integral_value = 0.0
    for xi_funs, y_funs in zip(domain_xi, domain_y):
        integral_value += dblquad(
            sigma_integral_Huber_double_noise,
            xi_funs[0],
            xi_funs[1],
            y_funs[0],
            y_funs[1],
            args=(q, m, sigma, delta_small, delta_large, eps, a),
            epsabs=EPSABS,
            epsrel=EPSREL,
        )[0]

    return integral_value


# ------------------
# BayesOpt equations decorrelated noise
# ------------------


def q_hat_equation_BO_decorrelated_noise(
    m, q, sigma, delta_small, delta_large, eps, beta
):
    borders = find_integration_borders_square(
        lambda y, xi: q_integral_BO_decorrelated_noise(
            y, xi, q, m, sigma, delta_small, delta_large, eps, beta
        ),
        np.sqrt((1 + max(delta_small, delta_large))),
        1.0,
    )

    # print(borders[0][0], m,q,sigma, " - ", delta_small, delta_large, eps, beta)

    # args = {"m": m, "q": q, "sigma": sigma}
    # domain_xi, domain_y = domains_line_constraint(
    #     borders, border_BO, test_fun_BO, args, args
    # )
    if delta_large <= 0.11 * delta_small:
        domain_xi, domain_y = divide_integration_borders_multiple_grid(borders, N=10)
    elif delta_large <= 0.5 * delta_small:
        domain_xi, domain_y = divide_integration_borders_multiple_grid(borders, N=7)
    elif delta_large <= delta_small:
        domain_xi, domain_y = divide_integration_borders_multiple_grid(borders, N=5)
    else: # delta_large <= delta_small:
        domain_xi, domain_y = divide_integration_borders_multiple_grid(borders, N=3)
    # else:
    #     args = {"m": m, "q": q, "sigma": sigma}
    #     domain_xi, domain_y = domains_line_constraint(
    #         borders, border_BO, test_fun_BO, args, args
    #     )

    # print(len(domain_xi))

    integral_value = 0.0
    for xi_funs, y_funs in zip(domain_xi, domain_y):
        integral_value += dblquad(
            q_integral_BO_decorrelated_noise,
            xi_funs[0],
            xi_funs[1],
            y_funs[0],
            y_funs[1],
            args=(q, m, sigma, delta_small, delta_large, eps, beta),
        )[0]

    return integral_value

    # return dblquad(
    #     q_integral_BO_decorrelated_noise,
    #     borders[0][0],
    #     borders[0][1],
    #     borders[1][0],
    #     borders[1][1],
    #     args=(q, m, sigma, delta_small, delta_large, eps, beta),
    #     epsabs=EPSABS,
    #     epsrel=EPSREL,
    # )[0]


# ------------------
# L2 equations decorrelated noise
# ------------------


def m_hat_equation_L2_decorrelated_noise(
    m, q, sigma, delta_small, delta_large, eps, beta
):
    borders = find_integration_borders_square(
        lambda y, xi: m_integral_L2_double_noise(
            y, xi, q, m, sigma, delta_small, delta_large, eps, beta
        ),
        np.sqrt((1 + max(delta_small, delta_large))),
        1.0,
    )

    domain_xi, domain_y = divide_integration_borders_grid(borders)

    integral_value = 0.0
    for xi_funs, y_funs in zip(domain_xi, domain_y):
        integral_value += dblquad(
            m_integral_L2_double_noise,
            xi_funs[0],
            xi_funs[1],
            y_funs[0],
            y_funs[1],
            args=(q, m, sigma, delta_small, delta_large, eps, beta),
        )[0]

    return integral_value


def q_hat_equation_L2_decorrelated_noise(
    m, q, sigma, delta_small, delta_large, eps, beta
):
    borders = find_integration_borders_square(
        lambda y, xi: q_integral_L2_double_noise(
            y, xi, q, m, sigma, delta_small, delta_large, eps, beta
        ),
        np.sqrt((1 + max(delta_small, delta_large))),
        1.0,
    )

    domain_xi, domain_y = divide_integration_borders_grid(borders)

    integral_value = 0.0
    for xi_funs, y_funs in zip(domain_xi, domain_y):
        integral_value += dblquad(
            q_integral_L2_double_noise,
            xi_funs[0],
            xi_funs[1],
            y_funs[0],
            y_funs[1],
            args=(q, m, sigma, delta_small, delta_large, eps, beta),
        )[0]

    return integral_value


def sigma_hat_equation_L2_decorrelated_noise(
    m, q, sigma, delta_small, delta_large, eps, beta
):
    borders = find_integration_borders_square(
        lambda y, xi: sigma_integral_L2_double_noise(
            y, xi, q, m, sigma, delta_small, delta_large, eps, beta
        ),
        np.sqrt((1 + max(delta_small, delta_large))),
        1.0,
    )

    domain_xi, domain_y = divide_integration_borders_grid(borders)

    integral_value = 0.0
    for xi_funs, y_funs in zip(domain_xi, domain_y):
        integral_value += dblquad(
            sigma_integral_L2_double_noise,
            xi_funs[0],
            xi_funs[1],
            y_funs[0],
            y_funs[1],
            args=(q, m, sigma, delta_small, delta_large, eps, beta),
        )[0]

    return integral_value


# ------------------
# L1 equations decorrelated noise
# ------------------


def m_hat_equation_L1_decorrelated_noise(
    m, q, sigma, delta_small, delta_large, eps, beta
):
    borders = find_integration_borders_square(
        lambda y, xi: m_integral_L1_decorrelated_noise(
            y, xi, q, m, sigma, delta_small, delta_large, eps, beta
        ),
        np.sqrt((1 + delta_large)),
        1.0,
    )

    args = {"m": m, "q": q, "sigma": sigma}
    domain_xi, domain_y = domains_double_line_constraint(
        borders, border_plus_L1, border_minus_L1, test_fun_upper_L1, args, args, args,
    )

    integral_value = 0.0
    for xi_funs, y_funs in zip(domain_xi, domain_y):
        integral_value += dblquad(
            m_integral_L1_decorrelated_noise,
            xi_funs[0],
            xi_funs[1],
            y_funs[0],
            y_funs[1],
            args=(q, m, sigma, delta_small, delta_large, eps, beta),
            epsabs=EPSABS,
            epsrel=EPSREL,
        )[0]

    return integral_value


def q_hat_equation_L1_decorrelated_noise(
    m, q, sigma, delta_small, delta_large, eps, beta
):
    borders = find_integration_borders_square(
        lambda y, xi: q_integral_L1_decorrelated_noise(
            y, xi, q, m, sigma, delta_small, delta_large, eps, beta
        ),
        np.sqrt((1 + delta_large)),
        1.0,
    )

    args = {"m": m, "q": q, "sigma": sigma}
    domain_xi, domain_y = domains_double_line_constraint(
        borders, border_plus_L1, border_minus_L1, test_fun_upper_L1, args, args, args,
    )

    integral_value = 0.0
    for xi_funs, y_funs in zip(domain_xi, domain_y):
        integral_value += dblquad(
            q_integral_L1_decorrelated_noise,
            xi_funs[0],
            xi_funs[1],
            y_funs[0],
            y_funs[1],
            args=(q, m, sigma, delta_small, delta_large, eps, beta),
            epsabs=EPSABS,
            epsrel=EPSREL,
        )[0]

    return integral_value


def sigma_hat_equation_L1_decorrelated_noise(
    m, q, sigma, delta_small, delta_large, eps, beta
):
    borders = find_integration_borders_square(
        lambda y, xi: q_integral_L1_decorrelated_noise(
            y, xi, q, m, sigma, delta_small, delta_large, eps, beta
        ),
        np.sqrt((1 + delta_large)),
        1.0,
    )

    args = {"m": m, "q": q, "sigma": sigma}
    domain_xi, domain_y = domains_double_line_constraint(
        borders, border_plus_L1, border_minus_L1, test_fun_upper_L1, args, args, args,
    )

    integral_value = 0.0
    for xi_funs, y_funs in zip(domain_xi, domain_y):
        integral_value += dblquad(
            sigma_integral_L1_decorrelated_noise,
            xi_funs[0],
            xi_funs[1],
            y_funs[0],
            y_funs[1],
            args=(q, m, sigma, delta_small, delta_large, eps, beta),
            epsabs=EPSABS,
            epsrel=EPSREL,
        )[0]

    return integral_value


# ------------------
#  Huber equations decorrelated noise
# ------------------


def m_hat_equation_Huber_decorrelated_noise(
    m, q, sigma, delta_small, delta_large, eps, beta, a
):
    borders = find_integration_borders_square(
        lambda y, xi: m_integral_Huber_decorrelated_noise(
            y, xi, q, m, sigma, delta_small, delta_large, eps, beta, a
        ),
        np.sqrt((1 + max(delta_small, delta_large))),
        1.0,
    )

    args = {"m": m, "q": q, "sigma": sigma, "a": a}
    domain_xi, domain_y = domains_double_line_constraint(
        borders,
        border_plus_Huber,
        border_minus_Huber,
        test_fun_upper_Huber,
        args,
        args,
        args,
    )

    integral_value = 0.0
    for xi_funs, y_funs in zip(domain_xi, domain_y):
        integral_value += dblquad(
            m_integral_Huber_decorrelated_noise,
            xi_funs[0],
            xi_funs[1],
            y_funs[0],
            y_funs[1],
            args=(q, m, sigma, delta_small, delta_large, eps, beta, a),
        )[0]

    return integral_value


def q_hat_equation_Huber_decorrelated_noise(
    m, q, sigma, delta_small, delta_large, eps, beta, a
):
    borders = find_integration_borders_square(
        lambda y, xi: q_integral_Huber_decorrelated_noise(
            y, xi, q, m, sigma, delta_small, delta_large, eps, beta, a
        ),
        np.sqrt((1 + max(delta_small, delta_large))),
        1.0,
    )

    args = {"m": m, "q": q, "sigma": sigma, "a": a}
    domain_xi, domain_y = domains_double_line_constraint(
        borders,
        border_plus_Huber,
        border_minus_Huber,
        test_fun_upper_Huber,
        args,
        args,
        args,
    )

    integral_value = 0.0
    for xi_funs, y_funs in zip(domain_xi, domain_y):
        integral_value += dblquad(
            q_integral_Huber_decorrelated_noise,
            xi_funs[0],
            xi_funs[1],
            y_funs[0],
            y_funs[1],
            args=(q, m, sigma, delta_small, delta_large, eps, beta, a),
        )[0]

    return integral_value


def sigma_hat_equation_Huber_decorrelated_noise(
    m, q, sigma, delta_small, delta_large, eps, beta, a
):
    borders = find_integration_borders_square(
        lambda y, xi: sigma_integral_Huber_decorrelated_noise(
            y, xi, q, m, sigma, delta_small, delta_large, eps, beta, a
        ),
        np.sqrt((1 + max(delta_small, delta_large))),
        1.0,
    )

    args = {"m": m, "q": q, "sigma": sigma, "a": a}
    domain_xi, domain_y = domains_double_line_constraint(
        borders,
        border_plus_Huber,
        border_minus_Huber,
        test_fun_upper_Huber,
        args,
        args,
        args,
    )

    integral_value = 0.0
    for xi_funs, y_funs in zip(domain_xi, domain_y):
        integral_value += dblquad(
            sigma_integral_Huber_decorrelated_noise,
            xi_funs[0],
            xi_funs[1],
            y_funs[0],
            y_funs[1],
            args=(q, m, sigma, delta_small, delta_large, eps, beta, a),
        )[0]

    return integral_value
