import src.numerical_functions as numfun
from numba import njit
import numpy as np
from math import erf, erfc, exp, log, sqrt


@njit(error_model="numpy", fastmath=True)
def var_func_L2(
    m_hat,
    q_hat,
    sigma_hat,
    reg_param,
):
    m = m_hat / (sigma_hat + reg_param)
    q = (m_hat**2 + q_hat) / (sigma_hat + reg_param) ** 2
    sigma = 1.0 / (sigma_hat + reg_param)
    return m, q, sigma


@njit(error_model="numpy", fastmath=True)
def var_hat_func_L1_single_noise(m, q, sigma, alpha, delta):
    sqrt_arg = 1 + q + delta - 2 * m
    erf_arg = sigma / np.sqrt(2 * sqrt_arg)

    m_hat = (alpha / sigma) * erf(erf_arg)
    q_hat = (alpha / sigma**2) * (
        sqrt_arg * erf(erf_arg)
        + sigma**2 * erfc(erf_arg)
        - sigma * np.sqrt(2 / np.pi) * np.sqrt(sqrt_arg) * np.exp(-(erf_arg**2))
    )
    sigma_hat = (alpha / sigma) * erf(erf_arg)
    return m_hat, q_hat, sigma_hat


def var_hat_func_L1_num_single_noise(m, q, sigma, alpha, delta):
    m_hat = alpha * numfun.m_hat_equation_L1_single_noise(m, q, sigma, delta)
    q_hat = alpha * numfun.q_hat_equation_L1_single_noise(m, q, sigma, delta)
    sigma_hat = -alpha * numfun.sigma_hat_equation_L1_single_noise(m, q, sigma, delta)
    return m_hat, q_hat, sigma_hat


@njit(error_model="numpy", fastmath=True)
def var_hat_func_L1_double_noise(m, q, sigma, alpha, delta_small, delta_large, percentage):
    small_sqrt = delta_small - 2 * m + q + 1
    large_sqrt = delta_large - 2 * m + q + 1
    small_exp = -(sigma**2) / (2 * small_sqrt)
    large_exp = -(sigma**2) / (2 * large_sqrt)
    small_erf = sigma / np.sqrt(2 * small_sqrt)
    large_erf = sigma / np.sqrt(2 * large_sqrt)

    m_hat = (alpha / sigma) * ((1 - percentage) * erf(small_erf) + percentage * erf(large_erf))
    q_hat = alpha * (
        (1 - percentage) * erfc(small_erf) + percentage * erfc(large_erf)
    ) + alpha / sigma**2 * (
        (
            (1 - percentage) * (small_sqrt) * erf(small_erf)
            + percentage * (large_sqrt) * erf(large_erf)
        )
        - np.exp(
            np.log(sigma)
            + 0.5 * np.log(2)
            - 0.5 * np.log(np.pi)
            + 0.5 * np.log(large_sqrt)
            + np.log(
                (1 - percentage) * np.sqrt(small_sqrt / large_sqrt) * np.exp(small_exp)
                + percentage * np.exp(large_exp)
            )
        )
    )
    sigma_hat = (alpha / sigma) * ((1 - percentage) * erf(small_erf) + percentage * erf(large_erf))
    return m_hat, q_hat, sigma_hat


def var_hat_func_L1_num_double_noise(m, q, sigma, alpha, delta_small, delta_large, percentage):
    m_hat = alpha * numfun.m_hat_equation_L1_double_noise(
        m, q, sigma, delta_small, delta_large, percentage
    )
    q_hat = alpha * numfun.q_hat_equation_L1_double_noise(
        m, q, sigma, delta_small, delta_large, percentage
    )
    sigma_hat = -alpha * numfun.sigma_hat_equation_L1_double_noise(
        m, q, sigma, delta_small, delta_large, percentage
    )
    return m_hat, q_hat, sigma_hat


@njit(error_model="numpy", fastmath=True)
def var_hat_func_L1_decorrelated_noise(
    m, q, sigma, alpha, delta_small, delta_large, percentage, beta
):
    # print(type(m), type(q), type(sigma), type(alpha), type(delta_small), type(delta_large), type(percentage), type(beta))
    small_sqrt = delta_small - 2 * m + q + 1
    large_sqrt = delta_large - 2 * m * beta + q + beta**2
    small_exp = -(sigma**2) / (2 * small_sqrt)
    large_exp = -(sigma**2) / (2 * large_sqrt)
    small_erf = sigma / sqrt(2 * small_sqrt)
    large_erf = sigma / sqrt(2 * large_sqrt)

    # print(type(small_sqrt), type(large_sqrt), type(small_exp), type(large_exp), type(small_erf), type(large_erf))
    # print(len(small_sqrt), len(large_sqrt), len(small_exp), len(large_exp), len(small_erf), len(large_erf))

    m_hat = (alpha / sigma) * (
        (1 - percentage) * erf(small_erf) + beta * percentage * erf(large_erf)
    )
    q_hat = alpha * (
        (1 - percentage) * erfc(small_erf) + percentage * erfc(large_erf)
    ) + alpha / sigma**2 * (
        (
            (1 - percentage) * (small_sqrt) * erf(small_erf)
            + percentage * (large_sqrt) * erf(large_erf)
        )
        - exp(
            log(sigma)
            + 0.5 * log(2)
            - 0.5 * log(np.pi)
            + log(
                (1 - percentage) * sqrt(small_sqrt) * exp(small_exp)
                + percentage * sqrt(large_sqrt) * exp(large_exp)
            )
        )
    )
    sigma_hat = (alpha / sigma) * ((1 - percentage) * erf(small_erf) + percentage * erf(large_erf))
    return m_hat, q_hat, sigma_hat


def var_hat_func_L1_num_decorrelated_noise(
    m, q, sigma, alpha, delta_small, delta_large, percentage, beta
):
    m_hat = alpha * numfun.m_hat_equation_L1_decorrelated_noise(
        m, q, sigma, delta_small, delta_large, percentage, beta
    )
    q_hat = alpha * numfun.q_hat_equation_L1_decorrelated_noise(
        m, q, sigma, delta_small, delta_large, percentage, beta
    )
    sigma_hat = -alpha * numfun.sigma_hat_equation_L1_decorrelated_noise(
        m, q, sigma, delta_small, delta_large, percentage, beta
    )
    return m_hat, q_hat, sigma_hat
