import src.numerical_functions as numfun
from numba import njit
import numpy as np
from math import erf, erfc


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
def var_hat_func_Huber_single_noise(m, q, sigma, alpha, delta, a):
    arg_sqrt = 1 + q + delta - 2 * m
    erf_arg = (a * (sigma + 1)) / np.sqrt(2 * arg_sqrt)

    m_hat = (alpha / (1 + sigma)) * erf(erf_arg)
    q_hat = (alpha / (1 + sigma) ** 2) * (
        arg_sqrt * erf(erf_arg)
        + a**2 * (1 + sigma) ** 2 * erfc(erf_arg)
        - a * (1 + sigma) * np.sqrt(2 / np.pi) * np.sqrt(arg_sqrt) * np.exp(-(erf_arg**2))
    )
    sigma_hat = (alpha / (1 + sigma)) * erf(erf_arg)
    return m_hat, q_hat, sigma_hat


def var_hat_func_Huber_num_single_noise(m, q, sigma, alpha, delta, a):
    m_hat = alpha * numfun.m_hat_equation_Huber_single_noise(m, q, sigma, delta, a)
    q_hat = alpha * numfun.q_hat_equation_Huber_single_noise(m, q, sigma, delta, a)
    sigma_hat = -alpha * numfun.sigma_hat_equation_Huber_single_noise(m, q, sigma, delta, a)
    return m_hat, q_hat, sigma_hat


@njit(error_model="numpy", fastmath=True)
def var_hat_func_Huber_double_noise(m, q, sigma, alpha, delta_small, delta_large, percentage, a):
    small_sqrt = delta_small - 2 * m + q + 1
    large_sqrt = delta_large - 2 * m + q + 1
    small_erf = (a * (sigma + 1)) / np.sqrt(2 * small_sqrt)
    large_erf = (a * (sigma + 1)) / np.sqrt(2 * large_sqrt)

    m_hat = (alpha / (1 + sigma)) * (
        (1 - percentage) * erf(small_erf) + percentage * erf(large_erf)
    )
    q_hat = alpha * (
        a**2
        - (np.sqrt(2 / np.pi) * a / (1 + sigma))
        * (
            (1 - percentage) * np.sqrt(small_sqrt) * np.exp(-(small_erf**2))
            + percentage * np.sqrt(large_sqrt) * np.exp(-(large_erf**2))
        )
        + (1 / (1 + sigma) ** 2)
        * (
            (1 - percentage) * (small_sqrt - (a * (1 + sigma)) ** 2) * erf(small_erf)
            + percentage * (large_sqrt - (a * (1 + sigma)) ** 2) * erf(large_erf)
        )
    )
    sigma_hat = (alpha / (1 + sigma)) * (
        (1 - percentage) * erf(small_erf) + percentage * erf(large_erf)
    )
    return m_hat, q_hat, sigma_hat


def var_hat_func_Huber_num_double_noise(
    m, q, sigma, alpha, delta_small, delta_large, percentage, a
):
    m_hat = alpha * numfun.m_hat_equation_Huber_double_noise(
        m,
        q,
        sigma,
        delta_small,
        delta_large,
        percentage,
        a,
    )
    q_hat = alpha * numfun.q_hat_equation_Huber_double_noise(
        m,
        q,
        sigma,
        delta_small,
        delta_large,
        percentage,
        a,
    )
    sigma_hat = -alpha * numfun.sigma_hat_equation_Huber_double_noise(
        m,
        q,
        sigma,
        delta_small,
        delta_large,
        percentage,
        a,
    )
    return m_hat, q_hat, sigma_hat


@njit(error_model="numpy", fastmath=True)
def var_hat_func_Huber_decorrelated_noise(
    m, q, sigma, alpha, delta_small, delta_large, percentage, beta, a
):
    small_sqrt = delta_small - 2 * m + q + 1
    large_sqrt = delta_large - 2 * m * beta + q + beta**2
    small_erf = (a * (sigma + 1)) / np.sqrt(2 * small_sqrt)
    large_erf = (a * (sigma + 1)) / np.sqrt(2 * large_sqrt)

    m_hat = (alpha / (1 + sigma)) * (
        (1 - percentage) * erf(small_erf) + beta * percentage * erf(large_erf)
    )
    q_hat = alpha * (
        a**2
        - (np.sqrt(2 / np.pi) * a / (1 + sigma))
        * (
            (1 - percentage) * np.sqrt(small_sqrt) * np.exp(-(small_erf**2))
            + percentage * np.sqrt(large_sqrt) * np.exp(-(large_erf**2))
        )
        + (1 / (1 + sigma) ** 2)
        * (
            (1 - percentage) * (small_sqrt - (a * (1 + sigma)) ** 2) * erf(small_erf)
            + percentage * (large_sqrt - (a * (1 + sigma)) ** 2) * erf(large_erf)
        )
    )
    sigma_hat = (alpha / (1 + sigma)) * (
        (1 - percentage) * erf(small_erf) + percentage * erf(large_erf)
    )
    return m_hat, q_hat, sigma_hat


def var_hat_func_Huber_num_decorrelated_noise(
    m, q, sigma, alpha, delta_small, delta_large, percentage, beta, a
):
    m_hat = alpha * numfun.m_hat_equation_Huber_decorrelated_noise(
        m,
        q,
        sigma,
        delta_small,
        delta_large,
        percentage,
        beta,
        a,
    )
    q_hat = alpha * numfun.q_hat_equation_Huber_decorrelated_noise(
        m,
        q,
        sigma,
        delta_small,
        delta_large,
        percentage,
        beta,
        a,
    )
    sigma_hat = -alpha * numfun.sigma_hat_equation_Huber_decorrelated_noise(
        m,
        q,
        sigma,
        delta_small,
        delta_large,
        percentage,
        beta,
        a,
    )
    return m_hat, q_hat, sigma_hat
