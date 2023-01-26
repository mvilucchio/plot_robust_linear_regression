import src.numerical_functions as numfun
from numba import njit
import numpy as np
from math import sqrt


def observables_L2(alpha, reg_param, delta_small, delta_large, percentage, beta, a):
    delta_eff = (1 - percentage) * delta_small + percentage * delta_large
    t = sqrt((alpha + reg_param - 1) ** 2 + 4 * reg_param)
    Gamma = 1 + percentage * (beta - 1)
    Lambda = 1 + delta_eff + percentage * (beta**2 - 1)

    sigma = (t - reg_param - alpha + 1) / (2 * reg_param)
    sigma_hat = 2 * reg_param * alpha / (t + reg_param - alpha + 1)
    m = 2 * alpha * Gamma / (t + reg_param + alpha + 1)
    m_hat = 2 * alpha * Gamma * reg_param / (t + reg_param - alpha + 1)
    q = (
        4
        * alpha
        * (alpha * Gamma**2 * (alpha + reg_param + t - 3) + Lambda * (alpha + reg_param + t + 1))
    ) / (
        (t + reg_param + alpha + 1) * ((alpha + reg_param) ** 2)
        - 2 * alpha
        + 2 * reg_param
        + t**2
        + 2 * t * (alpha + reg_param + 1)
        + 1
    )
    q_hat = (
        4
        * alpha
        * reg_param**2
        * (Lambda * (alpha + reg_param + t + 1) ** 2 - 4 * alpha * Gamma**2 * (reg_param + t + 1))
    ) / (
        (t + reg_param - alpha + 1) ** 2 * ((alpha + reg_param) ** 2)
        - 2 * alpha
        + 2 * reg_param
        + t**2
        + 2 * t * (alpha + reg_param + 1)
        + 1
    )

    return m, q, sigma, m_hat, q_hat, sigma_hat


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
def var_hat_func_L2_single_noise(m, q, sigma, alpha, delta):
    m_hat = alpha / (1 + sigma)
    q_hat = alpha * (1 + q + delta - 2 * np.abs(m)) / ((1 + sigma) ** 2)
    sigma_hat = alpha / (1 + sigma)
    return m_hat, q_hat, sigma_hat


def var_hat_func_L2_num_single_noise(m, q, sigma, alpha, delta):
    m_hat = alpha * numfun.m_hat_equation_L2_single_noise(m, q, sigma, delta)
    q_hat = alpha * numfun.q_hat_equation_L2_single_noise(m, q, sigma, delta)
    sigma_hat = -alpha * numfun.sigma_hat_equation_L2_single_noise(m, q, sigma, delta)
    return m_hat, q_hat, sigma_hat


@njit(error_model="numpy", fastmath=True)
def var_hat_func_L2_double_noise(m, q, sigma, alpha, delta_small, delta_large, percentage):
    delta_eff = (1 - percentage) * delta_small + percentage * delta_large
    m_hat = alpha / (1 + sigma)
    q_hat = alpha * (1 + q + delta_eff - 2 * np.abs(m)) / ((1 + sigma) ** 2)
    sigma_hat = alpha / (1 + sigma)
    return m_hat, q_hat, sigma_hat


def var_hat_func_L2_num_double_noise(m, q, sigma, alpha, delta_small, delta_large, percentage):
    m_hat = alpha * numfun.m_hat_equation_L2_double_noise(
        m, q, sigma, delta_small, delta_large, percentage
    )
    q_hat = alpha * numfun.q_hat_equation_L2_double_noise(
        m, q, sigma, delta_small, delta_large, percentage
    )
    sigma_hat = -alpha * numfun.sigma_hat_equation_L2_double_noise(
        m, q, sigma, delta_small, delta_large, percentage
    )
    return m_hat, q_hat, sigma_hat


@njit(error_model="numpy", fastmath=True)
def var_hat_func_L2_decorrelated_noise(
    m, q, sigma, alpha, delta_small, delta_large, percentage, beta
):
    delta_eff = (1 - percentage) * delta_small + percentage * delta_large
    intermediate_val = 1 + percentage * (beta - 1)

    m_hat = alpha * intermediate_val / (1 + sigma)
    q_hat = (
        alpha
        * (1 + q + delta_eff + percentage * (beta**2 - 1) - 2 * np.abs(m) * intermediate_val)
        / ((1 + sigma) ** 2)
    )
    sigma_hat = alpha / (1 + sigma)
    return m_hat, q_hat, sigma_hat


def var_hat_func_L2_num_decorrelated_noise(
    m, q, sigma, alpha, delta_small, delta_large, percentage, beta
):
    m_hat = alpha * numfun.m_hat_equation_L2_decorrelated_noise(
        m, q, sigma, delta_small, delta_large, percentage, beta
    )
    q_hat = alpha * numfun.q_hat_equation_L2_decorrelated_noise(
        m, q, sigma, delta_small, delta_large, percentage, beta
    )
    sigma_hat = -alpha * numfun.sigma_hat_equation_L2_decorrelated_noise(
        m, q, sigma, delta_small, delta_large, percentage, beta
    )
    return m_hat, q_hat, sigma_hat
