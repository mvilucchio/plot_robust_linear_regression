import src.numerical_functions as numfun
from numba import njit
import numpy as np
from math import erf, erfc


def var_hat_func_numerical_loss_single_noise(
    m, q, sigma, alpha, delta, precompute_proximal_func, loss_args
):
    m_int, q_int, sigma_int = numfun.hat_equations_numerical_loss_single_noise(
        m, q, sigma, delta, precompute_proximal_func, loss_args
    )
    m_hat = alpha * m_int
    q_hat = alpha * q_int
    sigma_hat = -alpha * sigma_int
    return m_hat, q_hat, sigma_hat


def var_hat_func_numerical_loss_double_noise(
    m,
    q,
    sigma,
    alpha,
    delta_small,
    delta_large,
    percentage,
    loss_derivative,
    loss_second_derivative,
    loss_args=None,
):
    m_int, q_int, sigma_int = numfun.hat_equations_numerical_loss_double_noise(
        m,
        q,
        sigma,
        delta_small,
        delta_large,
        percentage,
        loss_derivative,
        loss_second_derivative,
        loss_args=loss_args,
    )
    m_hat = alpha * m_int
    q_hat = alpha * q_int
    sigma_hat = -alpha * sigma_int
    return m_hat, q_hat, sigma_hat


def var_hat_func_numerical_loss_decorrelated_noise(
    m,
    q,
    sigma,
    alpha,
    delta_small,
    delta_large,
    percentage,
    beta,
    loss_derivative,
    loss_second_derivative,
    loss_args=None,
):
    m_int, q_int, sigma_int = numfun.hat_equations_numerical_loss_decorrelated_noise(
        m,
        q,
        sigma,
        delta_small,
        delta_large,
        percentage,
        beta,
        loss_derivative,
        loss_second_derivative,
        loss_args=loss_args,
    )
    m_hat = alpha * m_int
    q_hat = alpha * q_int
    sigma_hat = -alpha * sigma_int
    return m_hat, q_hat, sigma_hat
