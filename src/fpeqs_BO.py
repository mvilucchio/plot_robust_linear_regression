import src.numerical_functions as numfun
from numba import njit


@njit(error_model="numpy", fastmath=True)
def var_func_BO(
    m_hat,
    q_hat,
    sigma_hat,
    reg_param,
):
    q = q_hat / (1 + q_hat)
    return q, q, 1 - q


@njit(error_model="numpy", fastmath=True)
def var_hat_func_BO_single_noise(m, q, sigma, alpha, delta):
    q_hat = alpha / (1 + delta - q)
    return q_hat, q_hat, 1 - q_hat


def var_hat_func_BO_num_single_noise(m, q, sigma, alpha, delta):
    q_hat = alpha * numfun.q_hat_equation_BO_single_noise(m, q, sigma, delta)
    return q_hat, q_hat, 1 - q_hat


def var_hat_func_BO_double_noise(m, q, sigma, alpha, delta_small, delta_large, percentage):
    raise NotImplementedError


def var_hat_func_BO_num_double_noise(m, q, sigma, alpha, delta_small, delta_large, percentage):
    q_hat = alpha * numfun.q_hat_equation_BO_double_noise(
        m, q, sigma, delta_small, delta_large, percentage
    )
    return q_hat, q_hat, 1 - q_hat


def var_hat_func_BO_decorrelated_noise(
    m, q, sigma, alpha, delta_small, delta_large, percentage, beta
):
    raise NotImplementedError


def var_hat_func_BO_num_decorrelated_noise(
    m, q, sigma, alpha, delta_small, delta_large, percentage, beta
):
    q_hat = alpha * numfun.q_hat_equation_BO_decorrelated_noise(
        m, q, sigma, delta_small, delta_large, percentage, beta
    )
    return q_hat, q_hat, 1 - q_hat
