import numpy as np
from numba import njit
from src.integration_utils import x_ge, w_ge


@njit(error_model="numpy", fastmath=True)
def gaussian_prior(x):
    return np.exp(-0.5 * x ** 2) / np.sqrt(2 * np.pi)


@njit(error_model="numpy")
def input_functions_gaussian_prior(Rs, sigmas):

    fa = np.empty_like(sigmas)
    fv = np.empty_like(sigmas)

    for idx, (sigma, R) in enumerate(zip(sigmas, Rs)):
        z = np.sqrt(2.0) * np.sqrt(sigma) * x_ge + R
        jacobian = np.sqrt(2.0) * np.sqrt(sigma)

        simple_int = np.sum(w_ge * jacobian * gaussian_prior(z))
        fa[idx] = np.sum(w_ge * jacobian * gaussian_prior(z) * z) / (simple_int)

        first_term_fv = (
            np.sum(w_ge * jacobian * z * (z - R) * gaussian_prior(z)) / simple_int
        )
        second_term_fv = (
            fa[idx] * np.sum(w_ge * jacobian * (z - R) * gaussian_prior(z)) / simple_int
        )
        fv[idx] = first_term_fv - second_term_fv

    return fa, fv


@njit(error_model="numpy", fastmath=True)
def likelihood_single_gaussians(y, z, delta):
    return np.exp(-0.5 * (y - z) ** 2 / delta) / (np.sqrt(2 * np.pi * delta))


@njit(error_model="numpy")
def output_functions_single_noise(ys, omegas, Vs, delta):
    # broadcasting

    gout = np.empty_like(ys)
    Dgout = np.empty_like(ys)

    for idx, (y, omega, V) in enumerate(zip(ys, omegas, Vs)):
        z = np.sqrt(2.0) * np.sqrt(V) * x_ge + omega
        jacobian = np.sqrt(2.0) * np.sqrt(V)

        simple_int = np.sum(w_ge * jacobian * likelihood_single_gaussians(y, z, delta))
        gout[idx] = np.sum(
            w_ge * jacobian * likelihood_single_gaussians(y, z, delta) * (z - omega)
        ) / (V * simple_int)

        first_term_Dgout = np.sum(
            w_ge * jacobian * likelihood_single_gaussians(y, z, delta) * (z - omega) ** 2
        ) / (V ** 2 * simple_int)
        Dgout[idx] = first_term_Dgout - 1 / V - gout[idx] ** 2

    return gout, Dgout


@njit(error_model="numpy", fastmath=True)
def likelihood_double_gaussians(y, z, delta_small, delta_large, eps):
    return (1 - eps) / (np.sqrt(2 * np.pi * delta_small)) * np.exp(
        -0.5 * (y - z) ** 2 / (delta_small)
    ) + eps / (np.sqrt(2 * np.pi * delta_large)) * np.exp(
        -0.5 * (y - z) ** 2 / (delta_large)
    )


@njit(error_model="numpy")
def output_functions_double_noise(ys, omegas, Vs, delta_small, delta_large, eps):
    # broadcasting

    gout = np.empty_like(ys)
    Dgout = np.empty_like(ys)

    for idx, (y, omega, V) in enumerate(zip(ys, omegas, Vs)):
        z = np.sqrt(2.0) * np.sqrt(V) * x_ge + omega
        jacobian = np.sqrt(2.0) * np.sqrt(V)

        simple_int = np.sum(
            w_ge
            * jacobian
            * likelihood_double_gaussians(y, z, delta_small, delta_large, eps)
        )
        gout[idx] = np.sum(
            w_ge
            * jacobian
            * likelihood_double_gaussians(y, z, delta_small, delta_large, eps)
            * (z - omega)
        ) / (V * simple_int)

        first_term_Dgout = np.sum(
            w_ge
            * jacobian
            * likelihood_double_gaussians(y, z, delta_small, delta_large, eps)
            * (z - omega) ** 2
        ) / (V ** 2 * simple_int)
        Dgout[idx] = first_term_Dgout - 1 / V - gout[idx] ** 2

    return gout, Dgout


@njit(error_model="numpy", fastmath=True)
def likelihood_decorrelated_gaussians(y, z, delta_small, delta_large, eps, beta):
    return (1 - eps) / (np.sqrt(2 * np.pi * delta_small)) * np.exp(
        -0.5 * (y - z) ** 2 / (delta_small)
    ) + eps / (np.sqrt(2 * np.pi * delta_large)) * np.exp(
        -0.5 * (y - beta * z) ** 2 / (delta_large)
    )


@njit(error_model="numpy")
def output_functions_decorrelated_noise(
    ys, omegas, Vs, delta_small, delta_large, eps, beta
):
    # broadcasting

    gout = np.empty_like(ys)
    Dgout = np.empty_like(ys)

    for idx, (y, omega, V) in enumerate(zip(ys, omegas, Vs)):
        z = np.sqrt(2.0) * np.sqrt(V) * x_ge + omega
        jacobian = np.sqrt(2.0) * np.sqrt(V)

        simple_int = np.sum(
            w_ge
            * jacobian
            * likelihood_decorrelated_gaussians(y, z, delta_small, delta_large, eps, beta)
        )
        gout[idx] = np.sum(
            w_ge
            * jacobian
            * likelihood_decorrelated_gaussians(y, z, delta_small, delta_large, eps, beta)
            * (z - omega)
        ) / (V * simple_int)

        first_term_Dgout = np.sum(
            w_ge
            * jacobian
            * likelihood_decorrelated_gaussians(y, z, delta_small, delta_large, eps, beta)
            * (z - omega) ** 2
        ) / (V ** 2 * simple_int)
        Dgout[idx] = first_term_Dgout - 1 / V - gout[idx] ** 2

    return gout, Dgout

