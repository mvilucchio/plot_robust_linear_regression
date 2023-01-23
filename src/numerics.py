import numpy as np
from scipy import optimize
from numbers import Number
from sklearn.utils import axis0_safe_slice
from sklearn.utils.extmath import safe_sparse_dot
from numba import njit
import cvxpy as cp
from src.amp_funcs import (
    input_functions_gaussian_prior,
    output_functions_decorrelated_noise,
    output_functions_double_noise,
    output_functions_single_noise,
)

from multiprocessing import Pool

# from mpi4py.futures import MPIPoolExecutor as Pool

MAX_ITER_MINIMIZE = 1500
GTOL_MINIMIZE = 1e-6

BLEND_GAMP = 0.55
TOL_GAMP = 1e-4


def measure_gen_single(generalization, teacher_vector, xs, delta):
    n_samples, n_features = xs.shape
    w_xs = np.divide(xs @ teacher_vector, np.sqrt(n_features))
    if generalization:
        ys = w_xs
    else:
        error_sample = np.sqrt(delta) * np.random.normal(
            loc=0.0, scale=1.0, size=(n_samples,)
        )
        ys = w_xs + error_sample
    return ys


def measure_gen_double(
    generalization, teacher_vector, xs, delta_small, delta_large, percentage
):
    n_samples, n_features = xs.shape
    w_xs = np.divide(xs @ teacher_vector, np.sqrt(n_features))
    if generalization:
        ys = w_xs
    else:
        choice = np.random.choice(
            [0, 1], p=[1 - percentage, percentage], size=(n_samples,)
        )
        error_sample = np.empty((n_samples, 2))
        error_sample[:, 0] = np.sqrt(delta_small) * np.random.normal(
            loc=0.0, scale=1.0, size=(n_samples,)
        )
        error_sample[:, 1] = np.sqrt(delta_large) * np.random.normal(
            loc=0.0, scale=1.0, size=(n_samples,)
        )
        total_error = np.where(choice, error_sample[:, 1], error_sample[:, 0])
        ys = w_xs + total_error
    return ys


def measure_gen_decorrelated(
    generalization, teacher_vector, xs, delta_small, delta_large, percentage, beta
):
    n_samples, n_features = xs.shape
    w_xs = np.divide(xs @ teacher_vector, np.sqrt(n_features))
    if generalization:
        ys = w_xs
    else:
        choice = np.random.choice(
            [0, 1], p=[1 - percentage, percentage], size=(n_samples,)
        )
        error_sample = np.empty((n_samples, 2))
        error_sample[:, 0] = np.sqrt(delta_small) * np.random.normal(
            loc=0.0, scale=1.0, size=(n_samples,)
        )
        error_sample[:, 1] = np.sqrt(delta_large) * np.random.normal(
            loc=0.0, scale=1.0, size=(n_samples,)
        )
        total_error = np.where(choice, error_sample[:, 1], error_sample[:, 0])
        factor_in_front = np.where(choice, beta, 1.0)
        ys = factor_in_front * w_xs + total_error
    return ys


def data_generation(
    measure_fun, n_features, n_samples, n_generalization, measure_fun_args
):
    theta_0_teacher = np.random.normal(loc=0.0, scale=1.0, size=(n_features,))

    xs = np.random.normal(loc=0.0, scale=1.0, size=(n_samples, n_features))
    xs_gen = np.random.normal(loc=0.0, scale=1.0, size=(n_generalization, n_features))

    ys = measure_fun(False, theta_0_teacher, xs, *measure_fun_args)
    ys_gen = measure_fun(True, theta_0_teacher, xs_gen, *measure_fun_args)

    return xs, ys, xs_gen, ys_gen, theta_0_teacher


def _find_numerical_mean_std(
    alpha,
    measure_fun,
    find_coefficients_fun,
    n_features,
    repetitions,
    measure_fun_args,
    find_coefficients_fun_args,
):
    all_gen_errors = np.empty((repetitions,))

    for idx in range(repetitions):
        xs, ys, _, _, ground_truth_theta = data_generation(
            measure_fun,
            n_features=n_features,
            n_samples=max(int(np.around(n_features * alpha)), 1),
            n_generalization=1,
            measure_fun_args=measure_fun_args,
        )
        
        print(xs.shape, ys.shape)
        estimated_theta = find_coefficients_fun(ys, xs, *find_coefficients_fun_args)

        all_gen_errors[idx] = np.divide(
            np.sum(np.square(ground_truth_theta - estimated_theta)), n_features
        )

        del xs
        del ys
        del ground_truth_theta

    error_mean, error_std = np.mean(all_gen_errors), np.std(all_gen_errors)
    print(alpha, "Done.")

    del all_gen_errors

    return error_mean, error_std


def no_parallel_generate_different_alpha(
    measure_fun,
    find_coefficients_fun,
    alpha_1=0.01,
    alpha_2=100,
    n_features=100,
    n_alpha_points=10,
    repetitions=10,
    measure_fun_args=(),
    find_coefficients_fun_args={},
    alphas=None,
):
    if alphas is None:
        alphas = np.logspace(
            np.log(alpha_1) / np.log(10), np.log(alpha_2) / np.log(10), n_alpha_points
        )
    else:
        n_alpha_points = len(alphas)

    if not isinstance(find_coefficients_fun_args, list):
        find_coefficients_fun_args = [find_coefficients_fun_args] * len(alphas)

    errors_mean = np.empty((n_alpha_points,))
    errors_std = np.empty((n_alpha_points,))

    results = []
    i = 0
    for a, fckw in zip(alphas, find_coefficients_fun_args):
        print(i, "/", len(alphas))
        results.append(
            _find_numerical_mean_std(
                a,
                measure_fun,
                find_coefficients_fun,
                n_features,
                repetitions,
                measure_fun_args,
                fckw,
            )
        )
        i += 1

    for idx, r in enumerate(results):
        errors_mean[idx] = r[0]
        errors_std[idx] = r[1]

    return alphas, errors_mean, errors_std


def generate_different_alpha(
    measure_fun,
    find_coefficients_fun,
    alpha_1=0.01,
    alpha_2=100,
    n_features=100,
    n_alpha_points=10,
    repetitions=10,
    measure_fun_args=(),
    find_coefficients_fun_args=(),
    alphas=None,
):
    if alphas is None:
        alphas = np.logspace(
            np.log(alpha_1) / np.log(10), np.log(alpha_2) / np.log(10), n_alpha_points
        )
    else:
        n_alpha_points = len(alphas)

    if not isinstance(find_coefficients_fun_args, list):
        find_coefficients_fun_args = [find_coefficients_fun_args] * len(alphas)

    errors_mean = np.empty((n_alpha_points,))
    errors_std = np.empty((n_alpha_points,))

    inputs = [
        (
            a,
            measure_fun,
            find_coefficients_fun,
            n_features,
            repetitions,
            measure_fun_args,
            fc_agrs,
        )
        for a, fc_agrs in zip(alphas, find_coefficients_fun_args)
    ]

    with Pool() as pool:
        results = pool.starmap(_find_numerical_mean_std, inputs)

    for idx, r in enumerate(results):
        errors_mean[idx] = r[0]
        errors_std[idx] = r[1]

    return alphas, errors_mean, errors_std


@njit(error_model="numpy", fastmath=True)
def find_coefficients_AMP(input_funs, output_funs, ys, xs, *noise_args):
    _, d = xs.shape

    a_t_1 = 0.1 * np.random.rand(d) + 0.95
    v_t_1 = 0.5 * np.random.rand(d) + 0.01
    gout_t_1 = 0.5 * np.random.rand(1) + 0.001

    F = xs / np.sqrt(d)
    F2 = F ** 2

    err = 1.0
    while err > TOL_GAMP:
        V_t = F2 @ v_t_1
        omega_t = F @ a_t_1 - V_t * gout_t_1

        gout_t, Dgout_t = output_funs(ys, omega_t, V_t, *noise_args)

        sigma_t = -1 / (Dgout_t @ F2)
        R_t = a_t_1 + sigma_t * (gout_t @ F)

        a_t, v_t = input_funs(R_t, sigma_t)

        err = max(np.max(a_t - a_t_1), np.max(v_t - v_t_1))

        a_t_1 = BLEND_GAMP * a_t + (1 - BLEND_GAMP) * a_t_1
        v_t_1 = BLEND_GAMP * v_t + (1 - BLEND_GAMP) * v_t_1
        gout_t_1 = BLEND_GAMP * gout_t + (1 - BLEND_GAMP) * gout_t_1

    return a_t  # , v_t


def find_coefficients_AMP_single_noise(ys, xs, *noise_args):
    return find_coefficients_AMP(
        input_functions_gaussian_prior, output_functions_single_noise, ys, xs, *noise_args
    )


def find_coefficients_AMP_double_noise(ys, xs, *noise_args):
    return find_coefficients_AMP(
        input_functions_gaussian_prior, output_functions_double_noise, ys, xs, *noise_args
    )


def find_coefficients_AMP_decorrelated_noise(ys, xs, *noise_args):
    return find_coefficients_AMP(
        input_functions_gaussian_prior,
        output_functions_decorrelated_noise,
        ys,
        xs,
        *noise_args
    )


@njit(error_model="numpy", fastmath=True)
def find_coefficients_L2(ys, xs, reg_param):
    _, d = xs.shape
    a = np.divide(xs.T.dot(xs), d) + reg_param * np.identity(d)
    b = np.divide(xs.T.dot(ys), np.sqrt(d))
    return np.linalg.solve(a, b)


def find_coefficients_L1(ys, xs, reg_param):
    _, d = xs.shape
    # w = np.random.normal(loc=0.0, scale=1.0, size=(d,))
    xs_norm = np.divide(xs, np.sqrt(d))
    w = cp.Variable(shape=d)
    obj = cp.Minimize(cp.norm(ys - xs_norm @ w, 1) + 0.5 * reg_param * cp.sum_squares(w))
    prob = cp.Problem(obj)
    prob.solve(eps_abs=1e-3)

    return w.value


#  @njit(error_model="numpy", fastmath=True)
def _loss_and_gradient_Huber(w, xs_norm, ys, reg_param, a):
    linear_loss = ys - xs_norm @ w
    abs_linear_loss = np.abs(linear_loss)
    outliers_mask = abs_linear_loss > a

    outliers = abs_linear_loss[outliers_mask]
    num_outliers = np.count_nonzero(outliers_mask)
    n_non_outliers = xs_norm.shape[0] - num_outliers

    loss = a * np.sum(outliers) - 0.5 * num_outliers * a ** 2

    non_outliers = linear_loss[~outliers_mask]
    loss += 0.5 * np.dot(non_outliers, non_outliers)
    loss += 0.5 * reg_param * np.dot(w, w)

    xs_non_outliers = -axis0_safe_slice(xs_norm, ~outliers_mask, n_non_outliers)
    gradient = safe_sparse_dot(non_outliers, xs_non_outliers)

    signed_outliers = np.ones_like(outliers)
    signed_outliers_mask = linear_loss[outliers_mask] < 0
    signed_outliers[signed_outliers_mask] = -1.0

    xs_outliers = axis0_safe_slice(xs_norm, outliers_mask, num_outliers)

    gradient -= a * safe_sparse_dot(signed_outliers, xs_outliers)
    gradient += reg_param * w

    return loss, gradient


def find_coefficients_Huber(ys, xs, reg_param, a):
    _, d = xs.shape
    w = np.random.normal(loc=0.0, scale=1.0, size=(d,))
    xs_norm = np.divide(xs, np.sqrt(d))

    bounds = np.tile([-np.inf, np.inf], (w.shape[0], 1))
    bounds[-1][0] = np.finfo(np.float64).eps * 10

    opt_res = optimize.minimize(
        _loss_and_gradient_Huber,
        w,
        method="L-BFGS-B",
        jac=True,
        args=(xs_norm, ys, reg_param, a),
        options={"maxiter": MAX_ITER_MINIMIZE, "gtol": GTOL_MINIMIZE, "iprint": -1},
        bounds=bounds,
    )

    if opt_res.status == 2:
        raise ValueError(
            "HuberRegressor convergence failed: l-BFGS-b solver terminated with %s"
            % opt_res.message
        )

    return opt_res.x


# !!! !!! !!! !!! !!! !!! !!!
# !!! !!! !!! it lacks the constant value after
# @njit(error_model="numpy", fastmath=True)
def _loss_and_gradient_cutted_l2(w, xs_norm, ys, reg_param, a):
    linear_loss = ys - xs_norm @ w
    abs_linear_loss = np.abs(linear_loss)
    outliers_mask = abs_linear_loss > a

    outliers = abs_linear_loss[outliers_mask]
    num_outliers = np.count_nonzero(outliers_mask)
    n_non_outliers = xs_norm.shape[0] - num_outliers

    non_outliers = linear_loss[~outliers_mask]
    loss = 0.5 * np.dot(
        non_outliers, non_outliers
    )  # 0.0  # a * np.sum(outliers) - 0.5 * num_outliers * a ** 2
    loss += 0.5 * reg_param * np.dot(w, w)

    xs_non_outliers = -axis0_safe_slice(xs_norm, ~outliers_mask, n_non_outliers)

    signed_outliers = np.ones_like(outliers)
    signed_outliers_mask = linear_loss[outliers_mask] < 0
    signed_outliers[signed_outliers_mask] = -1.0

    # xs_outliers = axis0_safe_slice(xs_norm, outliers_mask, num_outliers)

    # gradient -= a * safe_sparse_dot(signed_outliers, xs_outliers)

    gradient = safe_sparse_dot(non_outliers, xs_non_outliers)
    gradient += reg_param * w

    return loss, gradient


def find_coefficients_cutted_l2(ys, xs, reg_param, a=1.0, max_iter=150, tol=1e-3):
    _, d = xs.shape
    w = np.random.normal(loc=0.0, scale=1.0, size=(d,))
    xs_norm = np.divide(xs, np.sqrt(d))

    bounds = np.tile([-np.inf, np.inf], (w.shape[0], 1))
    bounds[-1][0] = np.finfo(np.float64).eps * 10

    opt_res = optimize.minimize(
        _loss_and_gradient_cutted_l2,
        w,
        # method="L-BFGS-B",
        jac=True,
        args=(xs_norm, ys, reg_param, a),
        options={"maxiter": max_iter, "gtol": tol},
        # bounds=bounds,
    )

    if opt_res.status == 2:
        raise ValueError(
            "Cutted L2 Regressor convergence failed: solver terminated with %s"
            % opt_res.message
        )

    return opt_res.x


#  @njit(error_model="numpy", fastmath=True)
def _loss_and_gradient_double_quad(w, xs_norm, ys, reg_param, a):
    linear_loss = ys - xs_norm @ w
    print("linear loss shape ", linear_loss.shape)
    linear_loss_squared = (ys - xs_norm @ w) ** 2
    linear_loss_plus_width = linear_loss_squared + a

    loss = np.sum(
        0.5 * linear_loss_squared + linear_loss_squared / linear_loss_plus_width
    ) + 0.5 * reg_param * np.dot(w, w)
    # gradient = (
    #     safe_sparse_dot(
    #         linear_loss + 2 * a * linear_loss / (linear_loss_squared + a) ** 2, xs_norm
    #     )
    #     + reg_param * w
    # )
    gradient = (
        safe_sparse_dot(linear_loss, xs_norm)
        - 2 * safe_sparse_dot(linear_loss / linear_loss_plus_width, xs_norm)
        + 2 * safe_sparse_dot(linear_loss ** 3 / (linear_loss_plus_width ** 2), xs_norm)
        + reg_param * w
    )

    return loss, gradient


def find_coefficients_double_quad(ys, xs, reg_param, a=1.0, max_iter=1500, tol=5e-2):
    _, d = xs.shape
    w = np.random.normal(loc=0.0, scale=1.0, size=(d,))
    xs_norm = np.divide(xs, np.sqrt(d))

    bounds = np.tile([-np.inf, np.inf], (w.shape[0], 1))
    bounds[-1][0] = np.finfo(np.float64).eps * 10

    opt_res = optimize.minimize(
        _loss_and_gradient_double_quad,
        w,
        # method="L-BFGS-B",
        jac=True,
        args=(xs_norm, ys, reg_param, a),
        options={"maxiter": max_iter, "gtol": tol},
        # bounds=bounds,
    )

    if opt_res.status == 2:
        raise ValueError(
            "Double Quad Regressor convergence failed: solver terminated with %s"
            % opt_res.message
        )

    return opt_res.x
