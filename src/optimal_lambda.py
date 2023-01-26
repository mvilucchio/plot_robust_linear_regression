import numpy as np
from scipy.optimize import minimize, Bounds
import src.fpeqs as fp
from src.fpeqs_L2 import var_func_L2
from src.fpeqs_Huber import (
    var_hat_func_Huber_single_noise,
    var_hat_func_Huber_double_noise,
)
from multiprocessing import Pool

from tqdm.auto import tqdm

# from mpi4py.futures import MPIPoolExecutor as Pool

SMALLEST_REG_PARAM = None # 1e-10
# SMALLEST_REG_PARAM = 1e-8
SMALLEST_HUBER_PARAM = 1e-8
MAX_ITER = 2500
XATOL = 1e-10
FATOL = 1e-10


def _find_optimal_reg_param_gen_error(
    alpha, var_func, var_hat_func, initial_cond, var_hat_kwargs, inital_value
):
    def minimize_fun(reg_param):
        m, q, _ = fp.state_equations(
            var_func,
            var_hat_func,
            reg_param=reg_param,
            alpha=alpha,
            init=initial_cond,
            var_hat_kwargs=var_hat_kwargs,
        )
        return 1 + q - 2 * m

    bnds = [(SMALLEST_REG_PARAM, None)]
    obj = minimize(
        minimize_fun,
        x0=inital_value,
        method="Nelder-Mead",
        bounds=bnds,
        options={"xatol": XATOL, "fatol": FATOL},
    )  # , , "maxiter":MAX_ITER
    if obj.success:
        fun_val = obj.fun
        reg_param_opt = obj.x
        return fun_val, reg_param_opt
    else:
        raise RuntimeError("Minima could not be found.")


def no_parallel_optimal_lambda(
    var_func,
    var_hat_func,
    alpha_1=0.01,
    alpha_2=100,
    n_alpha_points=16,
    initial_cond=[0.6, 0.0, 0.0],
    var_hat_kwargs={},
    initial_2=True,
    reverse=False,
):
    alphas = np.logspace(np.log(alpha_1) / np.log(10), np.log(alpha_2) / np.log(10), n_alpha_points)

    fun_values = np.zeros(n_alpha_points)
    reg_param_opt = np.zeros(n_alpha_points)
    # init_param = (1 - var_hat_kwargs.get("percentage")) * var_hat_kwargs.get(
    #     "delta_small"
    # ) + var_hat_kwargs.get("percentage") * var_hat_kwargs.get("delta_large")

    # initial = initial_cond
    fun_values = np.zeros(n_alpha_points)
    reg_param_opt = np.zeros(n_alpha_points)

    initial_reg_params = alphas ** (-0.5)

    if reverse:
        old_rp_opt = initial_reg_params[-1]

        for idx, (a, init_rp) in enumerate(zip(tqdm(alphas[::-1]), initial_reg_params[::-1])):
            (
                fun_values[n_alpha_points - 1 - idx],
                reg_param_opt[n_alpha_points - 1 - idx],
            ) = _find_optimal_reg_param_gen_error(
                a, var_func, var_hat_func, initial_cond, var_hat_kwargs, old_rp_opt
            )

            if initial_2:
                old_rp_opt = reg_param_opt[n_alpha_points - 1 - idx]
            else:
                old_rp_opt = init_rp
    else:
        old_rp_opt = initial_reg_params[0]

        for idx, (a, init_rp) in enumerate(zip(tqdm(alphas), initial_reg_params)):
            (fun_values[idx], reg_param_opt[idx]) = _find_optimal_reg_param_gen_error(
                a, var_func, var_hat_func, initial_cond, var_hat_kwargs, old_rp_opt
            )

            if initial_2:
                old_rp_opt = reg_param_opt[idx]
            else:
                old_rp_opt = init_rp

    return alphas, fun_values, reg_param_opt


def optimal_lambda(
    var_func,
    var_hat_func,
    alpha_1=0.01,
    alpha_2=100,
    n_alpha_points=16,
    initial_cond=[0.6, 0.0, 0.0],
    var_hat_kwargs={},
):
    alphas = np.logspace(np.log(alpha_1) / np.log(10), np.log(alpha_2) / np.log(10), n_alpha_points)

    fun_values = np.zeros(n_alpha_points)
    reg_param_opt = np.zeros(n_alpha_points)
    init_param = (1 - var_hat_kwargs.get("percentage")) * var_hat_kwargs.get(
        "delta_small"
    ) + var_hat_kwargs.get("percentage") * var_hat_kwargs.get(
        "delta_large"
    )  # 0.1 * np.random.random() + 0.1
    # init_param = []

    inputs = [(a, var_func, var_hat_func, initial_cond, var_hat_kwargs, init_param) for a in alphas]

    with Pool() as pool:
        results = pool.starmap(_find_optimal_reg_param_gen_error, inputs)

    for idx, (e, regp) in enumerate(results):
        fun_values[idx] = e
        reg_param_opt[idx] = regp

    return alphas, fun_values, reg_param_opt


def _find_optimal_huber_parameter_gen_error(
    alpha, double_noise, reg_param, initial, var_hat_kwargs, inital_value
):
    def error_func(a):
        var_hat_kwargs.update({"a": a})
        m, q, _ = fp.state_equations(
            var_func_L2,
            var_hat_func_Huber_double_noise if double_noise else var_hat_func_Huber_single_noise,
            reg_param=reg_param,
            alpha=alpha,
            init=initial,
            var_hat_kwargs=var_hat_kwargs,
        )
        return 1 + q - 2 * m

    bnds = [(SMALLEST_HUBER_PARAM, None)]
    obj = minimize(
        error_func,
        x0=inital_value,
        # method="Nelder-Mead",
        bounds=bnds,
        options={"xatol": XATOL, "fatol": FATOL},
    )  # ,
    if obj.success:
        fun_val = obj.fun
        a_opt = obj.x
        return fun_val, a_opt
    else:
        raise RuntimeError("Minima could not be found.")


def optimal_huber_parameter(
    double_noise=True,
    reg_param=1.5,
    alpha_1=0.01,
    alpha_2=100,
    n_alpha_points=16,
    initial_cond=[0.6, 0.0, 0.0],
    var_hat_kwargs={},
):

    alphas = np.logspace(np.log(alpha_1) / np.log(10), np.log(alpha_2) / np.log(10), n_alpha_points)

    fun_values = np.zeros(n_alpha_points)
    a_opt = np.zeros(n_alpha_points)

    inputs = [(a, double_noise, reg_param, initial_cond, var_hat_kwargs, 1.0) for a in alphas]

    with Pool() as pool:
        results = pool.starmap(_find_optimal_huber_parameter_gen_error, inputs)

    for idx, (e, a) in enumerate(results):
        fun_values[idx] = e
        a_opt[idx] = a

    return alphas, fun_values, a_opt


def _find_optimal_reg_param_and_huber_parameter_gen_error(
    alpha, var_hat_func, initial, var_hat_kwargs, inital_values
):
    def minimize_fun(x):
        reg_param, a = x
        var_hat_kwargs.update({"a": a})
        m, q, _ = fp.state_equations(
            var_func_L2,
            var_hat_func,
            reg_param=reg_param,
            alpha=alpha,
            init=initial,
            var_hat_kwargs=var_hat_kwargs,
        )
        return 1 + q - 2 * m

    bnds = [(SMALLEST_REG_PARAM, None), (SMALLEST_HUBER_PARAM, None)]
    obj = minimize(
        minimize_fun,
        x0=inital_values,
        method="Nelder-Mead",
        bounds=bnds,
        options={
            "xatol": XATOL,
            "fatol": FATOL,
            "adaptive": True,
        },
    )
    if obj.success:
        fun_val = obj.fun
        reg_param_opt, a_opt = obj.x
        return fun_val, reg_param_opt, a_opt
    else:
        raise RuntimeError("Minima could not be found.")


def optimal_reg_param_and_huber_parameter(
    var_hat_func=var_hat_func_Huber_double_noise,
    alpha_1=0.01,
    alpha_2=100,
    n_alpha_points=16,
    initial_cond=[0.6, 0.0, 0.0],
    var_hat_kwargs={},
):
    alphas = np.logspace(np.log(alpha_1) / np.log(10), np.log(alpha_2) / np.log(10), n_alpha_points)

    initial = initial_cond
    fun_values = np.zeros(n_alpha_points)
    reg_param_opt = np.zeros(n_alpha_points)
    a_opt = np.zeros(n_alpha_points)

    inital_reg_param = 0.2 * np.random.random() + 0.9
    inital_hub_param = 0.2 * np.random.random() + 0.9

    inputs = [
        (a, var_hat_func, initial, var_hat_kwargs, [inital_reg_param, inital_hub_param])
        for a in alphas
    ]

    with Pool() as pool:
        results = pool.starmap(_find_optimal_reg_param_and_huber_parameter_gen_error, inputs)

    for idx, (e, regp, a) in enumerate(results):
        fun_values[idx] = e
        reg_param_opt[idx] = regp
        a_opt[idx] = a

    return alphas, fun_values, reg_param_opt, a_opt


def no_parallel_optimal_reg_param_and_huber_parameter(
    var_hat_func=var_hat_func_Huber_double_noise,
    alpha_1=0.01,
    alpha_2=100,
    n_alpha_points=16,
    initial_cond=[0.6, 0.0, 0.0],
    var_hat_kwargs={},
    initial_2=True,
    reverse=False,
):
    alphas = np.logspace(np.log(alpha_1) / np.log(10), np.log(alpha_2) / np.log(10), n_alpha_points)

    initial = initial_cond
    fun_values = np.zeros(n_alpha_points)
    reg_param_opt = np.zeros(n_alpha_points)
    a_opt = np.zeros(n_alpha_points)

    initial_reg_params = alphas ** (-0.5)
    initial_hub_params = alphas ** (-0.5)

    if reverse:
        old_a_opt = initial_hub_params[-1]
        old_rp_opt = initial_reg_params[-1]

        for idx, (a, init_rp, init_hp) in enumerate(
            zip(tqdm(alphas[::-1]), initial_reg_params[::-1], initial_hub_params[::-1])
        ):
            (
                fun_values[n_alpha_points - 1 - idx],
                reg_param_opt[n_alpha_points - 1 - idx],
                a_opt[n_alpha_points - 1 - idx],
            ) = _find_optimal_reg_param_and_huber_parameter_gen_error(
                a, var_hat_func, initial, var_hat_kwargs, [old_rp_opt, old_a_opt]
            )

            if initial_2:
                old_rp_opt = reg_param_opt[n_alpha_points - 1 - idx]
                old_a_opt = a_opt[n_alpha_points - 1 - idx]
            else:
                old_rp_opt = init_rp
                old_a_opt = init_hp

    else:
        old_a_opt = initial_hub_params[0]
        old_rp_opt = initial_reg_params[0]

        for idx, (a, init_rp, init_hp) in enumerate(
            zip(tqdm(alphas), initial_reg_params, initial_hub_params)
        ):
            (
                fun_values[idx],
                reg_param_opt[idx],
                a_opt[idx],
            ) = _find_optimal_reg_param_and_huber_parameter_gen_error(
                a, var_hat_func, initial, var_hat_kwargs, [old_rp_opt, old_a_opt]
            )

            if initial_2:
                old_rp_opt = reg_param_opt[idx]
                old_a_opt = a_opt[idx]
            else:
                old_rp_opt = init_rp
                old_a_opt = init_hp

    return alphas, fun_values, reg_param_opt, a_opt
