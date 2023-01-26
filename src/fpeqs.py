import numpy as np

# from numba import njit
# from math import erfc  # , erf
#  import src.numerical_functions as numfun
# from scipy.special import log_ndtr, erf
from multiprocessing import Pool
from tqdm.auto import tqdm

# from mpi4py import MPI
# from mpi4py.futures import MPIPoolExecutor as Pool

BLEND = 0.75
TOL_FPE = 1e-11


def state_equations(var_func, var_hat_func, reg_param, alpha, init, var_hat_kwargs):
    m, q, sigma = init[0], init[1], init[2]
    # print("-- ", type(m), type(q), type(sigma), type(reg_param))
    err = 1.0
    reg_param = float(reg_param)
    #  print("alpha : {:.3f} a : {:.9f} reg_par : {:.7f}".format(alpha, var_hat_kwargs["a"], reg_param))
    while err > TOL_FPE:
        m_hat, q_hat, sigma_hat = var_hat_func(m, q, sigma, alpha, **var_hat_kwargs)
        # print("after first ", type(m_hat), type(q_hat), type(sigma_hat))

        temp_m, temp_q, temp_sigma = m, q, sigma
        # print("after second ",type(temp_m), type(temp_q), type(temp_sigma))
        m, q, sigma = var_func(m_hat, q_hat, sigma_hat, reg_param)
        # print("after third ",type(m), type(q), type(sigma))

        err = np.max(np.abs([(temp_m - m), (temp_q - q), (temp_sigma - sigma)]))

        m = BLEND * m + (1 - BLEND) * temp_m
        q = BLEND * q + (1 - BLEND) * temp_q
        sigma = BLEND * sigma + (1 - BLEND) * temp_sigma
        
        # print(err)
        # print(
        #     "  err: {:.8f} alpha : {:.2f} m = {:.9f}; q = {:.9f}; \[CapitalSigma] = {:.9f}".format(
        #         err, alpha, m, q, sigma
        #     )
        # )

    return m, q, sigma


def _find_fixed_point(alpha, var_func, var_hat_func, reg_param, initial_cond, var_hat_kwargs):
    m, q, sigma = state_equations(
        var_func,
        var_hat_func,
        reg_param=reg_param,
        alpha=alpha,
        init=initial_cond,
        var_hat_kwargs=var_hat_kwargs,
    )
    return m, q, sigma


def no_parallel_different_alpha_observables_fpeqs_parallel(
    var_func,
    var_hat_func,
    funs=[lambda m, q, sigma: 1 + q - 2 * m],
    alpha_1=0.01,
    alpha_2=100,
    n_alpha_points=16,
    reg_param=0.1,
    initial_cond=[0.6, 0.0, 0.0],
    var_hat_kwargs={},
):
    n_observables = len(funs)
    alphas = np.logspace(np.log(alpha_1) / np.log(10), np.log(alpha_2) / np.log(10), n_alpha_points)
    out_values = np.empty((n_observables, n_alpha_points))
    results = [None] * len(alphas)

    for idx, (a, r, k) in enumerate(zip(tqdm(alphas), reg_param, var_hat_kwargs)):
        results[idx] = _find_fixed_point(a, var_func, var_hat_func, r, initial_cond, k)
    # inputs = [
    #     (a, var_func, var_hat_func, reg_param, initial_cond, var_hat_kwargs)
    #     for a in alphas
    # ]

    # with Pool() as pool:
    #     results = pool.starmap(_find_fixed_point, inputs)

    for idx, (m, q, sigma) in enumerate(results):
        fixed_point_sols = {"m": m, "q": q, "sigma": sigma}
        for jdx, f in enumerate(funs):
            out_values[jdx, idx] = f(**fixed_point_sols)

    out_list = [out_values[idx, :] for idx in range(len(funs))]
    return alphas, out_list


def no_parallel_different_alpha_observables_fpeqs(
    var_func,
    var_hat_func,
    funs=[lambda m, q, sigma: 1 + q - 2 * m],
    alpha_1=0.01,
    alpha_2=100,
    n_alpha_points=16,
    reg_param=0.1,
    initial_cond=[0.6, 0.0, 0.0],
    var_hat_kwargs={},
):
    n_observables = len(funs)
    alphas = np.logspace(np.log(alpha_1) / np.log(10), np.log(alpha_2) / np.log(10), n_alpha_points)
    out_values = np.empty((n_observables, n_alpha_points))
    results = [None] * len(alphas)

    for idx, a in enumerate(tqdm(alphas)):
        results[idx] = _find_fixed_point(
            a, var_func, var_hat_func, reg_param, initial_cond, var_hat_kwargs
        )

    for idx, (m, q, sigma) in enumerate(results):
        fixed_point_sols = {"m": m, "q": q, "sigma": sigma}
        for jdx, f in enumerate(funs):
            out_values[jdx, idx] = f(**fixed_point_sols)

    out_list = [out_values[idx, :] for idx in range(len(funs))]
    return alphas, out_list


def MPI_different_alpha_observables_fpeqs(
    var_func,
    var_hat_func,
    funs=[lambda m, q, sigma: 1 + q - 2 * m],
    alpha_1=0.01,
    alpha_2=100,
    n_alpha_points=16,
    reg_param=0.1,
    initial_cond=[0.6, 0.0, 0.0],
    var_hat_kwargs={},
):
    comm = MPI.COMM_WORLD
    i = comm.Get_rank()
    pool_size = comm.Get_size()

    n_observables = len(funs)
    alphas = np.logspace(np.log(alpha_1) / np.log(10), np.log(alpha_2) / np.log(10), pool_size)
    alpha = alphas[i]
    out_values = np.empty((n_observables, n_alpha_points))

    m, q, sigma = _find_fixed_point(
        alpha, var_func, var_hat_func, reg_param, initial_cond, var_hat_kwargs
    )

    ms = np.empty(pool_size)
    qs = np.empty(pool_size)
    sigmas = np.empty(pool_size)

    if i == 0:
        ms[0] = m
        qs[0] = q
        sigmas[0] = sigma

        for j in range(1, pool_size):
            ms[j] = comm.recv(source=j)
        for j in range(1, pool_size):
            qs[j] = comm.recv(source=j)
        for j in range(1, pool_size):
            sigmas[j] = comm.recv(source=j)

        for idx, (mm, qq, ssigma) in enumerate(zip(ms, qs, sigmas)):
            fixed_point_sols = {"m": mm, "q": qq, "sigma": ssigma}
            for jdx, f in enumerate(funs):
                out_values[jdx, idx] = f(**fixed_point_sols)

        out_list = [out_values[idx, :] for idx in range(len(funs))]
        return alphas, out_list
    else:
        print("Process {} sending {}".format(i, reg_param))
        comm.send(m, dest=0)
        comm.send(q, dest=0)
        comm.send(sigma, dest=0)


def different_alpha_observables_fpeqs(
    var_func,
    var_hat_func,
    funs=[lambda m, q, sigma: 1 + q - 2 * m],
    alpha_1=0.01,
    alpha_2=100,
    n_alpha_points=16,
    reg_param=0.1,
    initial_cond=[0.6, 0.0, 0.0],
    var_hat_kwargs={},
):
    n_observables = len(funs)
    alphas = np.logspace(np.log(alpha_1) / np.log(10), np.log(alpha_2) / np.log(10), n_alpha_points)
    out_values = np.empty((n_observables, n_alpha_points))

    inputs = [(a, var_func, var_hat_func, reg_param, initial_cond, var_hat_kwargs) for a in alphas]

    with Pool() as pool:
        results = pool.starmap(_find_fixed_point, inputs)

    for idx, (m, q, sigma) in enumerate(results):
        fixed_point_sols = {"m": m, "q": q, "sigma": sigma}
        for jdx, f in enumerate(funs):
            out_values[jdx, idx] = f(**fixed_point_sols)

    out_list = [out_values[idx, :] for idx in range(len(funs))]
    return alphas, out_list


def different_reg_param_gen_error(
    var_func,
    var_hat_func,
    funs=[lambda m, q, sigma: 1 + q - 2 * m],
    reg_param_1=0.01,
    reg_param_2=100,
    n_reg_param_points=16,
    alpha=0.1,
    initial_cond=[0.6, 0.0, 0.0],
    var_hat_kwargs={},
):
    n_observables = len(funs)
    reg_params = np.logspace(
        np.log(reg_param_1) / np.log(10),
        np.log(reg_param_2) / np.log(10),
        n_reg_param_points,
    )
    out_values = np.empty((n_observables, n_reg_param_points))

    inputs = [
        (alpha, var_func, var_hat_func, rp, initial_cond, var_hat_kwargs) for rp in reg_params
    ]

    with Pool() as pool:
        results = pool.starmap(_find_fixed_point, inputs)

    for idx, (m, q, sigma) in enumerate(results):
        fixed_point_sols = {"m": m, "q": q, "sigma": sigma}
        for jdx, f in enumerate(funs):
            out_values[jdx, idx] = f(**fixed_point_sols)

    out_list = [out_values[idx, :] for idx in range(len(funs))]
    return reg_params, out_list
