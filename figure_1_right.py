import numpy as np
from itertools import product
from src.optimal_lambda import (
    optimal_lambda,
    optimal_reg_param_and_huber_parameter,
    no_parallel_optimal_reg_param_and_huber_parameter,
)
import matplotlib.pyplot as plt

from scipy.optimize import minimize
import src.fpeqs as fp
from src.fpeqs_BO import (
    var_func_BO,
    var_hat_func_BO_num_decorrelated_noise,
)
from src.fpeqs_L2 import (
    var_func_L2,
    var_hat_func_L2_decorrelated_noise,
)
from src.fpeqs_L1 import (
    var_hat_func_L1_decorrelated_noise,
)
from src.fpeqs_Huber import (
    var_hat_func_Huber_decorrelated_noise,
)

SMALLEST_REG_PARAM = 1e-7
SMALLEST_HUBER_PARAM = 1e-7
MAX_ITER = 2500
XATOL = 1e-8
FATOL = 1e-8

delta_large = 5.0
beta = 1.0
p = 0.3
delta_small = 1.0

while True:
    m = 0.89 * np.random.random() + 0.1
    q = 0.89 * np.random.random() + 0.1
    sigma = 0.89 * np.random.random() + 0.1
    if np.square(m) < q + delta_small * q and np.square(m) < q + delta_large * q:
        initial_condition = [m, q, sigma]
        break

pup = {
    "delta_small": delta_small,
    "delta_large": delta_large,
    "percentage": p,
    "beta": beta,
}

alphas_L2, errors_L2, lambdas_L2 = optimal_lambda(
    var_func_L2,
    var_hat_func_L2_decorrelated_noise,
    alpha_1=0.1,
    alpha_2=1000,
    n_alpha_points=60,
    initial_cond=initial_condition,
    var_hat_kwargs=pup,
)

alphas_L1, errors_L1, lambdas_L1 = optimal_lambda(
    var_func_L2,
    var_hat_func_L1_decorrelated_noise,
    alpha_1=0.1,
    alpha_2=1000,
    n_alpha_points=60,
    initial_cond=initial_condition,
    var_hat_kwargs=pup,
)

pep = {
    "delta_small": delta_small,
    "delta_large": delta_large,
    "percentage": p,
    "beta": beta,
}

(
    alphas_Huber,
    errors_Huber,
    lambdas_Huber,
    huber_params,
) = no_parallel_optimal_reg_param_and_huber_parameter(
    var_hat_func=var_hat_func_Huber_decorrelated_noise,
    alpha_1=0.1,
    alpha_2=1000,
    n_alpha_points=60,
    initial_cond=initial_condition,
    var_hat_kwargs=pep,
)

plt.plot(alphas_L2, errors_L2, label="ell_2")  #

plt.plot(
    alphas_L1,
    errors_L1,
    label="ell_1",
)

plt.plot(alphas_Huber, errors_Huber, label="Huber")

pap = {
    "delta_small": delta_small,
    "delta_large": delta_large,
    "percentage": p,
    "beta": beta,
}

alphas_BO, (errors_BO,) = fp.no_parallel_different_alpha_observables_fpeqs(
    var_func_BO,
    var_hat_func_BO_num_decorrelated_noise,
    alpha_1=0.1,
    alpha_2=1000,
    n_alpha_points=60,
    initial_cond=initial_condition,
    var_hat_kwargs=pap,
)

plt.plot(
    alphas_BO,
    errors_BO,
    label="BO",
)

plt.ylabel("E_{gen}")
plt.xlabel("alpha")
plt.xscale("log")
plt.yscale("log")
plt.legend()

plt.show()
