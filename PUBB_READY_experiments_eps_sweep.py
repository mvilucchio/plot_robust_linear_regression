import numpy as np
from multiprocessing import Pool
from scipy.optimize import minimize
import src.numerics as num
import src.fpeqs as fpe
from src.fpeqs_BO import (
    var_func_BO,
    var_hat_func_BO_single_noise,
    var_hat_func_BO_num_double_noise,
    var_hat_func_BO_num_decorrelated_noise,
)
from src.fpeqs_L2 import (
    var_func_L2,
    var_hat_func_L2_single_noise,
    var_hat_func_L2_double_noise,
    var_hat_func_L2_decorrelated_noise,
)
from src.fpeqs_L1 import (
    var_hat_func_L1_single_noise,
    var_hat_func_L1_double_noise,
    var_hat_func_L1_decorrelated_noise,
)
from src.fpeqs_Huber import (
    var_hat_func_Huber_single_noise,
    var_hat_func_Huber_double_noise,
    var_hat_func_Huber_decorrelated_noise,
)

# num.find_coefficients_L2,
# num.find_coefficients_L1,
# num.find_coefficients_Huber,

alpha_cut = 0.1
delta_small = 0.1
delta_large = 5.0
beta = 0.0

n_features = 1000
repetitions = 10

epsilons = np.logspace(-4, np.log10(0.5), 36)
errors_mean = np.empty_like(epsilons)
errors_std = np.empty_like(epsilons)

measure_fun_args_all = [
    (
        delta_small,
        delta_large,
        eps,
        beta,
    )
    for eps in epsilons
]

find_coefficients_fun_args = [(reg_p, a) for reg_p, a in zip(optimal_reg_param, optimal_a)]

inputs = [
    (
        alpha_cut,
        measure_fun,
        find_coefficients_fun,
        n_features,
        repetitions,
        measure_fun_args,
        fc_agrs,
    )
    for fc_agrs, measure_fun_args in zip(find_coefficients_fun_args, measure_fun_args_all)
]

with Pool() as pool:
    results = pool.starmap(num._find_numerical_mean_std, inputs)

for idx, r in enumerate(results):
    errors_mean[idx] = r[0]
    errors_std[idx] = r[1]
