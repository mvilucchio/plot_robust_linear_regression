from csv import reader
from src.numerics import (
    _find_numerical_mean_std,
    measure_gen_decorrelated,
    find_coefficients_L2,
    find_coefficients_L1,
    find_coefficients_Huber,
)
from tqdm.auto import tqdm

from matplotlib.ticker import FormatStrFormatter
from matplotlib.lines import Line2D
from src.utils import load_file
import src.plotting_utils as pu
import matplotlib.pyplot as plt
import numpy as np
from itertools import product
from src.optimal_lambda import (
    optimal_lambda,
    optimal_reg_param_and_huber_parameter,
    no_parallel_optimal_reg_param_and_huber_parameter,
)

from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as ticker
import src.plotting_utils as pu

from scipy.optimize import minimize
import src.fpeqs as fp
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

delta_large = 5.0
# beta = 1.0
p = 0.3
delta_small = 1.0

d = 1000

# figleg1 = plt.figure(figsize=(tuple_size[0], tuple_size[1] / 17))
# figleg2 = plt.figure(figsize=(tuple_size[0], tuple_size[1] / 17))

cmap = plt.get_cmap("tab10")
color_lines = []
error_names = []
error_names_latex = []

reg_param_lines = []

while True:
    m = 0.89 * np.random.random() + 0.1
    q = 0.89 * np.random.random() + 0.1
    sigma = 0.89 * np.random.random() + 0.1
    if np.square(m) < q + delta_small * q and np.square(m) < q + delta_large * q:
        initial_condition = [m, q, sigma]
        break

alphas = []
reg_param_l2 = []
reg_param_l1 = []
reg_param_hub = []
huber_param = []

beta = 0.0
with open(
    "./data/GOOD_sweep_alpha_fixed_eps_0.30_beta_0.00_delta_large_5.00_delta_small.csv", "r"
) as read_obj:
    csv_reader = reader(read_obj)
    for idx, row in enumerate(csv_reader):
        if idx == 0:
            continue

        if float(row[0]) <= 1000 and idx % 3 == 0:
            alphas.append(float(row[0]))
            reg_param_l2.append(float(row[2]))
            reg_param_l1.append(float(row[4]))
            reg_param_hub.append(float(row[6]))
            huber_param.append(float(row[7]))


n_alphas = len(alphas)
error_l2_mean = np.empty(n_alphas)
error_l2_std = np.empty(n_alphas)
error_l1_mean = np.empty(n_alphas)
error_l1_std = np.empty(n_alphas)
error_hub_mean = np.empty(n_alphas)
error_hub_std = np.empty(n_alphas)

for idx, (al, regl2, regl1, reg_hub, a_hub) in enumerate(
    zip(tqdm(alphas), reg_param_l2, reg_param_l1, reg_param_hub, huber_param)
):
    measure_fun_args = (
        delta_small,
        delta_large,
        p,
        beta,
    )

    find_coefficients_fun_args_l2 = (regl2,)

    error_l2_mean[idx], error_l2_std[idx] = _find_numerical_mean_std(
        al,
        measure_gen_decorrelated,
        find_coefficients_L2,
        d,
        10,
        measure_fun_args,
        find_coefficients_fun_args_l2,
    )

    # find_coefficients_fun_args_l1 = (regl1,)

    # error_l1_mean[idx], error_l1_std[idx] = _find_numerical_mean_std(
    #     al,
    #     measure_gen_decorrelated,
    #     find_coefficients_L1,
    #     d,
    #     10,
    #     measure_fun_args,
    #     find_coefficients_fun_args_l1,
    # )

    find_coefficients_fun_args_hub = (reg_hub, a_hub)

    error_hub_mean[idx], error_hub_std[idx] = _find_numerical_mean_std(
        al,
        measure_gen_decorrelated,
        find_coefficients_Huber,
        d,
        10,
        measure_fun_args,
        find_coefficients_fun_args_hub,
    )

np.savetxt(
    "./data/numerics_sweep_alpha_just_l1_fixed_eps_{:.2f}_beta_{:.2f}_delta_large_{:.2f}_delta_small_{:.2f}_dim_{:.2f}.csv".format(
        p, beta, delta_large, delta_small, d
    ),
    np.vstack(
        (
            np.array(alphas),
            # error_l2_mean,
            # error_l2_std,
            error_l1_mean,
            error_l1_std,
            # error_hub_mean,
            # error_hub_std
        )
    ).T,
    delimiter=",",
    header="alpha,l1_mean,l1_std",
)

print("Done.")

# find_coefficients_fun_args_l2 = (reg_param_l2,)

# error_l2_mean[idx], error_l2_std[idx] = _find_numerical_mean_std(
#     al,
#     measure_gen_decorrelated,
#     find_coefficients_L2,
#     d,
#     10,
#     measure_fun_args,
#     find_coefficients_fun_args_l2,
# )

# find_coefficients_fun_args_l2 = (reg_param_l2,)

# error_l2_mean[idx], error_l2_std[idx] = _find_numerical_mean_std(
#     al,
#     measure_gen_decorrelated,
#     find_coefficients_L2,
#     d,
#     10,
#     measure_fun_args,
#     find_coefficients_fun_args_l2,
# )
