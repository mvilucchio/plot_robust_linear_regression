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

SMALLEST_REG_PARAM = 1e-7
SMALLEST_HUBER_PARAM = 1e-7
MAX_ITER = 2500
XATOL = 1e-8
FATOL = 1e-8

save = True
experimental_points = True
width = 1.0 * 458.63788

delta_large = 5.0
beta = 1.0
p = 0.1
delta_small = 1.0

pu.initialization_mpl()

tuple_size = pu.set_size(width, fraction=0.50)

multiplier = 0.9
second_multiplier = 0.7

fig, ax = plt.subplots(1, 1, figsize=(multiplier*tuple_size[0],multiplier*tuple_size[0]))
fig.subplots_adjust(left=0.16)
fig.subplots_adjust(bottom=0.16)
fig.subplots_adjust(top=0.97)
fig.subplots_adjust(right=0.97)
fig.set_zorder(30)
ax.set_zorder(30)

# while True:
#     m = 0.89 * np.random.random() + 0.1
#     q = 0.89 * np.random.random() + 0.1
#     sigma = 0.89 * np.random.random() + 0.1
#     if np.square(m) < q + delta_small * q and np.square(m) < q + delta_large * q:
#         initial_condition = [m, q, sigma]
#         break

# # alphas_L2, errors_L2, lambdas_L2 = load_file(**L2_settings)

# pup = {
#     "delta_small": delta_small,
#     "delta_large": delta_large,
#     "percentage": p,
#     "beta": beta,
# }

# alphas_L2, errors_L2, lambdas_L2 = optimal_lambda(
#     var_func_L2,
#     var_hat_func_L2_decorrelated_noise,
#     alpha_1=0.1,
#     alpha_2=10000,
#     n_alpha_points=100,
#     initial_cond=initial_condition,
#     var_hat_kwargs=pup,
# )

# alphas_L1, errors_L1, lambdas_L1 = optimal_lambda(
#     var_func_L2,
#     var_hat_func_L1_decorrelated_noise,
#     alpha_1=0.1,
#     alpha_2=10000,
#     n_alpha_points=100,
#     initial_cond=initial_condition,
#     var_hat_kwargs=pup,
# )

# # alphas_Huber, errors_Huber, lambdas_Huber, huber_params = load_file(**Huber_settings)
# pep = {
#     "delta_small": delta_small,
#     "delta_large": delta_large,
#     "percentage": p,
#     "beta": beta,
# }

# (
#     alphas_Huber,
#     errors_Huber,
#     lambdas_Huber,
#     huber_params,
# ) = no_parallel_optimal_reg_param_and_huber_parameter(
#     var_hat_func=var_hat_func_Huber_decorrelated_noise,
#     alpha_1=0.1,
#     alpha_2=10000,
#     n_alpha_points=100,
#     initial_cond=initial_condition,
#     var_hat_kwargs=pep,
# )

# pap = {
#     "delta_small": delta_small,
#     "delta_large": delta_large,
#     "percentage": p,
#     "beta": beta,
# }

# alphas_BO, (errors_BO,) = fp.no_parallel_different_alpha_observables_fpeqs(
#     var_func_BO,
#     var_hat_func_BO_num_decorrelated_noise,
#     alpha_1=0.1,
#     alpha_2=10000,
#     n_alpha_points=100,
#     initial_cond=initial_condition,
#     var_hat_kwargs=pap,
# )

# np.savetxt(
#     "./data/single_param_correlated_fig_1.csv",
#     np.vstack((alphas_L2, errors_L2, lambdas_L2, errors_L1, lambdas_L1, errors_Huber,lambdas_Huber, huber_params, errors_BO)).T,
#     delimiter=",",
#     header="# alphas_L2, errors_L2, lambdas_L2, errors_L1, lambdas_L1, errors_Huber,lambdas_Huber, huber_params, errors_BO",
# )

data_fp = np.genfromtxt(
    "./data/FIGURE_1_data_correlated.csv",
    delimiter=",",
    skip_header=1,
)

alphas_L2 = data_fp[:,0]
errors_L2 = data_fp[:,1]
lambdas_L2 = data_fp[:,2]
errors_L1 = data_fp[:,3]
lambdas_L1 = data_fp[:,4]
errors_Huber = data_fp[:,5]
lambdas_Huber = data_fp[:,6]
huber_params = data_fp[:,7]
errors_BO = data_fp[:,8]

# alphas_BO, errors_BO = load_file(**BO_settings)

dat_l2_hub = np.genfromtxt(
    "./data/FIGURE_1_data_numerics_correlated.csv",
    skip_header=1,
    delimiter=",",
)
alph_num = dat_l2_hub[:, 0]
err_mean_l2 = dat_l2_hub[:, 1]
err_std_l2 = dat_l2_hub[:, 2]
err_mean_l1 = dat_l2_hub[:, 3]
err_std_l1 = dat_l2_hub[:, 4]
err_mean_hub = dat_l2_hub[:, 5]
err_std_hub = dat_l2_hub[:, 6]

# dat_l1 = np.genfromtxt(
#     "./data/GOOD_beta_1.0_l1.csv", # "./data/numerics_sweep_alpha_just_l1_fixed_eps_0.30_beta_0.00_delta_large_5.00_delta_small_1.00_dim_500.00_bak.csv",
#     skip_header=1,
#     delimiter=",",
#     # dtype="float",
# )
# alpha_l1 = dat_l1[:, 0]
# err_mean_l1 = dat_l1[:, 1]
# err_std_l1 = dat_l1[:, 2]


ax.plot(
    alphas_L2,
    errors_L2,
    label=r"$\ell_2$",
    color="tab:blue",
    zorder=3,  # ,linewidth=1.0
)
ax.errorbar(
    alph_num,
    err_mean_l2,
    yerr=err_std_l2,
    color="tab:blue",
    linestyle="",
    elinewidth=0.75,
    markerfacecolor="none",
    markeredgecolor="tab:blue",
    marker="o",
    markersize=1.0,
    zorder=3,
)

ax.plot(
    alphas_L2,
    errors_L1,
    label=r"$\ell_1$",
    color="tab:green",
    zorder=5
    # linewidth=1.0
)
ax.errorbar(
    alph_num,
    err_mean_l1,
    yerr=err_std_l1,
    color="tab:green",
    linestyle="",
    elinewidth=0.75,
    markerfacecolor="none",
    markeredgecolor="tab:green",
    marker="o",
    markersize=1.0,
    zorder=5,
)

ax.plot(
    alphas_L2,
    errors_Huber,
    label="Huber",
    color="tab:orange",
    zorder=10,  # , linewidth=1.0
)  # r"$\mathcal{L}_{a_{\text{\tiny{opt}}}}$",
ax.errorbar(
    alph_num,
    err_mean_hub,
    yerr=err_std_hub,
    color="tab:orange",
    linestyle="",
    elinewidth=0.75,
    markerfacecolor="none",
    markeredgecolor="tab:orange",
    marker="o",
    markersize=1.0,
    zorder=10,
)

ax.plot(alphas_L2, errors_BO, label="BO", color="tab:red", linewidth=0.5, zorder=15)


# ax.set_ylabel(r"$E_{\text{gen}}$", labelpad=2.0)
# ax.set_ylabel(r"$E_{\text{gen}}-E_{\text{gen}}^{\text{BO}}$")
# ax.set_xlabel(r"$\alpha$", labelpad=2.0)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim([0.1, 100])
ax.set_ylim([0.005, 1.9])
# ax.legend(loc="upper right", handlelength=1.0)
ax.grid(zorder=20)

ax.tick_params(axis="y", pad=2.0)
ax.tick_params(axis="x", pad=2.0)

if save:
    pu.save_plot(
        fig,
        "FIGURE_1_left",
    )

plt.show()

tuple_size = pu.set_size(width, fraction=0.50)

fig_2, ax_2 = plt.subplots(1, 1, figsize=(multiplier*tuple_size[0], second_multiplier*multiplier*tuple_size[1]))
# important
fig_2.subplots_adjust(left=0.16)
fig_2.subplots_adjust(bottom=0.3)
fig_2.subplots_adjust(top=0.97)
fig_2.subplots_adjust(right=0.97)
fig_2.set_zorder(30)
ax_2.set_zorder(30)


ax_2.plot(alphas_L2, lambdas_L2, label=r"$\lambda_{\text{opt}}\,\ell_2$", color='tab:blue', linestyle='solid')
ax_2.plot(alphas_L2, lambdas_L1, label=r"$\lambda_{\text{opt}}\,\ell_1$", color='tab:green', linestyle='solid')
ax_2.plot(alphas_L2, lambdas_Huber, label=r"$\lambda_{\text{opt}}$ Huber", color='tab:orange', linestyle='solid')
ax_2.plot(alphas_L2, huber_params, label=r"$a_{\text{opt}}$", color='tab:gray', linestyle='solid')

# ax_2.set_ylabel(r"$a_{\text{opt}}$", labelpad=2.0)
# ax_2.set_xlabel(r"$\alpha$", labelpad=0.0)
ax_2.set_xscale("log")
# ax_2.set_yscale("log")
ax_2.set_xlim([0.1, 100])
# ax_2.set_ylim([1.5, 1.7])
ax_2.grid(zorder=20)
# leg = ax_2.legend(loc="best", handlelength=1.0)

ax_2.tick_params(axis="y", pad=2.0)
ax_2.tick_params(axis="x", pad=2.0)

if save:
    pu.save_plot(
        fig_2,
        "FIGURE_1_left_parameters",
    )

plt.show()
