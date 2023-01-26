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

from scipy.special import erf


def plateau_L1(D_IN, D_OUT, epsilon, x=0.01, toll=1e-8):
    err = np.inf
    while True:
        y_IN = D_IN + epsilon**2 * x**2
        y_OUT = D_OUT + (epsilon * x + 1) ** 2
        x_next = -1 / ((1 - epsilon) * np.sqrt(y_OUT / y_IN) + epsilon)
        err = np.abs(x_next - x)
        x = x_next
        if err < toll:
            return x


def plateau_H(D_IN, D_OUT, epsilon, a, x=0.01, toll=1e-8):
    err = np.inf
    while True:
        y_IN = D_IN + epsilon**2 * x**2
        y_OUT = D_OUT + (epsilon * x + 1) ** 2

        x_next = -1 / (
            (1 - epsilon) * erf(a / np.sqrt(2 * y_IN)) / erf(a / np.sqrt(2 * y_OUT)) + epsilon
        )
        err = np.abs(x_next - x)
        x = x_next
        if err < toll:
            return x


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


# remember to put lower bound also in optimal_lambda
SMALLEST_REG_PARAM = 1e-10
SMALLEST_HUBER_PARAM = 1e-8
MAX_ITER = 2500
XATOL = 1e-10
FATOL = 1e-10

save = True
experimental_points = True
width = 4/5 * 1.0 * 458.63788

delta_large = 5.0
beta = 0.0
p = 0.3
delta_small = 1.0

pu.initialization_mpl()

tuple_size = pu.set_size(width, fraction=0.50)

multiplier = 0.9
second_multiplier = 0.7

fig, ax = plt.subplots(1, 1, figsize=(multiplier*tuple_size[0],3/4 * multiplier*tuple_size[0]))
fig.subplots_adjust(left=0.16)
fig.subplots_adjust(bottom=0.16)
fig.subplots_adjust(top=0.97)
fig.subplots_adjust(right=0.97)
fig.set_zorder(30)
ax.set_zorder(30)

cmap = plt.get_cmap("tab10")
color_lines = []
error_names = []
error_names_latex = []

reg_param_lines = []

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
#     n_alpha_points=150,
#     initial_cond=initial_condition,
#     var_hat_kwargs=pup,
# )

# alphas_L1, errors_L1, lambdas_L1 = optimal_lambda(
#     var_func_L2,
#     var_hat_func_L1_decorrelated_noise,
#     alpha_1=0.1,
#     alpha_2=10000,
#     n_alpha_points=150,
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
#     n_alpha_points=150,
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
#     n_alpha_points=150,
#     initial_cond=initial_condition,
#     var_hat_kwargs=pap,
# )

# # alphas_BO, errors_BO = load_file(**BO_settings)

# np.savetxt(
#     "./data/single_param_uncorrelated_bounded_fig_1.csv",
#     np.vstack((alphas_L2, errors_L2, lambdas_L2, errors_L1, lambdas_L1, errors_Huber,lambdas_Huber, huber_params, errors_BO)).T,
#     delimiter=",",
#     header="# alphas_L2,errors_L2,lambdas_L2,errors_L1,lambdas_L1,errors_Huber,lambdas_Huber,huber_params,errors_BO",
# )

# these are the data for the figure
data_fp = np.genfromtxt(
    "./data/FIGURE_1_data_uncorrelated_bounded.csv",
    delimiter=",",
    skip_header=1,
)

alphas_L2 = data_fp[:, 0]
errors_L2 = data_fp[:, 1]
lambdas_L2 = data_fp[:, 2]
errors_L1 = data_fp[:, 3]
lambdas_L1 = data_fp[:, 4]
errors_Huber = data_fp[:, 5]
lambdas_Huber = data_fp[:, 6]
huber_params = data_fp[:, 7]
errors_BO = data_fp[:, 8]

ax.plot(
    alphas_L2,
    errors_BO,
    label="BO",
    color="tab:red"
)

dat = np.genfromtxt(
    "./data/FIGURE_1_data_numerics_uncorrelated_bounded.csv",
    skip_header=1,
    delimiter=",",
)
alph_num = dat[:, 0]
err_mean_l2 = dat[:, 1]
err_std_l2 = dat[:, 2]
err_mean_l1 = dat[:, 3]
err_std_l1 = dat[:, 4]
err_mean_hub = dat[:, 5]
err_std_hub = dat[:, 6]

# dat_l1 = np.genfromtxt(
#     "./data/GOOD_beta_0.0_l1.csv",
#     skip_header=1,
#     delimiter=",",
# )
# alpha_l1 = dat_l1[:, 0]
# err_mean_l1 = dat_l1[:, 1]
# err_std_l1 = dat_l1[:, 2]

new_err_l1 = []
new_err_l2 = []
new_err_hub = []

for idx, e in enumerate(err_std_l2):
    new_err_l2.append(e / np.sqrt(10))

for idx, e in enumerate(err_std_l1):
    new_err_l1.append(e / np.sqrt(10))

for idx, e in enumerate(err_std_hub):
    new_err_hub.append(e / np.sqrt(10))

new_err_l2 = np.array(new_err_l2)
new_err_l1 = np.array(new_err_l1)
new_err_hub = np.array(new_err_hub)

# alphas_BO, errors_BO = load_file(**BO_settings)

ax.plot(alphas_L2, errors_L2, label=r"$\ell_2$", color="tab:blue")
ax.errorbar(
    alph_num,
    err_mean_l2,
    yerr=new_err_l2,
    color="tab:blue",
    linestyle="",
    elinewidth=0.75,
    markerfacecolor="none",
    markeredgecolor="tab:blue",
    marker="o",
    markersize=1.0,
)

ax.plot(
    alphas_L2,
    errors_L1,
    label=r"$\ell_1$",
    color="tab:green"
    # linewidth=0.5
)
ax.errorbar(
    alph_num,
    err_mean_l1,
    yerr=new_err_l1,
    color="tab:green",
    linestyle="",
    elinewidth=0.75,
    markerfacecolor="none",
    markeredgecolor="tab:green",
    marker="o",
    markersize=1.0,
)

ax.plot(alphas_L2, errors_Huber, label="Huber", color="tab:orange")
ax.errorbar(
    alph_num,
    err_mean_hub,
    yerr=new_err_hub,
    color="tab:orange",
    linestyle="",
    elinewidth=0.75,
    markerfacecolor="none",
    markeredgecolor="tab:orange",
    marker="o",
    markersize=1.0,
)

# --- plateaus ---
params = {
    "delta_small": delta_small,
    "delta_large": delta_large,
    "percentage": float(p),
    "beta": beta,
}

while True:
    m = 0.89 * np.random.random() + 0.1
    q = 0.89 * np.random.random() + 0.1
    sigma = 0.89 * np.random.random() + 0.1
    if np.square(m) < q + delta_small * q and np.square(m) < q + delta_large * q:
        initial_condition = [m, q, sigma]
        break

_, _, aaa = _find_optimal_reg_param_and_huber_parameter_gen_error(
    1000000,
    var_hat_func_Huber_decorrelated_noise,
    initial_condition,
    params,
    [0.01, 1e-4],
)

val_plateau_huber = p**2 * (plateau_H(delta_small, delta_large, p, aaa, x=errors_Huber[-1] / p**2)) ** 2
val_plateau_L1 = p**2 * (plateau_L1(delta_small, delta_large, p, x=errors_L1[-1] / p**2)) ** 2

ax.axhline(y=p**2, xmin=0.0, xmax=1, linestyle="dashed", color="tab:blue", alpha=0.75)
ax.axhline(
    y=np.abs(val_plateau_huber),
    xmin=0.0,
    xmax=1,
    linestyle="dashed",
    color="tab:orange",
    alpha=0.75,
)
ax.axhline(
    y=np.abs(val_plateau_L1), xmin=0.0, xmax=1, linestyle="dashed", color="tab:green", alpha=0.75
)

# ax.set_ylabel(r"$E_{\text{gen}}$", labelpad=2.0)
# ax.set_xlabel(r"$\alpha$", labelpad=2.0)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim([0.1, 10000])
ax.set_ylim([0.005, 1.9])
# ax.legend(loc="upper right", handlelength=1.0)

ax.tick_params(axis="y", pad=2.0)
ax.tick_params(axis="x", pad=2.0)

if save:
    pu.save_plot(
        fig,
        "FIGURE_1_center",
    )

# plt.show()

tuple_size = pu.set_size(width, fraction=0.50)

fig_2, ax_2 = plt.subplots(1, 1, figsize=(multiplier*tuple_size[0],second_multiplier*multiplier*tuple_size[1]))
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
ax_2.plot(alphas_L2, huber_params, label=r"$a_{\text{opt}}$ Huber", color='tab:gray', linestyle='solid')

# ax_2.set_ylabel(r"$a_{\text{opt}}$", labelpad=2.0)
# ax_2.set_xlabel(r"$\alpha$", labelpad=0.0)
ax_2.set_xscale("log")
# ax_2.set_yscale("log")
ax_2.set_xlim([0.1, 10000])
# ax_2.set_ylim([0.0, 1.7])
ax_2.grid(zorder=20)
# leg = ax_2.legend(loc="upper right", handlelength=1.0)

small_value = 1e-8

final_idx_Hub = 1
final_idx_L1 = 1
final_idx_L2 = 1
for idx in range(len(alphas_L2)):
    if lambdas_Huber[idx] >= small_value:
        final_idx_Hub = idx+1

    if lambdas_L1[idx] >= small_value:
        final_idx_L1 = idx+1

    if lambdas_L2[idx] >= small_value:
        final_idx_L2 = idx+1

# ax_2.axvline(x=alphas_L2[final_idx_L2], ymin=0, ymax=1, linestyle="dashed", color='tab:blue', alpha=0.75)
# ax_2.axvline(x=alphas_L2[final_idx_L1], ymin=0, ymax=1, linestyle="dashed", color='tab:green', alpha=0.75)
# ax_2.axvline(x=alphas_L2[final_idx_Hub], ymin=0, ymax=1, linestyle="dashed", color='tab:orange', alpha=0.75)

ax_2.tick_params(axis="y", pad=2.0)
ax_2.tick_params(axis="x", pad=2.0)

if save:
    pu.save_plot(
        fig_2,
        "FIGURE_1_center_parameters",
    )

# plt.show()
