import numpy as np

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

SMALLEST_REG_PARAM = None
SMALLEST_HUBER_PARAM = 1e-8
MAX_ITER = 2500
XATOL = 1e-9
FATOL = 1e-9

save = True
width = 1.0 * 458.63788

alpha_cut = 10.0
delta_small = 1.0
beta = 0.0
eps = 0.3
UPPER_BOUND = 100.0
LOWER_BOUND = 0.01

pu.initialization_mpl()

tuple_size = pu.set_size(width, fraction=0.50)

multiplier = 0.9
second_multiplier = 0.7

fig, ax = plt.subplots(1, 1, figsize=(tuple_size[0], tuple_size[0]))
fig.subplots_adjust(left=0.16)
fig.subplots_adjust(bottom=0.16)
fig.subplots_adjust(top=0.97)
fig.subplots_adjust(right=0.97)

# -----------

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

# -------------------

N = 20
# delta_larges = np.linspace(0.04, UPPER_BOUND, N)
delta_larges = np.logspace(np.log10(LOWER_BOUND), np.log10(UPPER_BOUND), N)
# epsilons = np.logspace(-3, np.log10(0.5), N)
epsilons = np.linspace(0.01, 0.5, N)

l2_err = np.empty((N,N))
l2_lambda = np.empty((N,N))
huber_err = np.empty((N,N))
hub_lambda = np.empty((N,N))
a_hub = np.empty((N,N))

difference_hub_l2 = np.empty((N,N))

for jdx, eps in enumerate(epsilons):
    previous_aopt_hub = 1.0
    previous_rpopt_hub = 0.5
    previous_rpopt_l2 = 0.5
    previous_rpopt_l1 = 0.5
    for idx, delta_large in enumerate(delta_larges[::-1]):
        while True:
            m = 0.89 * np.random.random() + 0.1
            q = 0.89 * np.random.random() + 0.1
            sigma = 0.89 * np.random.random() + 0.1
            if np.square(m) < q + delta_small * q and np.square(m) < q + delta_large * q:
                initial_condition = [m, q, sigma]
                break

        params = {
            "delta_small": delta_small,
            "delta_large": delta_large,
            "percentage": float(eps),
            "beta": beta,
        }

        l2_err[jdx, N - idx - 1], l2_lambda[jdx,N - idx - 1] = _find_optimal_reg_param_gen_error(
            alpha_cut,
            var_func_L2,
            var_hat_func_L2_decorrelated_noise,
            initial_condition,
            params,
            previous_rpopt_l2,
        )
        previous_rpopt_l2 = l2_lambda[jdx,N - idx - 1]
        print("done l2 {}".format(jdx,N - idx - 1))

        (
            huber_err[jdx,N - idx - 1],
            hub_lambda[jdx,N - idx - 1],
            a_hub[jdx,N - idx - 1],
        ) = _find_optimal_reg_param_and_huber_parameter_gen_error(
            alpha_cut,
            var_hat_func_Huber_decorrelated_noise,
            initial_condition,
            params,
            [previous_rpopt_hub, previous_aopt_hub],
        )
        previous_rpopt_hub = hub_lambda[jdx,N - idx - 1]
        previous_aopt_hub = a_hub[jdx,N - idx - 1]
        print("done hub {}".format(jdx,N - idx - 1))

        difference_hub_l2[jdx,N - idx - 1] = l2_err[jdx,N - idx - 1] - huber_err[jdx,N - idx - 1]

np.savetxt(
    "./data/FIGURE_3_right_epsilons_delta_out.csv",
    np.vstack((epsilons,delta_larges)).T,
    delimiter=",",
    header="# epsilons, delta_larges",
)

np.savetxt(
    "./data/FIGURE_3_right_difference.csv",
    difference_hub_l2,
    delimiter=",",
)

# data_fp = np.genfromtxt(
#     "./data/sweep_delta_fig_4_bounded.csv",
#     delimiter=",",
#     skip_header=1,
# )

# delta_larges = data_fp[:, 0]
# l2_err = data_fp[:, 1]
# l2_lambda = data_fp[:, 2]
# l1_err = data_fp[:, 3]
# l1_lambda = data_fp[:, 4]
# huber_err = data_fp[:, 5]
# hub_lambda = data_fp[:, 6]
# a_hub = data_fp[:, 7]
# bo_err = data_fp[:, 8]

# data_fp = np.genfromtxt(
#     "./data/FIGURE_3_unbounded.csv",
#     delimiter=",",
#     skip_header=1,
# )

# delta_larges_ub = data_fp[:, 0]
# l2_err_ub = data_fp[:, 1]
# l2_lambda_ub = data_fp[:, 2]
# l1_err_ub = data_fp[:, 3]
# l1_lambda_ub = data_fp[:, 4]
# huber_err_ub = data_fp[:, 5]
# hub_lambda_ub = data_fp[:, 6]
# a_hub_ub = data_fp[:, 7]
# bo_err_ub = data_fp[:, 8]

# index_phase_trans = len(delta_larges_ub) - 1

# for idx, a_val in enumerate(a_hub_ub[::-1]):
#     if a_val <= 8.0:
#         index_phase_trans = len(delta_larges_ub) - 1 - idx

# ax.plot(delta_larges, l2_err_ub - huber_err_ub, label=r"$\ell_2$", color="tab:blue")
# ax.plot(delta_larges_ub, l1_err_ub, label=r"$\ell_1$", color="tab:green")
# ax.plot(delta_larges_ub, huber_err_ub, label="Huber", color="tab:orange")
# ax.plot(delta_larges_ub, bo_err_ub, label="BO", color="tab:red")

# ax.axvline(x=delta_larges[index_phase_trans], ymin=0.0, ymax=1.0, color="k", linestyle="dashed", alpha=0.75)

ax.imshow(difference_hub_l2, cmap='hot', interpolation='nearest')

# min_val = np.min(difference_hub_l2)
# max_val = np.max(difference_hub_l2)
# N_LINES = 30
# epseps, deltadelta_out = np.meshgrid(epsilons, delta_larges)
# cs = ax.contourf(
#     epseps, deltadelta_out, difference_hub_l2, levels=np.logspace(np.log10(min_val), np.log10(max_val), N_LINES)
# )  # locator=ticker.LogLocator(), , levels=np.linspace(0.01,5,30)
# ax.contour(
#     epseps, deltadelta_out, difference_hub_l2, levels=np.logspace(np.log10(min_val), np.log10(max_val), N_LINES), colors='black', alpha=0.7, linewidths=0.5
# )

# ax.set_xlabel(r"$\epsilon$", labelpad=2.0)
# ax.set_ylabel(r"$\Delta_\text{\tiny{OUT}}$", labelpad=2.0)
# # ax.set_xscale("log")
# ax.set_yscale("log")

# ax.set_xlim([LOWER_BOUND, UPPER_BOUND])
# ax.set_ylim([0.1, 1.3])
# ax.legend(ncol=2, handlelength=1.0)

# ax.tick_params(axis="y", pad=2.0)
# ax.tick_params(axis="x", pad=2.0)

if save:
    pu.save_plot(
        fig,
        "FIGURE_3_right_phase_diagram",
    )

plt.show()