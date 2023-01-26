import numpy as np

from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as ticker
import src.plotting_utils as pu

from scipy.optimize import minimize, curve_fit
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
experimental_points = True
width = 4/5 * 1.0 * 458.63788
random_number = np.random.randint(100)

alpha_cut = 10.0
delta_small = 0.1
delta_large = 5.0
beta = 0.0

pu.initialization_mpl()

multiplier = 0.9
second_multiplier = 0.7

tuple_size = pu.set_size(width, fraction=0.50)

fig, ax = plt.subplots(1, 1, figsize=(multiplier * tuple_size[0], 1/2 * multiplier * tuple_size[0]))
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
upper_bound = 0.5
UPPER_BOUND_AX = 0.5
LOWER_BOUND_AX = 1e-4

# N = 500
# epsilons = np.linspace(0.0, 0.5, N)
# # epsilons = np.logspace(-4, np.log10(upper_bound), N)
# # epsilons = np.logspace(-5, -3, N)
# l2_err = np.empty(len(epsilons))
# l2_lambda = np.empty(len(epsilons))
# l1_err = np.empty(len(epsilons))
# l1_lambda = np.empty(len(epsilons))
# huber_err = np.empty(len(epsilons))
# hub_lambda = np.empty(len(epsilons))
# a_hub = np.empty(len(epsilons))
# bo_err = np.empty(len(epsilons))

# SMALLEST_REG_PARAM = None

# previous_aopt_hub = 1.003256444782426149e+00
# previous_rpopt_hub = 9.970017587168589213e-02
# previous_rpopt_l2 = 9.978698670991027753e-02
# previous_rpopt_l1 = 3.891536556174015615e-01
# for idx, eps in enumerate(tqdm(epsilons)):
#     # print(eps)
#     while True:
#         m = 0.89 * np.random.random() + 0.1
#         q = 0.89 * np.random.random() + 0.1
#         sigma = 0.89 * np.random.random() + 0.1
#         if np.square(m) < q + delta_small * q and np.square(m) < q + delta_large * q:
#             initial_condition = [m, q, sigma]
#             break

#     params = {
#         "delta_small": delta_small,
#         "delta_large": delta_large,
#         "percentage": float(eps),
#         "beta": beta,
#     }

#     l2_err[idx], l2_lambda[idx] = _find_optimal_reg_param_gen_error(
#         alpha_cut,
#         var_func_L2,
#         var_hat_func_L2_decorrelated_noise,
#         initial_condition,
#         params,
#         previous_rpopt_l2,
#     )
#     previous_rpopt_l2 = l2_lambda[idx]
#     # print("done l2 {}".format(idx))

#     l1_err[idx], l1_lambda[idx] = _find_optimal_reg_param_gen_error(
#         alpha_cut,
#         var_func_L2,
#         var_hat_func_L1_decorrelated_noise,
#         initial_condition,
#         params,
#         previous_rpopt_l1,
#     )
#     previous_rpopt_l1 = l1_lambda[idx]
#     # print("done l1 {}".format(idx))

#     if eps < 0.001:
#         aaa = 10
#     else:
#         aaa = 1

#     (
#         huber_err[idx],
#         hub_lambda[idx],
#         a_hub[idx],
#     ) = _find_optimal_reg_param_and_huber_parameter_gen_error(
#         alpha_cut,
#         var_hat_func_Huber_decorrelated_noise,
#         initial_condition,
#         params,
#         [previous_rpopt_hub, previous_aopt_hub],
#     )
#     previous_rpopt_hub = hub_lambda[idx]
#     previous_aopt_hub = a_hub[idx]

#     if eps == 0.0:
#         ppp = {
#             "delta": delta_small,
#         }
#         m, q, sigma = fp._find_fixed_point(
#             alpha_cut, var_func_BO, var_hat_func_BO_single_noise, 1.0, initial_condition, ppp
#         )
#     else:
#         pup = {
#             "delta_small": delta_small,
#             "delta_large": delta_large,
#             "percentage": float(eps),
#             "beta": beta,
#         }
#         m, q, sigma = fp._find_fixed_point(
#             alpha_cut,
#             var_func_BO,
#             var_hat_func_BO_num_decorrelated_noise,
#             1.0,
#             initial_condition,
#             pup,
#         )
#     bo_err[idx] = 1 - 2 * m + q
#     # print("done bo {}".format(idx))

# np.savetxt(
#     "./data/FIGURE_2_unbounded.csv",
#     np.vstack(
#         (epsilons, l2_err, l2_lambda, l1_err, l1_lambda, huber_err, hub_lambda, a_hub, bo_err)
#     ).T,
#     delimiter=",",
#     header="# alphas_L2,errors_L2,lambdas_L2,errors_L1,lambdas_L1,errors_Huber,lambdas_Huber,huber_params,errors_BO",
# )

# these are the data for the figure
# unbounded
data_fp = np.genfromtxt(
    "./data/FIGURE_2_unbounded.csv",
    delimiter=",",
    skip_header=1,
)

epsilons = data_fp[:, 0]
l2_err = data_fp[:, 1]
l2_lambda = data_fp[:, 2]
l1_err = data_fp[:, 3]
l1_lambda = data_fp[:, 4]
huber_err = data_fp[:, 5]
hub_lambda = data_fp[:, 6]
a_hub = data_fp[:, 7]
bo_err = data_fp[:, 8]

ax.plot(epsilons, l2_err, label=r"$\ell_2$", color="tab:blue")
ax.plot(epsilons, l1_err, label=r"$\ell_1$", color="tab:green")
ax.plot(epsilons, huber_err, label="Huber", color="tab:orange")
ax.plot(epsilons, bo_err, label="BO", color="tab:red")

ax.set_ylabel(r"$E_{\text{gen}}$", labelpad=2.0)
ax.set_xlabel(r"$\epsilon$", labelpad=2.0)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim([LOWER_BOUND_AX, UPPER_BOUND_AX])
# ax.set_ylim([7e-6, 1.5])
# ax.legend(ncol=2, handlelength=1.0)

ax.tick_params(axis="y", pad=2.0)
ax.tick_params(axis="x", pad=2.0)

if save:
    pu.save_plot(
        fig,
        "FIGURE_2_left",
    )

# plt.show()

tuple_size = pu.set_size(width, fraction=0.50)

fig, ax = plt.subplots(
    1, 1, figsize=(multiplier * tuple_size[0], multiplier * second_multiplier * tuple_size[1])
)
fig.subplots_adjust(left=0.16)
fig.subplots_adjust(bottom=0.16)
fig.subplots_adjust(top=0.97)
fig.subplots_adjust(right=0.97)

ax.plot(epsilons, l2_lambda, label=r"$\lambda_{\text{opt}}\:\ell_2$", color="tab:blue")
ax.plot(epsilons, l1_lambda, label=r"$\lambda_{\text{opt}}\:\ell_1$", color="tab:green")
ax.plot(epsilons, hub_lambda, label=r"$\lambda_{\text{opt}}\:$ Huber", color="tab:orange")
ax.plot(epsilons, a_hub, label=r"$a_{\text{opt}}\:$Huber", color="tab:gray")

ax.set_ylabel(r"$(\lambda_{\rm opt}, a_{\rm opt})$", labelpad=0.0)
ax.set_xlabel(r"$\epsilon$", labelpad=2.0)
ax.set_xscale("log")
# ax.set_yscale("log")
ax.set_xlim([LOWER_BOUND_AX, UPPER_BOUND_AX])
# ax.set_ylim([-1, 2])
# ax.legend(ncol=2, handlelength=1.0)

ax.tick_params(axis="y", pad=3.0)
ax.tick_params(axis="x", pad=3.0)

ax.set_xticks([0.0001, 0.001, 0.01, 0.1, 0.5])
ax.set_xticklabels([r"$10^{-4}$", r"$10^{-3}$", r"$10^{-2}$", r"$10^{-1}$", r"$0.5$"])

if save:
    pu.save_plot(
        fig,
        "FIGURE_2_left_parameters",
    )

# plt.show()
