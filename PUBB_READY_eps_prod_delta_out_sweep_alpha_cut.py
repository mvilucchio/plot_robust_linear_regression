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

SMALLEST_REG_PARAM = 1e-7
SMALLEST_HUBER_PARAM = 1e-7
MAX_ITER = 2500
XATOL = 1e-8
FATOL = 1e-8

save = False
experimental_points = True
width = 1.2 * 458.63788
width = 398.3386
random_number = np.random.randint(100)

alpha_cut = 10.0
delta_small = 1.0
beta = 1.0
eps = 0.3
UPPER_BOUND = 0.01
LOWER_BOUND = 0.000001

pu.initialization_mpl()

tuple_size = pu.set_size(width, fraction=0.49)

fig, ax = plt.subplots(1, 1, figsize=tuple_size)
fig.subplots_adjust(left=0.2)
fig.subplots_adjust(bottom=0.23)
fig.subplots_adjust(top=0.99)
fig.subplots_adjust(right=0.96)

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
# # sweep over eps
# N = 100
# # delta_larges = np.linspace(0.04, UPPER_BOUND, N)
# delta_large = 10000
# epsilons = np.logspace(np.log10(LOWER_BOUND), np.log10(UPPER_BOUND), N)
# l2_err = np.empty(len(epsilons))
# l1_err = np.empty(len(epsilons))
# huber_err = np.empty(len(epsilons))
# bo_err = np.empty(len(epsilons))

# for idx, eps in enumerate(epsilons):
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

#     l2_err[idx], _ = _find_optimal_reg_param_gen_error(
#         alpha_cut,
#         var_func_L2,
#         var_hat_func_L2_decorrelated_noise,
#         initial_condition,
#         params,
#         0.5,
#     )
#     print("done l2 {}".format(idx))

#     l1_err[idx], _ = _find_optimal_reg_param_gen_error(
#         alpha_cut,
#         var_func_L2,
#         var_hat_func_L1_decorrelated_noise,
#         initial_condition,
#         params,
#         0.5,
#     )
#     print("done l1 {}".format(idx))

#     huber_err[idx], _, _ = _find_optimal_reg_param_and_huber_parameter_gen_error(
#         alpha_cut,
#         var_hat_func_Huber_decorrelated_noise,
#         initial_condition,
#         params,
#         [0.5, 0.1],
#     )
#     print("done hub {}".format(idx))

#     pup = {
#         "delta_small": delta_small,
#         "delta_large": delta_large,
#         "percentage": float(eps),
#         "beta": beta,
#     }
#     m, q, sigma = fp._find_fixed_point(
#         alpha_cut,
#         var_func_BO,
#         var_hat_func_BO_num_decorrelated_noise,
#         1.0,
#         initial_condition,
#         pup,
#     )
#     bo_err[idx] = 1 - 2 * m + q
#     print("done bo {}".format(idx))

# np.savetxt(
#     "./data/sweep_eps_prod_delta_out_fixed_eps_{:.2f}_beta_{:.2f}_alpha_cut_{:.2f}_delta_in_{:.2f}.csv".format(
#         eps, beta, alpha_cut, delta_small
#     ), 
#     np.vstack((epsilons * delta_large, l2_err, l1_err, huber_err, bo_err)).T, 
#     delimiter=",",
#     header="delta_large,l2,l1,Huber,BO"
# )

# x_val = epsilons * delta_large

# ------------------
# sweep over delta
N = 100
# delta_larges = np.linspace(0.04, UPPER_BOUND, N)
eps = 0.0001
delta_larges = np.logspace(0,3,N)
# delta_large = 10000
# epsilons = np.logspace(np.log10(LOWER_BOUND), np.log10(UPPER_BOUND), N)
l2_err = np.empty(len(delta_larges))
l1_err = np.empty(len(delta_larges))
huber_err = np.empty(len(delta_larges))
bo_err = np.empty(len(delta_larges))

for idx, delta_large in enumerate(delta_larges):
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

    l2_err[idx], _ = _find_optimal_reg_param_gen_error(
        alpha_cut,
        var_func_L2,
        var_hat_func_L2_decorrelated_noise,
        initial_condition,
        params,
        0.5,
    )
    print("done l2 {}".format(idx))

    l1_err[idx], _ = _find_optimal_reg_param_gen_error(
        alpha_cut,
        var_func_L2,
        var_hat_func_L1_decorrelated_noise,
        initial_condition,
        params,
        0.5,
    )
    print("done l1 {}".format(idx))

    huber_err[idx], _, _ = _find_optimal_reg_param_and_huber_parameter_gen_error(
        alpha_cut,
        var_hat_func_Huber_decorrelated_noise,
        initial_condition,
        params,
        [0.5, 0.1],
    )
    print("done hub {}".format(idx))

    pup = {
        "delta_small": delta_small,
        "delta_large": delta_large,
        "percentage": float(eps),
        "beta": beta,
    }
    m, q, sigma = fp._find_fixed_point(
        alpha_cut,
        var_func_BO,
        var_hat_func_BO_num_decorrelated_noise,
        1.0,
        initial_condition,
        pup,
    )
    bo_err[idx] = 1 - 2 * m + q
    print("done bo {}".format(idx))

# np.savetxt(
#     "./data/sweep_eps_prod_delta_out_fixed_eps_{:.2f}_beta_{:.2f}_alpha_cut_{:.2f}_delta_in_{:.2f}.csv".format(
#         eps, beta, alpha_cut, delta_small
#     ), 
#     np.vstack((eps * delta_larges, l2_err, l1_err, huber_err, bo_err)).T, 
#     delimiter=",",
#     header="delta_large,l2,l1,Huber,BO"
# )
x_vals = delta_large * eps

# out = np.genfromtxt("./data/sweep_delta_fixed_eps_0.30_beta_1.00_alpha_cut_10.00_delta_in_1.00.csv", skip_header=1, delimiter=',')

# delta_larges = out[:,0]
# l2_err = out[:,1]
# l1_err = out[:,2]
# huber_err = out[:,3]
# bo_err = out[:,4]

ax.plot(x_vals, l2_err, label=r"$\ell_2$")
ax.plot(x_vals, l1_err, label=r"$\ell_1$")
ax.plot(x_vals, huber_err, label="Huber")
ax.plot(x_vals, bo_err, label="BO")
ax.set_xlabel(r"$\epsilon\Delta_\text{\tiny{OUT}}$")
ax.set_ylabel(r"$E_{\text{gen}}$")
ax.set_xscale("log")
ax.set_yscale("log")
# ax.set_xlim([LOWER_BOUND, UPPER_BOUND])
# ax.set_ylim([0.1, 1.3])
ax.legend(ncol=2)

# labels = [item.get_text() for item in ax.get_xticklabels()]
# print(labels)
# # labels[2] =
# # ax.set_xticklabels([0.1, 1, 10, 100, 1000], [r"10^{-1}",r"10^{0}",r"$\alpha_{\text{cut}}$",r"10^{2}",r"10^{3}"])
# text = [m for m in ax.get_xticklabels()]
# positions = [m for m in ax.get_xticks()]

# xt = ax.get_xticks() 
# print(xt)
# xt = np.append(xt[1:], 0.1)

# xtl = xt.tolist()
# xtl[-1] = r"$\Delta_{\text{\tiny{IN}}}$"
# ax.set_xticks(xt)
# ax.set_xticklabels(xtl)

ax.set_xticks([0.01, 0.1, 1, 10,100])
ax.set_xticklabels([r"$10^{-2}$", r"$10^{-1}$",r"$\Delta_{\text{\tiny{IN}}}$", r"$10^{1}$", r"$10^{2}$"])

# ax.set_xticks([0.1, 1, 10,100])
# ax.set_xticklabels([r"$\Delta_{\text{\tiny{IN}}}$", r"$10^{0}$", r"$10^{1}$", r"$10^{2}$"])


if save:
    pu.save_plot(
        fig,
        "sweep_delta_fixed_epsilon_{:.2f}_beta_{:.2f}_alpha_cut_{:.2f}_delta_small_{:.2f}".format(eps, beta, alpha_cut, delta_small),
    )

plt.show()