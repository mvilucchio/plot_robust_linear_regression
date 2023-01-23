import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
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

save = True
experimental_points = True
width = 1.0 * 458.63788
# width = 398.3386
random_number = np.random.randint(100)

alpha_cut = 10.0
delta_small = 0.1
delta_large = 5.0
beta = 0.0

pu.initialization_mpl()

tuple_size = pu.set_size(width, fraction=0.49)

fig, axMain = plt.subplots(1, 1, figsize=tuple_size)
fig.subplots_adjust(left=0.2)
fig.subplots_adjust(bottom=0.2)
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

BETWEEN = 0.1
N = 50
# epsilons = np.linspace(0.0, 0.5, N)

epsilons_log = np.logspace(-4,np.log10(BETWEEN), int(N/2))
epsilons_lin = np.linspace(BETWEEN,0.5,int(N/2))
l2_err_log = np.empty(len(epsilons_log))
l1_err_log = np.empty(len(epsilons_log))
huber_err_log = np.empty(len(epsilons_log))
bo_err_log = np.empty(len(epsilons_log))

for idx, eps in enumerate(epsilons_log):
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

    l2_err_log[idx], _ = _find_optimal_reg_param_gen_error(
        alpha_cut,
        var_func_L2,
        var_hat_func_L2_decorrelated_noise,
        initial_condition,
        params,
        0.5,
    )
    print("done l2 {}".format(idx))

    l1_err_log[idx], _ = _find_optimal_reg_param_gen_error(
        alpha_cut,
        var_func_L2,
        var_hat_func_L1_decorrelated_noise,
        initial_condition,
        params,
        0.5,
    )
    print("done l1 {}".format(idx))

    huber_err_log[idx], _, _ = _find_optimal_reg_param_and_huber_parameter_gen_error(
        alpha_cut,
        var_hat_func_Huber_decorrelated_noise,
        initial_condition,
        params,
        [0.5, 0.1],
    )
    print("done hub {}".format(idx))

    if eps == 0.0:
        ppp = {
            "delta": delta_small,
        }
        m, q, sigma = fp._find_fixed_point(
            alpha_cut, var_func_BO, var_hat_func_BO_single_noise, 1.0, initial_condition, ppp
        )
    else:
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
    bo_err_log[idx] = 1 - 2 * m + q

    print("done bo {}".format(idx))

l2_err_lin = np.empty(len(epsilons_lin))
l1_err_lin = np.empty(len(epsilons_lin))
huber_err_lin = np.empty(len(epsilons_lin))
bo_err_lin = np.empty(len(epsilons_lin))

for idx, eps in enumerate(epsilons_lin):
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

    l2_err_lin[idx], _ = _find_optimal_reg_param_gen_error(
        alpha_cut,
        var_func_L2,
        var_hat_func_L2_decorrelated_noise,
        initial_condition,
        params,
        0.5,
    )
    print("done l2 {}".format(idx))

    l1_err_lin[idx], _ = _find_optimal_reg_param_gen_error(
        alpha_cut,
        var_func_L2,
        var_hat_func_L1_decorrelated_noise,
        initial_condition,
        params,
        0.5,
    )
    print("done l1 {}".format(idx))

    huber_err_lin[idx], _, _ = _find_optimal_reg_param_and_huber_parameter_gen_error(
        alpha_cut,
        var_hat_func_Huber_decorrelated_noise,
        initial_condition,
        params,
        [0.5, 0.1],
    )
    print("done hub {}".format(idx))

    if eps == 0.0:
        ppp = {
            "delta": delta_small,
        }
        m, q, sigma = fp._find_fixed_point(
            alpha_cut, var_func_BO, var_hat_func_BO_single_noise, 1.0, initial_condition, ppp
        )
    else:
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
    bo_err_lin[idx] = 1 - 2 * m + q

    print("done bo {}".format(idx))


# axMain = plt.subplot(111)
# axMain.plot(np.sin(xdomain), xdomain)
axMain.plot(epsilons_lin, l2_err_lin, label=r"$\ell_2$")
axMain.plot(epsilons_lin, l1_err_lin, label=r"$\ell_1$")
axMain.plot(epsilons_lin, huber_err_lin, label="Huber")
axMain.plot(epsilons_lin, bo_err_lin, label="BO")

axMain.set_xscale('linear')
axMain.set_xlim((BETWEEN, 0.5))
axMain.spines['left'].set_visible(False)
axMain.yaxis.set_ticks_position('right')
axMain.yaxis.set_visible(False)
axMain.set_yscale("log")



divider = make_axes_locatable(axMain)
axLin = divider.append_axes("left", size=1.0, pad=0, sharey=axMain)
axLin.set_xscale('log')
axLin.set_xlim((0.0001, BETWEEN))
# axLin.plot(np.sin(xdomain), xdomain)
axLin.plot(epsilons_log, l2_err_log, label=r"$\ell_2$")
axLin.plot(epsilons_log, l1_err_log, label=r"$\ell_1$")
axLin.plot(epsilons_log, huber_err_log, label="Huber")
axLin.plot(epsilons_log, bo_err_log, label="BO")

axLin.spines['right'].set_visible(False)
axLin.yaxis.set_ticks_position('left')
plt.setp(axLin.get_xticklabels(), visible=True)

# ax2 = axLin.twinx()
# ax2.spines['left'].set_visible(False)
# ax2.tick_params(axis='y',which='both',labelright='off')

# axLin.set_xlabel(r"$\epsilon$")
axLin.set_ylabel(r"$E_{\text{gen}}$")

# ax.plot(epsilons, l2_err, label=r"$\ell_2$")
# ax.plot(epsilons, l1_err, label=r"$\ell_1$")
# ax.plot(epsilons, huber_err, label="Huber")
# ax.plot(epsilons, bo_err, label="BO")
# ax.set_ylabel(r"$E_{\text{gen}}$")
# ax.set_xlabel(r"$\epsilon$")
# ax.set_xscale("log")
# ax.set_yscale("log")
# ax.set_xlim([0.0001, 0.51])
# ax.set_ylim([0.009, 1.5])
# ax.legend(ncol=2)

if save:
    pu.save_plot(
        fig,
        "double_axis_sweep_eps_fixed_delta_{:.2f}_beta_{:.2f}_alpha_cut_{:.2f}".format(delta_large, beta, alpha_cut),
    )

plt.show()