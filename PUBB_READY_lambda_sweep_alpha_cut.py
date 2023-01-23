import numpy as np

from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as ticker
import matplotlib.cm as cm
import matplotlib.colors as colors
import src.plotting_utils as pu
from mpl_toolkits.axes_grid1 import make_axes_locatable

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
width = 1.7 * 458.63788
# width = 398.3386
random_number = np.random.randint(100)

alpha_cut = 0.1
delta_small = 0.1
delta_large = 5.0
beta = 0.0
eps=0.3

pu.initialization_mpl()

tuple_size = pu.set_size(width, fraction=0.49)

fig, ax = plt.subplots(1, 1, figsize=tuple_size)
fig.subplots_adjust(left=0.2)
fig.subplots_adjust(bottom=0.2)
fig.subplots_adjust(top=0.99)
fig.subplots_adjust(right=0.9)

def get_cmap(n, name="hsv"):
    return plt.cm.get_cmap(name, n)

# -------------------

N = 100
# epsilons = np.linspace(0.0, 0.5, N)
reg_params = np.logspace(-4, 2, N)
l2_err = np.empty(len(reg_params))
l1_err = np.empty(len(reg_params))

NN = 9
a_hubers = np.logspace(-2, 0, NN)
huber_err = np.empty((NN, N))
# huber_err_01 = np.empty(len(reg_params))
# huber_err_10 = np.empty(len(reg_params))
# huber_err_100 = np.empty(len(reg_params))
bo_err = np.empty(len(reg_params))

for idx, reg_par in enumerate(reg_params):
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

    m, q, _ = fp._find_fixed_point(
        alpha_cut,
        var_func_L2,
        var_hat_func_L2_decorrelated_noise,
        reg_par,
        initial_condition,
        params,
    )
    l2_err[idx] = 1 - 2 * m + q

    print("done l2 {}".format(idx))

    m, q, _ = fp._find_fixed_point(
        alpha_cut,
        var_func_L2,
        var_hat_func_L1_decorrelated_noise,
        reg_par,
        initial_condition,
        params,
    )
    l1_err[idx] = 1 - 2 * m + q

    print("done l1 {}".format(idx))

    for idx_a, a in enumerate(a_hubers):
        params = {
            "delta_small": delta_small,
            "delta_large": delta_large,
            "percentage": float(eps),
            "beta": beta,
            "a": a
        }

        m, q, _ = fp._find_fixed_point(
            alpha_cut,
            var_func_L2,
            var_hat_func_Huber_decorrelated_noise,
            reg_par,
            initial_condition,
            params,
        )
        huber_err[idx_a][idx] = 1 - 2 * m + q

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
bo_err_tmp = 1 - 2 * m + q

for idx in range(len(bo_err)):
    bo_err[idx] = bo_err_tmp

print("done bo")

header_str = "reg_params,l2,l1,"
for idx_a, a in enumerate(a_hubers):
    header_str += "Huber {:.2f},".format(a)
header_str += "BO"

np.savetxt(
    "./data/sweep_lambda_fixed_delta_{:.2f}_eps_{:.2f}_beta_{:.2f}_alpha_cut_{:.2f}.csv".format(
        delta_large, eps, beta, alpha_cut
    ), 
    np.vstack((reg_params, l2_err, l1_err, huber_err, bo_err)).T, 
    delimiter=",",
    header=header_str
)

print("done bo {}".format(idx))

ax.plot(reg_params, l2_err, label=r"$\ell_2$")
ax.plot(reg_params, l1_err, label=r"$\ell_1$")

colormap = get_cmap(int(len(a_hubers)*1.7), name="Greens")
for idx_a, a in enumerate(a_hubers):
    ax.plot(reg_params, huber_err[idx_a], label="Huber {:.2f}".format(a), color=colormap(idx_a))

ax.plot(reg_params, bo_err, label="BO", color='red')
ax.set_ylabel(r"$E_{\text{gen}}$")
ax.set_xlabel(r"$\lambda$")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim([0.0001, 100])
# ax.set_ylim([bo_err_tmp, 1.5])
ax.legend(ncol=2)


# fig.colorbar(cm.ScalarMappable(norm=colors.Normalize(vmin=0.01, vmax=1.5, clip=False), cmap=get_cmap(int(len(a_hubers)*1.7), name="Greens")), cax=ax, orientation='vertical')

# fig.colorbar(ticks=range(6), label='digit value')

divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='10%', pad=0.05)
cbar = fig.colorbar(
    cm.ScalarMappable(
        norm=colors.Normalize(vmin=0.01, vmax=1.5, clip=False), 
        cmap=get_cmap(int(len(a_hubers)*1.7), name="Greens")
    ), 
    cax=cax, 
    orientation='vertical'
)
# cbar.ax.get_yaxis().labelpad = 50
cbar.ax.set_ylabel(r'$a$', rotation=90)

# ax.set_yticks([bo_err_tmp, 0.1, 1.0])
# ax.set_yticklabels([r"$E_{\text{gen}}^{\text{BO}}$", r"$10^{-1}$", r"$10^{0}$"])


if save:
    pu.save_plot(
        fig,
        "sweep_lambda_fixed_delta_{:.2f}_eps_{:.2f}_beta_{:.2f}_alpha_cut_{:.2f}".format(
            delta_large, eps, beta, alpha_cut
        ),
    )

plt.show()
