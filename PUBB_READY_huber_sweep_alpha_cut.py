import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import numpy as np
from src.numerical_functions import q_integral_BO_decorrelated_noise

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
width = 2.0 * 458.63788
# width = 398.3386
random_number = np.random.randint(100)

alpha_cut = 10
delta_small = 0.1
delta_large = 5.0
beta = 0.0
eps = 0.00001

pu.initialization_mpl()
tuple_size = pu.set_size(width, fraction=0.49)
fig, ax = plt.subplots(1, 1, figsize=tuple_size)
fig.subplots_adjust(left=0.15)
fig.subplots_adjust(bottom=0.15)
fig.subplots_adjust(top=0.96)
fig.subplots_adjust(right=0.96)

N = 150
# reg_param
X = np.logspace(-2, 0, N)
# huber parram
Y = np.logspace(0, 4, N)
# # reg_param
# X = np.linspace(0.0, 1.0, N)
# # huber parram
# Y = np.linspace(0.0, 1.0, N)


XX, YY = np.meshgrid(X, Y)
# R = np.sqrt(X**2 + Y**2)
# Z = np.sin(R)
Z = np.empty_like(XX)

while True:
    m = 0.89 * np.random.random() + 0.1
    q = 0.89 * np.random.random() + 0.1
    sigma = 0.89 * np.random.random() + 0.1
    if np.square(m) < q + delta_small * q and np.square(m) < q + delta_large * q:
        initial_condition = [m, q, sigma]
        break

min_val = 10
max_val = -1
min_reg_p = 0
min_a = 0

for idx, reg_par in enumerate(X):
    print(idx)
    for jdx, a_hub in enumerate(Y):
        params = {
            "delta_small": delta_small,
            "delta_large": delta_large,
            "percentage": float(eps),
            "beta": beta,
            "a": a_hub,
        }

        m, q, _ = fp._find_fixed_point(
            alpha_cut,
            var_func_L2,
            var_hat_func_Huber_decorrelated_noise,
            reg_par,
            initial_condition,
            params,
        )
        Z[idx, jdx] = 1 - 2 * m + q

        if Z[idx, jdx] <= min_val:
            print(Z[idx, jdx])
            min_val = Z[idx, jdx]
            min_reg_p = idx
            min_a = jdx

        if Z[idx, jdx] >= max_val:
            max_val = Z[idx, jdx]

# min_val -= 0.1
# max_val += 0.1

# Plot the surface.
# surf = ax.plot_surface(XX, YY, Z, cmap=cm.coolwarm, linewidth=0, antialiased=True)

# Customize the z axis.
# ax.set_zlim(-1.01, 1.01)
# ax.zaxis.set_major_locator(LinearLocator(10))
# A StrMethodFormatter is used automatically
# ax.zaxis.set_major_formatter("{x:.02f}")

# Add a color bar which maps values to colors.
# fig.colorbar(surf, shrink=0.5, aspect=5)
N_LINES = 30
cs = ax.contourf(
    XX, YY, Z, levels=np.logspace(np.log10(min_val), np.log10(max_val), N_LINES)
)  # locator=ticker.LogLocator(), , levels=np.linspace(0.01,5,30)
ax.contour(
    XX, YY, Z, levels=np.logspace(np.log10(min_val), np.log10(max_val), N_LINES), colors='black', alpha=0.7, linewidths=0.5
)
ax.plot(XX[min_reg_p,min_a], YY[min_reg_p,min_a], marker="x", color='red')

# Alternatively, you can manually set the levels
# and the norm:
# lev_exp = np.arange(np.floor(np.log10(z.min())-1),
#                    np.ceil(np.log10(z.max())+1))
# levs = np.power(10, lev_exp)
# cs = ax.contourf(X, Y, z, levs, norm=colors.LogNorm())

# cbar = fig.colorbar(cs, pad=0.02)
# cbar.ax.set_ylabel(r"$E_{\text{gen}}$", rotation=90, labelpad=+1)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$\lambda$")
ax.set_ylabel(r"$a$")

if save:
    pu.save_plot(
        fig,
        "huber_3d_fixed_delta_{:.2f}_eps_{:.2f}_beta_{:.2f}_alpha_cut_{:.2f}".format(
            delta_large, eps, beta, alpha_cut
        ),
    )

plt.show()
