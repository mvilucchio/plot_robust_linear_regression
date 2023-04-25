import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import src.plotting_utils as pu
from scipy.signal import find_peaks

from robust_regression.fixed_point_equations.fpeqs import fixed_point_finder
from robust_regression.fixed_point_equations.fpe_projection_denoising import (
    var_func_projection_denoising,
)
from robust_regression.aux_functions.stability_functions import (
    stability_l1_l2,
    stability_huber,
    stability_ridge,
)
from robust_regression.fixed_point_equations.fpe_L2_loss import var_hat_func_L2_decorrelated_noise
from robust_regression.fixed_point_equations.fpe_L1_loss import var_hat_func_L1_decorrelated_noise
from robust_regression.fixed_point_equations.fpe_L2_regularization import var_func_L2
from robust_regression.aux_functions.training_errors import (
    training_error_l2_loss,
    training_error_l1_loss,
)
from robust_regression.utils.errors import ConvergenceError
from robust_regression.aux_functions.misc import damped_update

save = True
width = 1.0 * 458.63788

pu.initialization_mpl()

tuple_size = pu.set_size(width, fraction=0.50)

multiplier = 0.9
second_multiplier = 0.7

delta_in, delta_out, percentage, beta = 1.0, 5.0, 0.3, 0.0
reg_params = [-0.06, -0.03, 0.0, 0.03]
n_reg_params = len(reg_params)
alpha = 2.0

color_zero = (0.0, 0.0, 0.0, 0.8)

# beginning of script

fig, ax = plt.subplots(
    1, 1, figsize=(multiplier * tuple_size[0], 0.75 * multiplier * tuple_size[0])
)
fig.subplots_adjust(left=0.16)
fig.subplots_adjust(bottom=0.16)
fig.subplots_adjust(top=0.97)
fig.subplots_adjust(right=0.97)


blend = 1.0
max_iter = 100000
min_iter = 100
abs_tol = 1e-8

delta_in, delta_out, percentage, beta = 1.0, 5.0, 0.3, 0.0
alpha = 2.0

N = 5000
qs = np.logspace(-1, 3, N)
training_error = np.empty_like(qs)
ms = np.empty_like(qs)
sigmas = np.empty_like(qs)
m_hats = np.empty_like(qs)
q_hats = np.empty_like(qs)
sigma_hats = np.empty_like(qs)

q = qs[0]
while True:
    m = 10 * np.random.random() + 0.01
    sigma = 10 * np.random.random() + 0.01
    if np.square(m) < q + delta_in * q and np.square(m) < q + delta_out * q:
        break

for idx, q in enumerate(qs):
    try:
        iter_nb = 0
        err = 100.0
        while err > abs_tol or iter_nb < min_iter:
            m_hat, q_hat, sigma_hat = var_hat_func_L1_decorrelated_noise(
                m, q, sigma, alpha, delta_in, delta_out, percentage, beta
            )
            new_m, new_q, new_sigma = var_func_projection_denoising(m_hat, q_hat, sigma_hat, q)

            err = max([abs(new_m - m), abs(new_sigma - sigma)])

            m = damped_update(new_m, m, blend)
            sigma = damped_update(new_sigma, sigma, blend)

            iter_nb += 1
            if iter_nb > max_iter:
                raise ConvergenceError("fixed_point_finder", iter_nb)

        ms[idx] = m
        sigmas[idx] = sigma
        m_hats[idx] = m_hat
        sigma_hats[idx] = sigma_hat
        q_hats[idx] = q_hat

        training_error[idx] = training_error_l1_loss(
            m, q, sigma, delta_in, delta_out, percentage, beta
        )
    except (ConvergenceError, ValueError) as e:
        ms[idx:] = np.nan
        sigmas[idx:] = np.nan
        m_hats[idx:] = np.nan
        sigma_hats[idx:] = np.nan
        q_hats[idx:] = np.nan
        training_error[idx:] = np.nan
        break

min_idx = np.argmin(training_error)

m_true, q_true, sigma_true = fixed_point_finder(
    var_func_projection_denoising,
    var_hat_func_L1_decorrelated_noise,
    (ms[min_idx], qs[min_idx], sigmas[min_idx]),
    {"q_fixed": qs[min_idx]},
    {
        "alpha": alpha,
        "delta_in": delta_in,
        "delta_out": delta_out,
        "percentage": percentage,
        "beta": beta,
    },
)

m_hat_true, q_hat_true, sigma_hat_true = var_hat_func_L1_decorrelated_noise(
    m_true, q_true, sigma_true, alpha, delta_in, delta_out, percentage, beta
)

training_error_true = training_error_l1_loss(
    m_true, q_true, sigma_true, delta_in, delta_out, percentage, beta
)

print("q_true = ", q_true)

# plt.axhline(training_error_true, linestyle="--", color=color, alpha=0.5)
# plt.axvline(q_true, linestyle="--", color=color, alpha=0.5)

RdYlGn = plt.get_cmap("RdYlGn")
for reg_param in reg_params:
    color = RdYlGn((reg_param - np.amin(reg_params)) / 0.12)
    if reg_param == 0.0:
        color = color_zero

    peaks = find_peaks(-(training_error + reg_param / (2 * alpha) * qs))
    for p in peaks[0]:
        ax.axvline(qs[p], linestyle="--", color=color, alpha=0.5)
        ax.scatter(
            qs[p], training_error[p] + reg_param / (2 * alpha) * qs[p], marker=".", color=color
        )

    ax.plot(
        qs,
        training_error + reg_param / (2 * alpha) * qs,
        color=color,
        label=r"$\lambda$ = {:>5.2f}".format(reg_param),
    )

# color of the RS stable region

# ax.plot(qs, stability_l1_l2(ms, qs, sigmas, alpha, 1.0, delta_in, delta_out, percentage, beta), label="stability")

ax.set_ylabel(r"$\epsilon_t$")
ax.set_xlabel(r"$q$", labelpad=0)
ax.set_xscale("log")
# ax.legend()
ax.set_ylim([-0.0, 5])
ax.set_xlim([0.1, 1000])
ax.grid(zorder=20)

if save:
    pu.save_plot(
        fig,
        "FIGURE_4_left_minima_negative_lambda",
    )


# for the colorbar
fig_c, ax_c = plt.subplots(
    1, 1, figsize=(0.3 * multiplier * tuple_size[0], 0.75 * multiplier * tuple_size[0])
)
fig_c.subplots_adjust(left=0.05)
fig_c.subplots_adjust(bottom=0.16)
fig_c.subplots_adjust(top=0.97)
fig_c.subplots_adjust(right=0.2)

colors_list = []
for reg_param in reg_params:
    if reg_param != 0.0:
        colors_list.append(RdYlGn((reg_param - np.amin(reg_params)) / 0.12))
    else:
        colors_list.append(color_zero)

cmap = mpl.colors.LinearSegmentedColormap.from_list(
    "Custom cmap",
    colors_list,
    n_reg_params,
)

step_reg_params = reg_params[1] - reg_params[0]
bounds = np.array(reg_params) - 0.5 * step_reg_params
bounds = np.append(bounds, reg_params[-1] + 0.5 * step_reg_params)

norm = mpl.colors.BoundaryNorm(bounds, n_reg_params)
color_bar = mpl.colorbar.ColorbarBase(
    ax_c,
    cmap=cmap,
    norm=norm,
    ticks=bounds,
    boundaries=bounds,
)

tick_locs = reg_params
color_bar.set_ticks(tick_locs)
color_bar.set_ticklabels(tick_locs)
color_bar.set_label(r"$\lambda$", rotation=0, labelpad=20)

if save:
    pu.save_plot(
        fig_c,
        "FIGURE_4_left_colorbar_minima_negative_lambda",
    )


plt.show()
