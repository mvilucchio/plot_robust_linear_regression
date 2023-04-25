import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import src.plotting_utils as pu
import os
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
from robust_regression.fixed_point_equations.fpe_Huber_loss import var_hat_func_Huber_decorrelated_noise
from robust_regression.fixed_point_equations.fpe_L2_regularization import var_func_L2
from robust_regression.aux_functions.training_errors import (
    training_error_l2_loss,
    training_error_l1_loss,
    training_error_huber_loss,
)
from robust_regression.utils.errors import ConvergenceError
from robust_regression.aux_functions.misc import damped_update

save = False
width = 1.0 * 458.63788

blend = 1.0
max_iter = 100000
min_iter = 100
abs_tol = 1e-8
N = 1000

pu.initialization_mpl()

tuple_size = pu.set_size(width, fraction=0.50)

multiplier = 0.9
second_multiplier = 0.7

delta_in, delta_out, percentage, beta, a = 1.0, 5.0, 0.3, 0.0, 1.0
reg_param = -1.5
alpha = 30.0

color_zero = (0.0, 0.0, 0.0, 0.8)

# beginning of script

# loading of AMP data from cluster
fig, axs = plt.subplots(
    nrows=2,
    ncols=2,
    # sharex=True,
    # sharey=True,
    figsize=(multiplier * tuple_size[0], 0.75 * multiplier * tuple_size[0]),
    gridspec_kw={"hspace": 0, "wspace": 0, "width_ratios": [1, 0.5 * (1 + np.sqrt(5))]},
)
fig.subplots_adjust(left=0.16)
fig.subplots_adjust(bottom=0.16)
fig.subplots_adjust(top=0.97)
fig.subplots_adjust(right=0.97)

# load the data from the file


# compute the training error landscape
N = 1000
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
            m_hat, q_hat, sigma_hat = var_hat_func_Huber_decorrelated_noise(
                m, q, sigma, alpha, delta_in, delta_out, percentage, beta, a
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

        training_error[idx] = training_error_huber_loss(m, q, sigma, delta_in, delta_out, percentage, beta, a)
    except (ConvergenceError, ValueError) as e:
        ms[idx:] = np.nan
        sigmas[idx:] = np.nan
        m_hats[idx:] = np.nan
        sigma_hats[idx:] = np.nan
        q_hats[idx:] = np.nan
        training_error[idx:] = np.nan
        break


xs = np.linspace(0, 1, N)
iter_nb = np.arange(0, 1000)

q_min, q_max = np.amin(qs), np.amax(qs)
e_train_min, e_train_max = -3, 4
e_gen_min, e_gen_max = 0.5, 11

# plot of generalization error as function of GD iterations
axs[0,1].set_ylim([e_gen_min, e_gen_max])
axs[0, 1].set_xscale("log")
axs[0, 1].set_yscale("log")
# axs[0, 1].set_ylim([0.1, 1000])
axs[0, 1].tick_params(axis="y", which="major", length=0)
# axs[0, 1].set_yticklabels([])
axs[0, 1].set_ylabel(r"$E_{\text{gen}}$", labelpad=0)

# find the peaks of the training error landscape
peaks, _ = find_peaks(-(training_error + reg_param / (2 * alpha) * qs))

# plot of training error landscape
axs[1, 0].plot(training_error + reg_param / (2 * alpha) * qs, qs)
axs[1, 0].scatter(training_error[peaks] + reg_param / (2 * alpha) * qs[peaks], qs[peaks], color="red", zorder=10)
axs[1, 0].plot(
    [training_error[peaks] + reg_param / (2 * alpha) * qs[peaks], e_train_min],
    [qs[peaks], qs[peaks]],
    color="red",
    linestyle="--",
)
axs[1, 1].axhline(y=qs[peaks], color="red", linestyle="--")

# iterate over all the files in the folder ./data/vanilla_GD_lr005
for idx, fname in enumerate(os.listdir("./data/vanillaGD-huber/data_GD_lr001")):
    if not fname.endswith(".csv"):
        continue
    dat = np.loadtxt("./data/vanillaGD-huber/data_GD_lr001/" + fname, delimiter=",")

    its, gen_error, norm = dat[:, 0], dat[:, 1], dat[:, 2]
    color = next(plt.gca()._get_lines.prop_cycler)["color"]

    axs[1, 1].plot(its, norm, color=color, marker=".")
    axs[0, 1].plot(its, gen_error, color=color, marker=".")

    y0 = np.interp(norm[0], qs, training_error + reg_param / (2 * alpha) * qs)
    # Plot the vertical line starting from (x0, 0) and ending at (x0, y0)
    axs[1, 0].plot([y0, e_train_min], [norm[0], norm[0]], color=color, linestyle="--")

axs[1, 0].set_xlim([e_train_min, e_train_max])
axs[1, 0].set_ylim([q_min, q_max])
axs[1, 0].set_yscale("log")
axs[1, 0].set_xlim(axs[1, 0].get_xlim()[::-1])
axs[1, 0].set_xlabel(r"$\epsilon_t$", labelpad=0)
axs[1, 0].set_ylabel(r"$q$", labelpad=0)

# plot of vector norm as function of GD iterations
axs[1, 1].set_xscale("log")
axs[1, 1].set_yscale("log")
axs[1, 1].set_ylim([q_min, q_max])
axs[1, 1].tick_params(axis="y", which="both", length=0)
axs[1, 1].set_yticklabels([])
axs[1, 1].set_xlabel(r"iter. nb.", labelpad=0)

fig.delaxes(axs[0, 0])

if save:
    pu.save_plot(
        fig,
        "FIGURE_5_left",
    )

plt.show()
