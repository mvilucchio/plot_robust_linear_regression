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

delta_in, delta_out, percentage, beta = 1.0, 5.0, 0.3, 0.0
alpha = 2.0

color_zero = (0.0, 0.0, 0.0, 0.8)

# beginning of script

# loading of AMP data from cluster
fig, (ax1, ax2) = plt.subplots(
    nrows=2,
    ncols=1,
    sharex=True,
    figsize=(multiplier * tuple_size[0], 0.75 * multiplier * tuple_size[0]),
    gridspec_kw={"hspace": 0},
)
fig.subplots_adjust(left=0.16)
fig.subplots_adjust(bottom=0.16)
fig.subplots_adjust(top=0.97)
fig.subplots_adjust(right=0.97)

# load the data from cluster
dat_AMP = np.loadtxt(
    "./data/FIGURE_4_right_AMP_data.csv"
)
qs_amp = dat_AMP[:, 0]
gen_error_mean = dat_AMP[:, 1]
gen_error_std = dat_AMP[:, 2]
train_error_mean = dat_AMP[:, 3]
train_error_std = dat_AMP[:, 4]
iters_mean = dat_AMP[:, 5]
iters_std = dat_AMP[:, 6]

qs = np.logspace(-1, 2, N)
ms = np.zeros_like(qs)
sigmas = np.zeros_like(qs)
m_hats = np.zeros_like(qs)
sigma_hats = np.zeros_like(qs)
q_hats = np.zeros_like(qs)
training_error = np.zeros_like(qs)

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


# plot of generalization error in ax1

x = np.logspace(-1, 1, 1000)
ax1.errorbar(qs_amp, train_error_mean, yerr=train_error_std, color="tab:blue")
ax1.errorbar(qs_amp, gen_error_mean, yerr=gen_error_std, color="tab:orange")
ax1.plot(qs, training_error, label="Training Error", color="tab:blue")
ax1.plot(qs, 1 + qs - 2.0 * ms, label="Generalization Error", color="tab:orange")

ax1.set_xscale("log")
ax1.set_yscale("log")
ax1.set_xlim([0.1, 100])


# plot of iterations to convergence error in ax2

ax2.errorbar(qs_amp, iters_mean, yerr=iters_std, label="Iterations", color="gray")
ax2.set_yscale("log")
ax2.set_xlabel(r"$q$", labelpad=0)

if save:
    pu.save_plot(
        fig,
        "FIGURE_4_right",
    )

plt.show()
