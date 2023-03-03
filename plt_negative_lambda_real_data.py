import matplotlib.pyplot as plt
import numpy as np
import src.plotting_utils as pu

save = True
width = 1.0 * 458.63788

pu.initialization_mpl()

tuple_size = pu.set_size(width, fraction=0.50)

multiplier = 0.9
second_multiplier = 0.7

fig, ax = plt.subplots(1, 1, figsize=(multiplier*tuple_size[0], 0.75*multiplier*tuple_size[0]))
fig.subplots_adjust(left=0.16)
fig.subplots_adjust(bottom=0.16)
fig.subplots_adjust(top=0.93)
fig.subplots_adjust(right=0.93)


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]


for i in range(10):
    d = np.loadtxt("./data/negative-lambda-real-data/full_comp_{:d}_cv_MSE_lambda_sweep_a_opt_alpha_0.03.csv".format(i), delimiter=",", skiprows=1)
    # d2 = np.loadtxt("./data-neg-lambda/upper_comp_{:d}_cv_MSE_lambda_sweep_a_opt_alpha_0.03.csv".format(i), delimiter=",", skiprows=1)
    color = next(plt.gca()._get_lines.prop_cycler)['color']
    idx, _ = find_nearest(d[:,0], 0.0)
    ax.plot(d[:,0], (d[:,1] - d[idx,1]) / d[idx,1], '.-', color = color)
    # plt.plot(d2[:,0], d2[:,1], color = color)

ax.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
# ax.set_title(r"$\alpha$ = 0.03")
ax.set_xlabel(r"$\lambda$")
ax.set_ylabel(r"$\Delta$ CV-MSE")
# ax.grid(which="both", axis="both")
# plt.xscale('log')
# plt.yscale('log')
# plt.ylim(0, 1.5)


if save:
    pu.save_plot(
        fig,
        "real-data-negative-lambda",
    )

plt.show()