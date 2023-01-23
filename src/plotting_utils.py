import os
import datetime as dt
import matplotlib.pyplot as plt

IMG_DIRECTORY = "./imgs" # "/Volumes/LaCie/final_imgs_hproblem/new3/new5"  #  "./imgs" #  #  # "/Volumes/LaCie/final_imgs_hproblem" #
STYLES_DIRECTORY = "./src/mpl_styles"


def initialization_mpl(stylesheet_name="latex_ready"):
    plt.style.use(os.path.join(STYLES_DIRECTORY, stylesheet_name + ".mplstyle"))


def save_plot(fig, name, formats=["png", "pdf", "svg"], date=True):
    now = dt.datetime.now()
    for f in formats:
        fig.savefig(
            os.path.join(
                IMG_DIRECTORY,
                "{}".format(name) + "_" + now.strftime("%Y_%m_%d_%H_%M_%S") + "." + f,
            ),
            format=f,
        )


def set_size(width, fraction=1, subplots=(1, 1)):
    if width == "thesis":
        width_pt = 426.79135
    elif width == "beamer":
        width_pt = 307.28987
    else:
        width_pt = width

    fig_width_pt = width_pt * fraction
    inches_per_pt = 1 / 72.27

    golden_ratio = (5 ** 0.5 - 1) / 2

    fig_width_in = fig_width_pt * inches_per_pt
    fig_height_in = fig_width_in * (golden_ratio) * (subplots[0] / subplots[1])

    return (fig_width_in, fig_height_in)
