import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from L2_usable import L2_fixed_point
from L1_usable import L1_fixed_point
from scipy.signal import argrelextrema

def main_min_line():
    lambd_list = np.linspace(-5, 2, 1024)
    alpha_list = np.logspace(1, 1.47, 32)
    delta_small = 1
    delta_large = 5
    percentage = .3
    beta = 0

    lambd_min_list = np.zeros_like(alpha_list)
    for i,alpha in tqdm(enumerate(alpha_list)):
        E_FP = np.zeros_like(lambd_list)
        for j,lambd in enumerate(lambd_list):
            E_FP[j] = L1_fixed_point(lambd, alpha, delta_small, delta_large, percentage, beta)

        # Find the indices of local minima
        local_minima_indices = argrelextrema(E_FP, np.less)
        
        assert len(local_minima_indices) <= 1

        lambd_min_list[i] = lambd_list[local_minima_indices[0]]
    plt.plot(alpha_list, lambd_min_list)
    plt.xscale("log")
    plt.show()

def main_line():
    lambd_list = np.linspace(-20, 10, 128)
    alpha = 100
    delta_small = 1
    delta_large = 5
    percentage = .1
    beta = 0

    E_FP_list = np.zeros_like(lambd_list)
    for i, lambd in tqdm(enumerate(lambd_list)):
        E_FP_list[i] = L1_fixed_point(lambd, alpha, delta_small, delta_large, percentage, beta)

    plt.plot(lambd_list, E_FP_list)
    plt.show()

if __name__=="__main__":
    main_min_line()