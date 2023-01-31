import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from L2_usable import L2_fixed_point
from scipy.signal import argrelextrema

def main_heatmap():
    lambd_list = np.linspace(-.05, 10, 32)
    alpha_list = np.logspace(-1, 2, 32)
    delta_small = 1
    delta_large = 5
    percentage = .1
    beta = 1

    E_FP_list = np.zeros([32,32])
    for i, lambd in tqdm(enumerate(lambd_list)):
        for j, alpha in enumerate(alpha_list):
            E_FP_list[i,j] = L2_fixed_point(lambd, alpha, delta_small, delta_large, percentage, beta)

    lambd_min = []
    for j,alpha in enumerate(alpha_list):
        lambd_min.append(argrelextrema(E_FP_list[:,j], np.less))
        for lambd in lambd_min:
            print(alpha, lambd[0])
            plt.scatter(alpha, lambd)  
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
        E_FP_list[i] = L2_fixed_point(lambd, alpha, delta_small, delta_large, percentage, beta)

    plt.plot(lambd_list, E_FP_list)
    plt.show()

if __name__=="__main__":
    main_line()