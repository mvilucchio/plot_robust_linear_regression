import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from L2_usable import L2_fixed_point
from L1_usable import L1_fixed_point
from Huber_usable import best_a_param_Huber
from scipy.signal import argrelextrema

def main_min_line_alpha():
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

def main_min_line_delta():
    lambd_list = np.linspace(-.1, 20, 1024)
    delta_large_list = np.logspace(-3, 2, 64)
    delta_small = 1
    alpha = 10
    percentage = .3
    beta = 1

    lambd_min_list = np.zeros_like(delta_large_list)
    for i,delta_large in tqdm(enumerate(delta_large_list)):
        E_FP = np.zeros_like(lambd_list)
        for j,lambd in enumerate(lambd_list):
            E_FP[j] = L2_fixed_point(lambd, alpha, delta_small, delta_large, percentage, beta)
            
        # Find the indices of local minima
        local_minima_indices = argrelextrema(E_FP, np.less)
        
        assert len(local_minima_indices) <= 1
        # The assert checks we didn't loos any solution.
        # If we have 0 means the minimum is at the edges so we have an error below

        print(lambd_list[local_minima_indices[0]])
        lambd_min_list[i] = lambd_list[local_minima_indices[0]]
    plt.plot(delta_large_list, lambd_min_list)
    plt.xscale("log")
    plt.show()


def main_min_line_delta_Huber():
    lambd_list = np.linspace(-10, .1, 128)
    delta_large_list = np.logspace(-1, -3, 32)
    delta_small = 1
    alpha = 10
    percentage = .3
    beta = 0

    lambd_min_list = np.zeros_like(delta_large_list)
    # lambd_min_list_second = np.zeros_like(delta_large_list)
    for i,delta_large in tqdm(enumerate(delta_large_list)):
        E_FP = np.zeros_like(lambd_list)
        a_opt = np.zeros_like(lambd_list)
        for j,lambd in tqdm(enumerate(lambd_list)):
            E_FP[j], a_opt[j] = best_a_param_Huber(lambd, alpha, delta_small, delta_large, percentage, beta, initial_a=.01)
        
        # Find the indices of local minima
        local_minima_indices = argrelextrema(E_FP, np.less)
        
        assert len(local_minima_indices) <= 1
        # The assert checks we didn't loos any solution.
        # If we have 0 means the minimum is at the edges so we have an error below

        plt.plot(lambd_list, E_FP)
        plt.show()
        print(f"\nMin idx: {local_minima_indices}")
        print(f"lambda: {lambd_list[local_minima_indices[0]]}")
        print(f"a: {a_opt[local_minima_indices[0]]}\n")
        lambd_min_list[i] = lambd_list[local_minima_indices[0]]
    plt.plot(delta_large_list, lambd_min_list)
    # plt.plot(delta_large_list, lambd_min_list_second)
    plt.xscale("log")
    plt.show()

def main_line():
    lambd_list = np.linspace(-20, 10, 128)
    alpha = 10
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
    main_min_line_delta_Huber()