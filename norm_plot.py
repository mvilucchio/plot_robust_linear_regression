import numpy as np
import matplotlib.pyplot as plt
from src.numerics import data_generation, measure_gen_decorrelated, find_coefficients_Huber

alpha = 10

d = 100

# params = (delta_small, delta_large, percentage, beta)
# mean, std = _find_numerical_mean_std(alpha, measure_gen_decorrelated, find_coefficients_Huber, d, repetitions, params, (lambd,a))

def find_weights(d, lambd, a, alpha, delta_small, delta_large, percentage, beta):
    params = (delta_small, delta_large, percentage, beta)

    xs, ys, _, _, ground_truth_theta = data_generation(
        measure_gen_decorrelated,
        n_features=d,
        n_samples=max(int(np.around(d * alpha)), 1),
        n_generalization=1,
        measure_fun_args=params
    )

    estimated_theta = find_coefficients_Huber(ys, xs, lambd,a)
    return np.linalg.norm(estimated_theta)

def main():
    lambd_list = np.logspace(-5,1,32)
    a = .0001
    alpha = 10
    delta_small = 1
    delta_large = .01
    percentage = .3
    beta = 0

    d = 100

    params = (delta_small, delta_large, percentage, beta)
    xs, ys, _, _, ground_truth_theta = data_generation(
        measure_gen_decorrelated,
        n_features=d,
        n_samples=max(int(np.around(d * alpha)), 1),
        n_generalization=1,
        measure_fun_args=params
    )

    norm_list = np.zeros_like(lambd_list)
    for i,lambd in enumerate(lambd_list):
        estimated_theta = find_coefficients_Huber(ys, xs, lambd,a)
        norm_list[i] = np.mean(np.abs(ys - estimated_theta@xs.T/np.sqrt(d)))

    plt.plot(lambd_list, norm_list, marker='.')
    plt.xscale('log')
    plt.xlabel('lambda')
    plt.ylabel('norm recovered vector')
    plt.title(f"d={d}, a={a}alpha={alpha}, delta_small={delta_small}, delta_large={delta_large}, percentage={percentage}, beta={beta}")
    plt.show()
    return

if __name__=="__main__":
    main()