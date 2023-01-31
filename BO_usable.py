import numpy as np

# Fixed point imports
from src.fpeqs import _find_fixed_point
from src.fpeqs_BO import var_hat_func_BO_num_decorrelated_noise, var_func_BO

# BO simulation imports
from src.numerics import measure_gen_decorrelated, find_coefficients_AMP_decorrelated_noise, _find_numerical_mean_std


def BO_fixed_point(alpha, delta_small, delta_large, percentage, beta):
    """Given the parameters iterates the fixed point equations for BO. Returns the generalisation error.

    Args:
        alpha (float): sample complexity
        delta_small (float): inliers noise variance
        delta_large (float): outliers noise variance
        percentage (float): percentage of outliers
        beta (float): correlation parameters (0 for decorrelated outliers, 1 for correlated)

    Returns:
        (float): generalisation error
    """    

    initial_condition = [.3,.2,.1]
    params = {
        "delta_small": delta_small,
        "delta_large": delta_large,
        "percentage": percentage,
        "beta": beta,
    }
    m, q, _ = _find_fixed_point(alpha, var_func_BO, var_hat_func_BO_num_decorrelated_noise, 1, initial_condition, var_hat_kwargs=params)

    # Generalisation error formula. Here it's BO so m and q are trivially related (read Aubin '20)
    return 1 + q - 2 * m


def AMP_BO(d, alpha, delta_small, delta_large, percentage, beta, repetitions=10):
    """Runs AMP on simluated data to compute the generalisation error. It should match the output of BO_fixed_point

    Args:
        d (float): data dimension
        alpha (float): sample complexity
        delta_small (float): inliers noise variance
        delta_large (float): outliers noise variance
        percentage (float): percentage of outliers
        beta (float): correlation parameters (0 for decorrelated outliers, 1 for correlated)
        repetitions (int, optional): number of different experiments. Defaults to 10.

    Returns:
        (float, float): mean and std from the mean of the generalisation error
    """
    params = (delta_small, delta_large, percentage, beta)
    mean, std = _find_numerical_mean_std(alpha, measure_gen_decorrelated, find_coefficients_AMP_decorrelated_noise, d, repetitions, params, params)
    return mean, std/np.sqrt(repetitions)


def main():
    alpha = 8.54
    delta_small = 1
    delta_large = 5
    percentage = .3
    beta = 0

    E_FP = BO_fixed_point(alpha, delta_small, delta_large, percentage, beta)
    print(f"E (fixed point): {E_FP}")

    d = 100
    E_AMP = AMP_BO(d, alpha, delta_small, delta_large, percentage, beta, repetitions=10)
    print(f"E (AMP): {E_AMP}")


if __name__=="__main__":
    main()

