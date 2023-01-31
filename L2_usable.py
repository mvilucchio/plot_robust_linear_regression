import numpy as np

# Fixed point imports
from src.fpeqs_L2 import var_func_L2, var_hat_func_L2_decorrelated_noise
from src.fpeqs import state_equations

# BO simulation imports
from src.numerics import measure_gen_decorrelated, find_coefficients_L2, _find_numerical_mean_std

def L2_fixed_point(lambd, alpha, delta_small, delta_large, percentage, beta):
    """Given the parameters iterates the fixed point equations for L2. Returns the generalisation error.

    Args:
        lambd (float): regularisation parameter
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

    m, q, _ = state_equations(
                var_func_L2,
                var_hat_func_L2_decorrelated_noise,
                reg_param=lambd,
                alpha=alpha,
                init=initial_condition,
                var_hat_kwargs=params
    )

    # Generalisation error formula
    return 1 + q - 2 * m

def L2_sim(d, lambd, alpha, delta_small, delta_large, percentage, beta, repetitions=10):
    """Uses the closed formula for solving L2 ridge regression to find the generalisation error on syntetic data. It should match the output of L2_fixed_point

    Args:
        d (float): data dimension
        lambd (float): regularisation parameter
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
    mean, std = _find_numerical_mean_std(alpha, measure_gen_decorrelated, find_coefficients_L2, d, repetitions, params, (lambd,))
    return mean, std/np.sqrt(repetitions)

# 1.113519427471848927e-01,9.814863459332267004e-01,3.133160610198977736e+00

def main():
    lambd = 3.133160610198977736e+00
    alpha = 1.113519427471848927e-01
    delta_small = 1
    delta_large = 5
    percentage = .3
    beta = 0

    E_FP = L2_fixed_point(lambd, alpha, delta_small, delta_large, percentage, beta)
    print(f"E (fixed point): {E_FP}")

    d = 1000
    E_L2 =  L2_sim(d, lambd, alpha, delta_small, delta_large, percentage, beta, repetitions=10)
    print(f"E (L2): {E_L2}")


if __name__=="__main__":
    main()