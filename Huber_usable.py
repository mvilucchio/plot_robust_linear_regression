import numpy as np

# Fixed point imports
from src.fpeqs_Huber import var_func_L2, var_hat_func_Huber_decorrelated_noise
from src.fpeqs import state_equations

# BO simulation imports
from src.numerics import measure_gen_decorrelated, find_coefficients_Huber, _find_numerical_mean_std

# Hyperparameter optimisation imports
from scipy.optimize import minimize

def Huber_fixed_point(lambd, a, alpha, delta_small, delta_large, percentage, beta):
    """Given the parameters iterates the fixed point equations for L1. Returns the generalisation error.

    Args:
        lambd (float): regularisation parameter
        a (float): Huber scale parameter
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
        "a": a
    }

    m, q, _ = state_equations(
                var_func_L2,
                var_hat_func_Huber_decorrelated_noise,
                reg_param=lambd,
                alpha=alpha,
                init=initial_condition,
                var_hat_kwargs=params
    )

    # Generalisation error formula
    return 1 + q - 2 * m

def Huber_sim(d, lambd, a, alpha, delta_small, delta_large, percentage, beta, repetitions=10):
    """Uses the closed formula for solving L2 ridge regression to find the generalisation error on syntetic data. It should match the output of L2_fixed_point

    Args:
        alpha (float): sample complexity
        lambd (float): regularisation parameter
        a (float): Huber scale parameter
        delta_small (float): inliers noise variance
        delta_large (float): outliers noise variance
        percentage (float): percentage of outliers
        beta (float): correlation parameters (0 for decorrelated outliers, 1 for correlated)
        repetitions (int, optional): number of different experiments. Defaults to 10.

    Returns:
        (float, float): mean and std from the mean of the generalisation error
    """
    params = (delta_small, delta_large, percentage, beta)
    mean, std = _find_numerical_mean_std(alpha, measure_gen_decorrelated, find_coefficients_Huber, d, repetitions, params, (lambd,a))
    return mean, std/np.sqrt(repetitions)



def best_a_param_Huber(lambd, alpha, delta_small, delta_large, percentage, beta, initial_a):
    XATOL = 1e-10
    FATOL = 1e-10

    def minimize_fun(a):
        return Huber_fixed_point(lambd, a, alpha, delta_small, delta_large, percentage, beta)

    obj = minimize(
        minimize_fun,
        x0=initial_a,
        method="Nelder-Mead",
        options={
            "xatol": XATOL,
            "fatol": FATOL,
            "adaptive": True,
        },
    )

    if obj.success:
        E_opt = obj.fun
        a_opt = obj.x
        return E_opt, a_opt
    else:
        raise RuntimeError("Minima could not be found.")


def main():
    lambd = 1.681086912923367427e+00
    a = 1.107195765398673704e+00
    alpha = 1.113519427471848927e-01
    delta_small = 1
    delta_large = 5
    percentage = .3
    beta = 0

    E_FP = Huber_fixed_point(lambd, a, alpha, delta_small, delta_large, percentage, beta)
    print(f"E (fixed point): {E_FP}")

    E_best = best_a_param_Huber(lambd, alpha, delta_small, delta_large, percentage, beta, initial_a=1)
    print(f"E (best a): {E_best}")

    d = 1000
    E_Huber =  Huber_sim(d, lambd, a, alpha, delta_small, delta_large, percentage, beta, repetitions=10)
    print(f"E (Huber): {E_Huber}")


if __name__=="__main__":
    main()