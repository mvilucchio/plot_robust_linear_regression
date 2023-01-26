import numpy as np
from scipy.integrate import romb
from numba import njit
from sklearn.metrics import max_error

MULT_INTEGRAL = 10
TOL_INT = 1e-6
N_TEST_POINTS = 200
K_ROMBERG = 13
N_GAUSS_HERMITE = 95

# if _check_nested_list(square_borders):
#     max_range = square_borders[0][1]
# elif isinstance(square_borders, (int, float)):
#     max_range = square_borders
# else:
#     raise ValueError("square_borders should be either a list of lists or a number.")

x_ge, w_ge = np.polynomial.hermite.hermgauss(N_GAUSS_HERMITE)


@njit(error_model="numpy", fastmath=True)
def gauss_hermite_quadrature(fun, mean, std):
    #  x, w = np.polynomial.hermite.hermgauss(N_GAUSS_HERMITE)
    y = np.sqrt(2.0) * std * x_ge + mean
    jacobian = np.sqrt(2.0) * std
    return np.sum(w_ge * jacobian * fun(y))


def _check_nested_list(nested_list):
    if isinstance(nested_list, list):
        if isinstance(nested_list[0], list):
            if isinstance(nested_list[0][0], (int, float)):
                return True
    return False


def find_integration_borders_square(
    fun, scale1, scale2, mult=MULT_INTEGRAL, tol=TOL_INT, n_points=N_TEST_POINTS, args=[]
):
    borders = [[-mult * scale1, mult * scale1], [-mult * scale2, mult * scale2]]

    for idx, ax in enumerate(borders):
        for jdx, border in enumerate(ax):

            while True:
                if idx == 0:
                    max_val = np.max(
                        [
                            fun(borders[idx][jdx], pt, *args)
                            for pt in np.linspace(
                                borders[1 if idx == 0 else 0][0],
                                borders[1 if idx == 0 else 0][1],
                                n_points,
                            )
                        ]
                    )
                else:
                    max_val = np.max(
                        [
                            fun(pt, borders[idx][jdx], *args)
                            for pt in np.linspace(
                                borders[1 if idx == 0 else 0][0],
                                borders[1 if idx == 0 else 0][1],
                                n_points,
                            )
                        ]
                    )
                if max_val > tol:
                    borders[idx][jdx] = borders[idx][jdx] + (-1.0 if jdx == 0 else 1.0) * (
                        scale1 if idx == 0 else scale2
                    )
                else:
                    break

    for ax in borders:
        ax[0] = -np.max(np.abs(ax))
        ax[1] = np.max(np.abs(ax))

    max_val = np.max([borders[0][1], borders[1][1]])

    borders = [[-max_val, max_val], [-max_val, max_val]]

    return borders


def divide_integration_borders_grid(square_borders, proportion=0.5):  # , sides_square=3
    if proportion >= 1.0 or proportion <= 0.0:
        raise ValueError("proportion should be a number between 0.0 and 1.0 not included.")

    max_range = square_borders[0][1]
    mid_range = proportion * max_range

    # 1 | 2 | 3
    # 4 | 5 | 6
    # 7 | 8 | 9

    domain_x = [
        [-mid_range, mid_range],
        [-mid_range, mid_range],
        [mid_range, max_range],
        [mid_range, max_range],
        [mid_range, max_range],
        [-mid_range, mid_range],
        [-max_range, -mid_range],
        [-max_range, -mid_range],
        [-max_range, -mid_range],
    ]

    domain_y = [
        [-mid_range, mid_range],
        [mid_range, max_range],
        [mid_range, max_range],
        [-mid_range, mid_range],
        [-max_range, -mid_range],
        [-max_range, -mid_range],
        [-max_range, -mid_range],
        [-mid_range, mid_range],
        [mid_range, max_range],
    ]

    return domain_x, domain_y


def divide_integration_borders_multiple_grid(square_borders, N=100):  # , sides_square=3
    max_range = square_borders[0][1]
    step = 2 * max_range / N
    # mid_range = proportion * max_range

    # 1 | 2 | 3
    # 4 | 5 | 6
    # 7 | 8 | 9

    domain_x = []
    domain_y = []
    for idx in range(N):
        for jdx in range(N):
            domain_x.append([-max_range + idx * step, -max_range + (idx + 1) * step])
            domain_y.append([-max_range + jdx * step, -max_range + (jdx + 1) * step])

    # print(len(domain_x), len(domain_y))

    return domain_x, domain_y


def domains_line_constraint(square_borders, y_fun, x_fun, args_y, args_x):
    max_range = square_borders[0][1]

    x_test_val = x_fun(max_range, **args_x)

    # if x_test_val > max_range:
    #     domain_x = [[-max_range, max_range]] * 2
    #     domain_y = [
    #         [lambda x: y_fun(x, **args_y), lambda x: max_range],
    #         [lambda x: -max_range, lambda x: y_fun(x, **args_y)],
    #     ]
    # else:
    #     domain_x = [
    #         [x_test_val, max_range],
    #         [-x_test_val, x_test_val],
    #         [-x_test_val, x_test_val],
    #         [-max_range, -x_test_val],
    #     ]
    #     domain_y = [
    #         [-max_range, max_range],
    #         [lambda x: y_fun(x, **args_y), max_range],
    #         [-max_range, lambda x: y_fun(x, **args_y)],
    #         [-max_range, max_range],
    #     ]

    if x_test_val > max_range:
        # print("First case ", x_test_val)
        domain_x = [
            [-max_range, -1.0],
            [-1.0, 0.0],
            [0.0, 1.0],
            [1.0, max_range],
            [-max_range, -1.0],
            [-1.0, 0.0],
            [0.0, 1.0],
            [1.0, max_range],
        ]
        domain_y = [
            [lambda x: y_fun(x, **args_y), lambda x: max_range],
            [lambda x: y_fun(x, **args_y), lambda x: max_range],
            [lambda x: y_fun(x, **args_y), lambda x: max_range],
            [lambda x: y_fun(x, **args_y), lambda x: max_range],
            [lambda x: -max_range, lambda x: y_fun(x, **args_y)],
            [lambda x: -max_range, lambda x: y_fun(x, **args_y)],
            [lambda x: -max_range, lambda x: y_fun(x, **args_y)],
            [lambda x: -max_range, lambda x: y_fun(x, **args_y)],
        ]
    else:
        # print("Second case")
        domain_x = [
            [x_test_val, max_range],
            [-x_test_val, 0.0],
            [0.0, x_test_val],
            [-x_test_val, 0.0],
            [0.0, x_test_val],
            [-max_range, -x_test_val],
        ]
        domain_y = [
            [-max_range, max_range],
            [lambda x: y_fun(x, **args_y), max_range],
            [lambda x: y_fun(x, **args_y), max_range],
            [-max_range, lambda x: y_fun(x, **args_y)],
            [-max_range, lambda x: y_fun(x, **args_y)],
            [-max_range, max_range],
        ]

    return domain_x, domain_y


def domains_double_line_constraint(
    square_borders, y_fun_upper, y_fun_lower, x_fun_upper, args1, args2, args3
):
    max_range = square_borders[0][1]

    x_test_val = x_fun_upper(max_range, **args3)
    x_test_val_2 = x_fun_upper(-max_range, **args3)  # attention

    if x_test_val > max_range:
        # print("Case 1")
        domain_x = [[-max_range, max_range]] * 3
        domain_y = [
            [lambda x: y_fun_upper(x, **args1), lambda x: max_range],
            [
                lambda x: y_fun_lower(x, **args2),
                lambda x: y_fun_upper(x, **args1),
            ],
            [lambda x: -max_range, lambda x: y_fun_lower(x, **args2)],
        ]
    elif x_test_val >= 0:
        if x_test_val_2 < -max_range:
            # print("Case 2.A")
            domain_x = [
                [-max_range, x_test_val],
                [-x_test_val, max_range],
                [-max_range, -x_test_val],
                [x_test_val, max_range],
                [-x_test_val, x_test_val],
            ]
            domain_y = [
                [lambda x: y_fun_upper(x, **args1), lambda x: max_range],
                [lambda x: -max_range, lambda x: y_fun_lower(x, **args2)],
                [lambda x: -max_range, lambda x: y_fun_upper(x, **args1)],
                [lambda x: y_fun_lower(x, **args2), lambda x: max_range],
                [lambda x: y_fun_lower(x, **args2), lambda x: y_fun_upper(x, **args1)],
            ]
        else:
            # print("Case 2.B")
            x_test_val_2 = x_fun_upper(-max_range, **args3)
            domain_x = [
                [x_test_val_2, x_test_val],
                [-x_test_val, -x_test_val_2],
                [x_test_val_2, -x_test_val],
                [x_test_val, -x_test_val_2],
                [-x_test_val, x_test_val],
                [-max_range, x_test_val_2],
                [-x_test_val_2, max_range],
            ]
            domain_y = [
                [lambda x: y_fun_upper(x, **args1), lambda x: max_range],
                [lambda x: -max_range, lambda x: y_fun_lower(x, **args2)],
                [lambda x: -max_range, lambda x: y_fun_upper(x, **args1)],
                [lambda x: y_fun_lower(x, **args2), lambda x: max_range],
                [lambda x: y_fun_lower(x, **args2), lambda x: y_fun_upper(x, **args1)],
                [lambda x: -max_range, lambda x: max_range],
                [lambda x: -max_range, lambda x: max_range],
            ]
    elif x_test_val > -max_range:
        if x_test_val_2 < -max_range:
            # print("Case 3.A")
            domain_x = [
                [-max_range, x_test_val],
                [-x_test_val, max_range],
                [-max_range, x_test_val],
                [-x_test_val, max_range],
                [x_test_val, -x_test_val],
            ]
            domain_y = [
                [lambda x: y_fun_upper(x, **args1), lambda x: max_range],
                [lambda x: -max_range, lambda x: y_fun_lower(x, **args2)],
                [lambda x: -max_range, lambda x: y_fun_upper(x, **args1)],
                [lambda x: y_fun_lower(x, **args2), lambda x: max_range],
                [lambda x: -max_range, lambda x: max_range],
            ]
        else:
            # print("Case 3.B")
            x_test_val_2 = x_fun_upper(-max_range, **args3)
            domain_x = [
                [x_test_val_2, x_test_val],
                [-x_test_val, -x_test_val_2],
                [x_test_val_2, x_test_val],
                [-x_test_val, -x_test_val_2],
                [x_test_val, -x_test_val],
                [-max_range, x_test_val_2],
                [-x_test_val_2, max_range],
            ]
            domain_y = [
                [lambda x: y_fun_upper(x, **args1), lambda x: max_range],
                [lambda x: -max_range, lambda x: y_fun_lower(x, **args2)],
                [lambda x: -max_range, lambda x: y_fun_upper(x, **args1)],
                [lambda x: y_fun_lower(x, **args2), lambda x: max_range],
                [lambda x: -max_range, lambda x: max_range],
                [lambda x: -max_range, lambda x: max_range],
                [lambda x: -max_range, lambda x: max_range],
            ]
    else:
        # print("Case 4")
        domain_x = [[-max_range, max_range]]
        domain_y = [[lambda x: -max_range, lambda x: max_range]]

    return domain_x, domain_y


def domains_double_line_constraint_only_inside(
    square_borders, y_fun_upper, y_fun_lower, x_fun_upper, args1, args2, args3
):
    max_range = square_borders[0][1]

    x_test_val = x_fun_upper(max_range, **args3)
    x_test_val_2 = x_fun_upper(-max_range, **args3)  # attention

    if x_test_val > max_range:
        # print("Case 1")
        domain_x = [[-max_range, max_range]]
        domain_y = [
            [
                lambda x: y_fun_lower(x, **args2),
                lambda x: y_fun_upper(x, **args1),
            ],
        ]
    elif x_test_val >= 0:
        if x_test_val_2 < -max_range:
            # print("Case 2.A")
            domain_x = [
                [-max_range, -x_test_val],
                [x_test_val, max_range],
                [-x_test_val, x_test_val],
            ]
            domain_y = [
                [lambda x: -max_range, lambda x: y_fun_upper(x, **args1)],
                [lambda x: y_fun_lower(x, **args2), lambda x: max_range],
                [lambda x: y_fun_lower(x, **args2), lambda x: y_fun_upper(x, **args1)],
            ]
        else:
            # print("Case 2.B")
            x_test_val_2 = x_fun_upper(-max_range, **args3)
            domain_x = [
                [x_test_val_2, -x_test_val],
                [x_test_val, -x_test_val_2],
                [-x_test_val, x_test_val],
            ]
            domain_y = [
                [lambda x: -max_range, lambda x: y_fun_upper(x, **args1)],
                [lambda x: y_fun_lower(x, **args2), lambda x: max_range],
                [lambda x: y_fun_lower(x, **args2), lambda x: y_fun_upper(x, **args1)],
            ]
    elif x_test_val > -max_range:
        if x_test_val_2 < -max_range:
            # print("Case 3.A")
            domain_x = [
                [-max_range, x_test_val],
                [-x_test_val, max_range],
                [x_test_val, -x_test_val],
            ]
            domain_y = [
                [lambda x: -max_range, lambda x: y_fun_upper(x, **args1)],
                [lambda x: y_fun_lower(x, **args2), lambda x: max_range],
                [lambda x: -max_range, lambda x: max_range],
            ]
        else:
            # print("Case 3.B")
            x_test_val_2 = x_fun_upper(-max_range, **args3)
            domain_x = [
                [x_test_val_2, x_test_val],
                [-x_test_val, -x_test_val_2],
                [x_test_val, -x_test_val],
            ]
            domain_y = [
                [lambda x: -max_range, lambda x: y_fun_upper(x, **args1)],
                [lambda x: y_fun_lower(x, **args2), lambda x: max_range],
                [lambda x: -max_range, lambda x: max_range],
            ]
    else:
        # print("Case 4")
        domain_x = [[-max_range, max_range]]
        domain_y = [[lambda x: -max_range, lambda x: max_range]]

    return domain_x, domain_y


def romberg_linspace(lower, upper):
    return np.linspace(lower, upper, 2**K_ROMBERG + 1)


def full_procedure_double_romb_integration(fun, x_lower, x_upper, y_lower, y_upper, args=[]):
    x = np.linspace(x_lower, x_upper, 2**K_ROMBERG + 1)
    y = np.linspace(y_lower, y_upper, 2**K_ROMBERG + 1)

    dx = x[1] - x[0]
    dy = y[1] - y[0]

    X, Y = np.meshgrid(x, y)

    F = fun(X, Y, *args)

    return romb(romb(F, dy), dx)


def precompute_values_double_romb_integration(fun, x_range, y_range, args=[]):
    # if not isinstance(x_range, list) and not isinstance(y_range, list):
    #     x = x_range
    #     y = y_range
    # elif isinstance(x_range, list) and isinstance(y_range, list):
    #     if

    dx = x_range[1] - x_range[0]
    dy = y_range[1] - y_range[0]

    X, Y = np.meshgrid(x_range, y_range)

    F = fun(X, Y, *args)

    return F


def double_romb_integration(F, dx, dy):
    return romb(romb(F, dy), dx)
