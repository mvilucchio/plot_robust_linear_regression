import numpy as np
import os
from re import search
import src.numerics as num
import src.fpeqs as fpe
from src.optimal_lambda import (
    optimal_lambda,
    optimal_reg_param_and_huber_parameter,
    no_parallel_optimal_reg_param_and_huber_parameter,
)
from src.fpeqs_BO import (
    var_func_BO,
    var_hat_func_BO_single_noise,
    var_hat_func_BO_num_double_noise,
    var_hat_func_BO_num_decorrelated_noise,
)
from src.fpeqs_L2 import (
    var_func_L2,
    var_hat_func_L2_single_noise,
    var_hat_func_L2_double_noise,
    var_hat_func_L2_decorrelated_noise,
)
from src.fpeqs_L1 import (
    var_hat_func_L1_single_noise,
    var_hat_func_L1_double_noise,
    var_hat_func_L1_decorrelated_noise,
)
from src.fpeqs_Huber import (
    var_hat_func_Huber_single_noise,
    var_hat_func_Huber_double_noise,
    var_hat_func_Huber_decorrelated_noise,
)


def _get_spaced_index(array, num_elems, max_val=100):
    cond_array = array <= max_val
    return np.round(np.linspace(0, len(array[cond_array]) - 1, num_elems)).astype(int)


DATA_FOLDER_PATH = "./" # "/Volumes/LaCie/final_data_hproblem/"  # "/Users/matteovilucchio/Documents/ENS/HProblem/robust_linear_regression/data"  # "./data"  # "/Volumes/LaCie/final_data_hproblem" #  # "/Volumes/LaCie/final_data_hproblem" #  #  #

CSV_FOLDER_PATHS = [
    "./data/csv/experiments",
    "./data/csv/GAMP_experiments",
    "./data/csv/theory",
    "./data/csv/bayes_optimal",
    "./data/csv/reg_param_optimal",
    "./data/csv/reg_param_optimal_experimental",
    "./data/csv/reg_and_huber_param_optimal",
    "./data/csv/reg_and_huber_param_optimal_experimental",
    "./data/csv/others",
]

FOLDER_PATHS = [
    "data/experiments/",
    "data/GAMP_experiments/",
    "data/theory/",
    "data/bayes_optimal/",
    "data/reg_param_optimal/",
    "data/reg_param_optimal_experimental/",
    "data/reg_and_huber_param_optimal/",
    "data/reg_and_huber_param_optimal_experimental/",
    "data/others/",
]

REG_EXPS = [
    "(^exp)",
    "(^GAMP)",
    "(^theory)",
    "(BO|Bayes[ ]{0,1}Optimal)",
    "((reg[\_\s]{0,1}param|lambda)[\_\s]{0,1}optimal$)",
    "((reg[\_\s]{0,1}param|lambda)[\_\s]{0,1}(optimal)[\_\s]{1}(exp))",
    "((reg[\_\s]{0,1}param|lambda)[\s]{1}(huber[\_\s]{0,1}param)[\_\s]{0,1}optimal$)",
    "((reg[\_\s]{0,1}param|lambda)[\s]{1}(huber[\_\s]{0,1}param)[\_\s]{0,1}optimal)[\_\s]{1}(exp)",
]

LOSS_NAMES = ["L2", "L1", "Huber", "DoubleQuad"]

EXPERIMENTAL_FUNCTIONS = []

SINGLE_NOISE_NAMES = [
    "{loss_name} single noise - exp - alphas [{alpha_min} {alpha_max} {alpha_pts:d}] - dim {n_features:d} - rep {repetitions:d} - delta {delta} - lambda {reg_param}",
    "G-AMP single noise - alphas [{alpha_min} {alpha_max} {alpha_pts:d}] - dim {n_features:d} - rep {repetitions:d} - delta {delta}",
    "{loss_name} single noise - theory - alphas [{alpha_min} {alpha_max} {alpha_pts:d}] - delta {delta} - lambda {reg_param}",
    "BO single noise - alphas [{alpha_min} {alpha_max} {alpha_pts:d}] - delta {delta}",
    "{loss_name} single noise - reg_param optimal - alphas [{alpha_min} {alpha_max} {alpha_pts:d}] - delta {delta}",
    "{loss_name} single noise - reg_param optimal experimental - alphas [{alpha_min} {alpha_max} {alpha_pts_experimental:d}] - delta {delta}",
    "Huber single noise - reg_param and huber_param optimal - alphas [{alpha_min} {alpha_max} {alpha_pts:d}] - delta {delta}",
    "Huber single noise - reg_param and huber_param optimal experimental - alphas [{alpha_min} {alpha_max} {alpha_pts_experimental:d}] - delta {delta}",
    "{loss_name} single noise - alphas [{alpha_min} {alpha_max} {alpha_pts:d}] - delta {delta} - lambda {reg_param}",
]

DOUBLE_NOISE_NAMES = [
    "{loss_name} double noise - eps {percentage} - exp - alphas [{alpha_min} {alpha_max} {alpha_pts:d}] - dim {n_features:d} - rep {repetitions:d} - delta [{delta_small} {delta_large}] - lambda {reg_param}",
    "G-AMP double noise - eps {percentage} - alphas [{alpha_min} {alpha_max} {alpha_pts:d}] - dim {n_features:d} - rep {repetitions:d} - delta [{delta_small} {delta_large}]",
    "{loss_name} double noise - eps {percentage} - theory - alphas [{alpha_min} {alpha_max} {alpha_pts:d}] - delta [{delta_small} {delta_large}] - lambda {reg_param}",
    "BO double noise - eps {percentage} - alphas [{alpha_min} {alpha_max} {alpha_pts:d}] - delta [{delta_small} {delta_large}]",
    "{loss_name} double noise - eps {percentage} - reg_param optimal - alphas [{alpha_min} {alpha_max} {alpha_pts:d}] - delta [{delta_small} {delta_large}]",
    "{loss_name} double noise - eps {percentage} - reg_param optimal experimental - alphas [{alpha_min} {alpha_max} {alpha_pts_experimental:d}] - delta [{delta_small} {delta_large}]",
    "Huber double noise - eps {percentage} - reg_param and huber_param optimal - alphas [{alpha_min} {alpha_max} {alpha_pts:d}] - delta [{delta_small} {delta_large}]",
    "Huber double noise - eps {percentage} - reg_param and huber_param optimal experimental - alphas [{alpha_min} {alpha_max} {alpha_pts_experimental:d}] - delta [{delta_small} {delta_large}]",
    "{loss_name} double noise - eps {percentage} - alphas [{alpha_min} {alpha_max} {alpha_pts:d}] - delta [{delta_small} {delta_large}] - lambda {reg_param}",
]

DECORRELATED_NOISE_NAMES = [
    "{loss_name} decorrelated noise {beta} - eps {percentage} - exp - alphas [{alpha_min} {alpha_max} {alpha_pts:d}] - dim {n_features:d} - rep {repetitions:d} - delta [{delta_small} {delta_large}] - lambda {reg_param}",
    "G-AMP decorrelated noise {beta} - eps {percentage} - alphas [{alpha_min} {alpha_max} {alpha_pts:d}] - dim {n_features:d} - rep {repetitions:d} - delta [{delta_small} {delta_large}]",
    "{loss_name} decorrelated noise {beta} - eps {percentage} - theory - alphas [{alpha_min} {alpha_max} {alpha_pts:d}] - delta [{delta_small} {delta_large}] - lambda {reg_param}",
    "BO decorrelated noise {beta} - eps {percentage} - alphas [{alpha_min} {alpha_max} {alpha_pts:d}] - delta [{delta_small} {delta_large}]",
    "{loss_name} decorrelated noise {beta} - eps {percentage} - reg_param optimal - alphas [{alpha_min} {alpha_max} {alpha_pts:d}] - delta [{delta_small} {delta_large}]",
    "{loss_name} decorrelated noise {beta} - eps {percentage} - reg_param optimal experimental - alphas [{alpha_min} {alpha_max} {alpha_pts_experimental:d}] - delta [{delta_small} {delta_large}]",
    "Huber decorrelated noise {beta} - eps {percentage} - reg_param and huber_param optimal - alphas [{alpha_min} {alpha_max} {alpha_pts:d}] - delta [{delta_small} {delta_large}]",
    "Huber decorrelated noise {beta} - eps {percentage} - reg_param and huber_param optimal experimental - alphas [{alpha_min} {alpha_max} {alpha_pts_experimental:d}] - delta [{delta_small} {delta_large}]",
    "{loss_name} decorrelated noise {beta} - eps {percentage} - alphas [{alpha_min} {alpha_max} {alpha_pts:d}] - delta [{delta_small} {delta_large}] - lambda {reg_param}",
]

# ------------


def _exp_type_choser(test_string, values=[0, 1, 2, 3, 4, 5, 6, -1]):
    for idx, re in enumerate(REG_EXPS):
        if search(re, test_string):
            return values[idx]
    return values[-1]


def _loss_type_chose(test_string, values=[0, 1, 2, 3, 4, 5, 6, -1]):
    for idx, re in enumerate(LOSS_NAMES):
        if search(re, test_string):
            return values[idx]
    return values[-1]


def file_name_generator(**kwargs):
    experiment_code = _exp_type_choser(kwargs["experiment_type"])

    if not (
        experiment_code == 1 or experiment_code == 3 or experiment_code == 6 or experiment_code == 7
    ):
        if kwargs["loss_name"] == "Huber":
            kwargs["loss_name"] += " " + str(kwargs.get("a", 1.0))

    if kwargs.get("beta") is not None:
        return DECORRELATED_NOISE_NAMES[experiment_code].format(**kwargs)
    else:
        if float(kwargs.get("percentage", 0.0)) == 0.0:
            return SINGLE_NOISE_NAMES[experiment_code].format(**kwargs)
        else:
            return DOUBLE_NOISE_NAMES[experiment_code].format(**kwargs)


def create_check_folders():
    if not os.path.exists(DATA_FOLDER_PATH):
        os.makedirs(DATA_FOLDER_PATH)
        for folder_path in FOLDER_PATHS:
            os.makedirs(folder_path)

    for folder_path in FOLDER_PATHS:
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)


def check_saved(**kwargs):
    # create_check_folders()

    experiment_code = _exp_type_choser(kwargs["experiment_type"])
    folder_path = FOLDER_PATHS[experiment_code]

    file_path = os.path.join(folder_path, file_name_generator(**kwargs))

    file_exists = os.path.exists(file_path + ".npz")

    return file_exists, (file_path + ".npz")


def save_file(**kwargs):
    file_path = kwargs.get("file_path")
    experiment_code = _exp_type_choser(kwargs["experiment_type"])
    print(experiment_code)

    if file_path is None:
        file_path = os.path.join(FOLDER_PATHS[experiment_code], file_name_generator(**kwargs))

    if experiment_code == 0 or experiment_code == 1:
        np.savez(
            file_path,
            alphas=kwargs["alphas"],
            errors_mean=kwargs["errors_mean"],
            errors_std=kwargs["errors_std"],
        )
    elif experiment_code == 2 or experiment_code == 3:
        np.savez(file_path, alphas=kwargs["alphas"], errors=kwargs["errors"])
    elif experiment_code == 4:
        np.savez(
            file_path,
            alphas=kwargs["alphas"],
            errors=kwargs["errors"],
            lambdas=kwargs["lambdas"],
        )
    elif experiment_code == 5:
        np.savez(
            file_path,
            alphas=kwargs["alphas"],
            errors_mean=kwargs["errors_mean"],
            errors_std=kwargs["errors_std"],
            lambdas=kwargs["lambdas"],
        )
    elif experiment_code == 6:
        np.savez(
            file_path,
            alphas=kwargs["alphas"],
            errors=kwargs["errors"],
            lambdas=kwargs["lambdas"],
            huber_params=kwargs["huber_params"],
        )
    elif experiment_code == 7:
        np.savez(
            file_path,
            alphas=kwargs["alphas"],
            errors_mean=kwargs["errors_mean"],
            errors_std=kwargs["errors_std"],
            lambdas=kwargs["lambdas"],
            huber_params=kwargs["huber_params"],
        )
    else:
        raise ValueError("experiment_type not recognized.")


def load_file(**kwargs):
    file_path = kwargs.get("file_path")

    experiment_code = _exp_type_choser(kwargs["experiment_type"])
    print(experiment_code)

    if file_path is None:
        file_path = os.path.join(
            FOLDER_PATHS[experiment_code], file_name_generator(**kwargs) + ".npz"
        )

    print(file_path)

    saved_data = np.load(file_path)

    if experiment_code == 0 or experiment_code == 1:
        alphas = saved_data["alphas"]
        errors_mean = saved_data["errors_mean"]
        errors_std = saved_data["errors_std"]
        return alphas, errors_mean, errors_std
    elif experiment_code == 2 or experiment_code == 3:
        alphas = saved_data["alphas"]
        errors = saved_data["errors"]
        return alphas, errors
    elif experiment_code == 4:
        alphas = saved_data["alphas"]
        errors = saved_data["errors"]
        lambdas = saved_data["lambdas"]
        return alphas, errors, lambdas
    elif experiment_code == 5:
        alphas = saved_data["alphas"]
        errors_mean = saved_data["errors_mean"]
        errors_std = saved_data["errors_std"]
        lambdas = saved_data["lambdas"]
        return alphas, errors_mean, errors_std, lambdas
    elif experiment_code == 6:
        alphas = saved_data["alphas"]
        errors = saved_data["errors"]
        lambdas = saved_data["lambdas"]
        huber_params = saved_data["huber_params"]
        return alphas, errors, lambdas, huber_params
    elif experiment_code == 7:
        alphas = saved_data["alphas"]
        errors_mean = saved_data["errors_mean"]
        errors_std = saved_data["errors_std"]
        lambdas = saved_data["lambdas"]
        huber_params = saved_data["huber_params"]
        return alphas, errors_mean, errors_std, lambdas, huber_params
    else:
        raise ValueError("experiment_type not recognized.")


def npz_to_csv_converter(**kwargs):
    file_path = kwargs.get("file_path")

    experiment_code = _exp_type_choser(kwargs["experiment_type"])

    if file_path is None:
        file_path = os.path.join(
            DATA_FOLDER_PATH, FOLDER_PATHS[experiment_code], file_name_generator(**kwargs) + ".npz"
        )

    csv_file_path = os.path.join(
        CSV_FOLDER_PATHS[experiment_code], file_name_generator(**kwargs) + ".csv"
    )

    saved_data = np.load(file_path)

    if experiment_code == 0 or experiment_code == 1:
        alphas = saved_data["alphas"]
        errors_mean = saved_data["errors_mean"]
        errors_std = saved_data["errors_std"]

        np.savetxt(
            csv_file_path,
            np.stack((alphas, errors_mean, errors_std)).T,
            header="alphas,errors-mean,errors-std",
            delimiter=",",
            comments="",
        )
    elif experiment_code == 2 or experiment_code == 3:
        alphas = saved_data["alphas"]
        errors = saved_data["errors"]

        np.savetxt(
            csv_file_path,
            np.stack((alphas, errors)).T,
            header="alphas,errors",
            delimiter=",",
            comments="",
        )
    elif experiment_code == 4:
        alphas = saved_data["alphas"]
        errors = saved_data["errors"]
        lambdas = saved_data["lambdas"]

        np.savetxt(
            csv_file_path,
            np.stack((alphas, errors, lambdas)).T,
            header="alphas,errors,lambdas",
            delimiter=",",
            comments="",
        )
    elif experiment_code == 5:
        alphas = saved_data["alphas"]
        errors_mean = saved_data["errors_mean"]
        errors_std = saved_data["errors_std"]
        lambdas = saved_data["lambdas"]

        np.savetxt(
            csv_file_path,
            np.stack((alphas, errors_mean, errors_std, lambdas)).T,
            header="alphas,errors-mean,errors-std,lambdas",
            delimiter=",",
            comments="",
        )
    elif experiment_code == 6:
        alphas = saved_data["alphas"]
        errors = saved_data["errors"]
        lambdas = saved_data["lambdas"]
        huber_params = saved_data["huber_params"]

        np.savetxt(
            csv_file_path,
            np.stack((alphas, errors, lambdas, huber_params)).T,
            header="alphas,errors,lambdas,huber-params",
            delimiter=",",
            comments="",
        )
    elif experiment_code == 7:
        alphas = saved_data["alphas"]
        errors_mean = saved_data["errors_mean"]
        errors_std = saved_data["errors_std"]
        lambdas = saved_data["lambdas"]
        huber_params = saved_data["huber_params"]

        np.savetxt(
            csv_file_path,
            np.stack((alphas, errors_mean, errors_std, lambdas, huber_params)).T,
            header="alphas,errors-mean,errors-std,lambdas,huber-params",
            delimiter=",",
            comments="",
        )
    else:
        raise ValueError("experiment_type not recognized.")


# ------------


def experiment_runner(**kwargs):
    experiment_code = _exp_type_choser(kwargs["experiment_type"])

    if experiment_code == 0:
        experimental_points_runner(**kwargs)
    elif experiment_code == 1:
        theory_curve_runner(**kwargs)
    elif experiment_code == 2:
        bayes_optimal_runner(**kwargs)
    elif experiment_code == 3:
        reg_param_optimal_runner(**kwargs)
    elif experiment_code == 4:
        reg_param_optimal_experiment_runner(**kwargs)
    elif experiment_code == 5:
        reg_param_and_huber_param_optimal_runner(**kwargs)
    elif experiment_code == 6:
        reg_param_and_huber_param_experimental_optimal_runner(**kwargs)
    else:
        raise ValueError("experiment_type not recognized.")


def experimental_points_runner(**kwargs):
    _, file_path = check_saved(**kwargs)

    double_noise = not float(kwargs.get("percentage", 0.0)) == 0.0
    decorrelated_noise = not (kwargs.get("beta", 1.0) == 1.0)

    if decorrelated_noise:
        measure_fun_args = (
            kwargs["delta_small"],
            kwargs["delta_large"],
            kwargs["percentage"],
            kwargs["beta"],
        )
        error_function = num.measure_gen_decorrelated
    else:
        if double_noise:
            error_function = num.measure_gen_double
            measure_fun_args = (
                kwargs["delta_small"],
                kwargs["delta_large"],
                kwargs["percentage"],
            )
        else:
            error_function = num.measure_gen_single
            measure_fun_args = (kwargs["delta"],)

    if kwargs["loss_name"] == "Huber":
        find_coefficients_fun_args = (kwargs["reg_param"], kwargs["a"])
    else:
        find_coefficients_fun_args = (kwargs["reg_param"],)

    alphas, errors_mean, errors_std = num.generate_different_alpha(
        error_function,
        _loss_type_chose(
            kwargs["loss_name"],
            values=[
                num.find_coefficients_L2,
                num.find_coefficients_L1,
                num.find_coefficients_Huber,
                -1,
            ],
        ),
        alpha_1=kwargs["alpha_min"],
        alpha_2=kwargs["alpha_max"],
        n_features=kwargs["n_features"],
        n_alpha_points=kwargs["alpha_pts"],
        repetitions=kwargs["repetitions"],
        measure_fun_args=measure_fun_args,
        find_coefficients_fun_args=find_coefficients_fun_args,
    )

    kwargs.update(
        {
            "file_path": file_path,
            "alphas": alphas,
            "errors_mean": errors_mean,
            "errors_std": errors_std,
        }
    )

    save_file(**kwargs)


def AMP_points_runner(**kwargs):
    _, file_path = check_saved(**kwargs)

    double_noise = not float(kwargs.get("percentage", 0.0)) == 0.0
    decorrelated_noise = not (kwargs.get("beta", 1.0) == 1.0)

    if decorrelated_noise:
        measure_fun_args = (
            kwargs["delta_small"],
            kwargs["delta_large"],
            kwargs["percentage"],
            kwargs["beta"],
        )
        error_function = num.measure_gen_decorrelated
        amp_function = num.find_coefficients_AMP_decorrelated_noise
        find_coefficients_fun_args = (
            kwargs["delta_small"],
            kwargs["delta_large"],
            kwargs["percentage"],
            kwargs["beta"],
        )
    else:
        if double_noise:
            error_function = num.measure_gen_double
            measure_fun_args = (
                kwargs["delta_small"],
                kwargs["delta_large"],
                kwargs["percentage"],
            )
            amp_function = num.find_coefficients_AMP_double_noise
            find_coefficients_fun_args = (
                kwargs["delta_small"],
                kwargs["delta_large"],
                kwargs["percentage"],
            )
        else:
            error_function = num.measure_gen_single
            measure_fun_args = (kwargs["delta"],)
            amp_function = num.find_coefficients_AMP_single_noise
            find_coefficients_fun_args = (kwargs["delta"],)

    alphas, errors_mean, errors_std = num.generate_different_alpha(
        error_function,
        amp_function,
        alpha_1=kwargs["alpha_min"],
        alpha_2=kwargs["alpha_max"],
        n_features=kwargs["n_features"],
        n_alpha_points=kwargs["alpha_pts"],
        repetitions=kwargs["repetitions"],
        measure_fun_args=measure_fun_args,
        find_coefficients_fun_args=find_coefficients_fun_args,
    )

    kwargs.update(
        {
            "file_path": file_path,
            "alphas": alphas,
            "errors_mean": errors_mean,
            "errors_std": errors_std,
        }
    )

    save_file(**kwargs)


def theory_curve_runner(**kwargs):
    _, file_path = check_saved(**kwargs)

    double_noise = not float(kwargs.get("percentage", 0.0)) == 0.0
    decorrelated_noise = not (kwargs.get("beta", 1.0) == 1.0)

    if decorrelated_noise:
        var_hat_kwargs = {
            "delta_small": kwargs["delta_small"],
            "delta_large": kwargs["delta_large"],
            "percentage": kwargs["percentage"],
            "beta": kwargs["beta"],
        }

        delta_small = kwargs["delta_small"]
        delta_large = kwargs["delta_large"]

        while True:
            m = 0.89 * np.random.random() + 0.1
            q = 0.89 * np.random.random() + 0.1
            sigma = 0.89 * np.random.random() + 0.1
            if np.square(m) < q + delta_small * q and np.square(m) < q + delta_large * q:
                initial_condition = [m, q, sigma]
                break

        var_functions = [
            var_hat_func_L2_decorrelated_noise,
            var_hat_func_L1_decorrelated_noise,  # var_hat_func_L1_num_decorrelated_noise,
            var_hat_func_Huber_decorrelated_noise,  # var_hat_func_Huber_num_decorrelated_noise,
            -1,
        ]
    else:
        if double_noise:
            delta_small = kwargs["delta_small"]
            delta_large = kwargs["delta_large"]

            var_hat_kwargs = {
                "delta_small": delta_small,
                "delta_large": delta_large,
                "percentage": kwargs["percentage"],
            }

            while True:
                m = 0.89 * np.random.random() + 0.1
                q = 0.89 * np.random.random() + 0.1
                sigma = 0.89 * np.random.random() + 0.1
                if np.square(m) < q + delta_small * q and np.square(m) < q + delta_large * q:
                    initial_condition = [m, q, sigma]
                    break

            var_functions = [
                var_hat_func_L2_double_noise,
                var_hat_func_L1_double_noise,  # var_hat_func_L1_num_double_noise,
                var_hat_func_Huber_double_noise,  # var_hat_func_Huber_num_double_noise,
                -1,
            ]
        else:
            var_hat_kwargs = {"delta": kwargs["delta"]}

            delta = kwargs["delta"]

            while True:
                m = 0.89 * np.random.random() + 0.1
                q = 0.89 * np.random.random() + 0.1
                sigma = 0.89 * np.random.random() + 0.1
                if np.square(m) < q + delta * q:
                    initial_condition = [m, q, sigma]
                    break

            var_functions = [
                var_hat_func_L2_single_noise,
                var_hat_func_L1_single_noise,  # var_hat_func_L1_num_single_noise,
                var_hat_func_Huber_single_noise,  # var_hat_func_Huber_num_single_noise,
                -1,
            ]

    if _loss_type_chose(kwargs["loss_name"], values=[False, False, True, False]):
        var_hat_kwargs.update({"a": kwargs["a"]})

    alphas, [errors] = fpe.no_parallel_different_alpha_observables_fpeqs(
        var_func_L2,
        _loss_type_chose(kwargs["loss_name"], values=var_functions),
        alpha_1=kwargs["alpha_min"],
        alpha_2=kwargs["alpha_max"],
        n_alpha_points=kwargs["alpha_pts"],
        reg_param=kwargs["reg_param"],
        initial_cond=initial_condition,
        var_hat_kwargs=var_hat_kwargs,
    )

    kwargs.update(
        {
            "file_path": file_path,
            "alphas": alphas,
            "errors": errors,
        }
    )

    save_file(**kwargs)


def bayes_optimal_runner(**kwargs):
    _, file_path = check_saved(**kwargs)

    double_noise = not float(kwargs.get("percentage", 0.0)) == 0.0
    decorrelated_noise = kwargs.get("beta") is not None

    if decorrelated_noise:
        var_hat_kwargs = {
            "delta_small": kwargs["delta_small"],
            "delta_large": kwargs["delta_large"],
            "percentage": kwargs["percentage"],
            "beta": kwargs["beta"],
        }

        delta_small = kwargs["delta_small"]
        delta_large = kwargs["delta_large"]

        while True:
            m = 0.89 * np.random.random() + 0.1
            q = 0.89 * np.random.random() + 0.1
            sigma = 0.89 * np.random.random() + 0.1
            if np.square(m) < q + delta_small * q and np.square(m) < q + delta_large * q:
                initial_condition = [m, q, sigma]
                break

        var_function = var_hat_func_BO_num_decorrelated_noise
    else:
        if double_noise:
            var_hat_kwargs = {
                "delta_small": kwargs["delta_small"],
                "delta_large": kwargs["delta_large"],
                "percentage": kwargs["percentage"],
            }

            delta_small = kwargs["delta_small"]
            delta_large = kwargs["delta_large"]

            while True:
                m = 0.89 * np.random.random() + 0.1
                q = 0.89 * np.random.random() + 0.1
                sigma = 0.89 * np.random.random() + 0.1
                if np.square(m) < q + delta_small * q and np.square(m) < q + delta_large * q:
                    initial_condition = [m, q, sigma]
                    break

            var_function = var_hat_func_BO_num_double_noise
        else:
            var_hat_kwargs = {"delta": kwargs["delta"]}

            delta = kwargs["delta"]

            while True:
                m = 0.89 * np.random.random() + 0.1
                q = 0.89 * np.random.random() + 0.1
                sigma = 0.89 * np.random.random() + 0.1
                if np.square(m) < q + delta * q:
                    initial_condition = [m, q, sigma]
                    break

            var_function = var_hat_func_BO_single_noise

    alphas, (errors,) = fpe.no_parallel_different_alpha_observables_fpeqs(
        var_func_BO,
        var_function,
        alpha_1=kwargs["alpha_min"],
        alpha_2=kwargs["alpha_max"],
        n_alpha_points=kwargs["alpha_pts"],
        initial_cond=initial_condition,
        var_hat_kwargs=var_hat_kwargs,
    )

    print(errors)

    kwargs.update(
        {
            "file_path": file_path,
            "alphas": alphas,
            "errors": errors,
        }
    )

    save_file(**kwargs)


def reg_param_optimal_runner(**kwargs):
    _, file_path = check_saved(**kwargs)

    double_noise = not float(kwargs.get("percentage", 0.0)) == 0.0
    decorrelated_noise = kwargs.get("beta") is not None

    if decorrelated_noise:
        var_hat_kwargs = {
            "delta_small": kwargs["delta_small"],
            "delta_large": kwargs["delta_large"],
            "percentage": kwargs["percentage"],
            "beta": kwargs["beta"],
        }

        delta_small = kwargs["delta_small"]
        delta_large = kwargs["delta_large"]

        while True:
            m = 0.89 * np.random.random() + 0.1
            q = 0.89 * np.random.random() + 0.1
            sigma = 0.89 * np.random.random() + 0.1
            if np.square(m) < q + delta_small * q and np.square(m) < q + delta_large * q:
                initial_condition = [m, q, sigma]
                break

        var_functions = [
            var_hat_func_L2_decorrelated_noise,
            var_hat_func_L1_decorrelated_noise,
            var_hat_func_Huber_decorrelated_noise,
            -1,
        ]
    else:
        if double_noise:
            var_hat_kwargs = {
                "delta_small": kwargs["delta_small"],
                "delta_large": kwargs["delta_large"],
                "percentage": kwargs["percentage"],
            }

            delta_small = kwargs["delta_small"]
            delta_large = kwargs["delta_large"]

            while True:
                m = 0.89 * np.random.random() + 0.1
                q = 0.89 * np.random.random() + 0.1
                sigma = 0.89 * np.random.random() + 0.1
                if np.square(m) < q + delta_small * q and np.square(m) < q + delta_large * q:
                    initial_condition = [m, q, sigma]
                    break

            var_functions = [
                var_hat_func_L2_double_noise,
                var_hat_func_L1_double_noise,
                var_hat_func_Huber_double_noise,
                -1,
            ]
        else:
            var_hat_kwargs = {"delta": kwargs["delta"]}
            delta = kwargs["delta"]

            while True:
                m = 0.89 * np.random.random() + 0.1
                q = 0.89 * np.random.random() + 0.1
                sigma = 0.89 * np.random.random() + 0.1
                if np.square(m) < q + delta * q:
                    initial_condition = [m, q, sigma]
                    break

            var_functions = [
                var_hat_func_L2_single_noise,
                var_hat_func_L1_single_noise,
                var_hat_func_Huber_single_noise,
                -1,
            ]

    if _loss_type_chose(kwargs["loss_name"], values=[False, False, True, False]):
        var_hat_kwargs.update({"a": kwargs["a"]})

    alphas, errors, lambdas = optimal_lambda(
        var_func_L2,
        _loss_type_chose(kwargs["loss_name"], values=var_functions),
        alpha_1=kwargs["alpha_min"],
        alpha_2=kwargs["alpha_max"],
        n_alpha_points=kwargs["alpha_pts"],
        initial_cond=initial_condition,
        var_hat_kwargs=var_hat_kwargs,
    )

    kwargs.update(
        {
            "file_path": file_path,
            "alphas": alphas,
            "errors": errors,
            "lambdas": lambdas,
        }
    )

    save_file(**kwargs)


def reg_param_optimal_experiment_runner(**kwargs):
    n_pts_theoretical = kwargs["alpha_pts_theoretical"]
    n_pts_experimental = kwargs["alpha_pts_experimental"]

    theoretical_exp_dict = kwargs.copy()
    theoretical_exp_dict.update(
        {
            "experiment_type": "reg_param optimal",
            "alpha_pts": n_pts_theoretical,
        }
    )
    file_exists, file_path = check_saved(**theoretical_exp_dict)

    if not file_exists:
        raise RuntimeError("The file corresponding to the experiment do not exists.")

    theoretical_exp_dict.update(
        {
            "file_path": file_path,
        }
    )
    alphas_theoretical, _, lambdas_theoretical = load_file(**theoretical_exp_dict)

    idxs = _get_spaced_index(alphas_theoretical, n_pts_experimental)

    alphas_experimental = alphas_theoretical[idxs]
    lambdas_experimental = lambdas_theoretical[idxs]

    double_noise = not float(kwargs.get("percentage", 0.0)) == 0.0
    decorrelated_noise = not (kwargs.get("beta", 1.0) == 1.0)

    if decorrelated_noise:
        measure_fun_kwargs = {
            "delta_small": kwargs["delta_small"],
            "delta_large": kwargs["delta_large"],
            "percentage": kwargs["percentage"],
            "beta": kwargs["beta"],
        }
        error_function = num.measure_gen_decorrelated
    else:
        if double_noise:
            error_function = num.measure_gen_double
            measure_fun_kwargs = {
                "delta_small": kwargs["delta_small"],
                "delta_large": kwargs["delta_large"],
                "percentage": kwargs["percentage"],
            }
        else:
            error_function = num.measure_gen_single
            measure_fun_kwargs = {"delta": kwargs["delta"]}

    if kwargs["loss_name"] == "Huber":
        find_coefficients_fun_kwargs = {"a": kwargs["a"]}
    else:
        find_coefficients_fun_kwargs = {}

    _, errors_mean, errors_std = num.generate_different_alpha(
        error_function,
        _loss_type_chose(
            kwargs["loss_name"],
            values=[
                num.find_coefficients_L2,
                num.find_coefficients_L1,
                num.find_coefficients_Huber,
                -1,
            ],
        ),
        alpha_1=kwargs["alpha_min"],
        alpha_2=kwargs["alpha_max"],
        n_features=kwargs["n_features"],
        repetitions=kwargs["repetitions"],
        reg_param=lambdas_experimental,
        measure_fun_kwargs=measure_fun_kwargs,
        find_coefficients_fun_kwargs=find_coefficients_fun_kwargs,
        alphas=alphas_experimental,
    )

    kwargs.update(
        {
            "alpha_pts": n_pts_experimental,
        }
    )

    _, file_path = check_saved(**kwargs)

    kwargs.update(
        {
            "file_path": file_path,
            "alphas": alphas_experimental,
            "errors_mean": errors_mean,
            "errors_std": errors_std,
            "lambdas": lambdas_experimental,
        }
    )

    save_file(**kwargs)


def reg_param_and_huber_param_optimal_runner(**kwargs):
    _, file_path = check_saved(**kwargs)

    double_noise = not float(kwargs.get("percentage", 0.0)) == 0.0
    decorrelated_noise = kwargs.get("beta") is not None

    if decorrelated_noise:
        var_hat_kwargs = {
            "delta_small": kwargs["delta_small"],
            "delta_large": kwargs["delta_large"],
            "percentage": kwargs["percentage"],
            "beta": kwargs["beta"],
        }

        delta_small = kwargs["delta_small"]
        delta_large = kwargs["delta_large"]

        while True:
            m = 0.89 * np.random.random() + 0.1
            q = 0.89 * np.random.random() + 0.1
            sigma = 0.89 * np.random.random() + 0.1
            if np.square(m) < q + delta_small * q and np.square(m) < q + delta_large * q:
                initial_condition = [m, q, sigma]
                break

        var_hat_func = var_hat_func_Huber_decorrelated_noise
    else:
        if double_noise:
            var_hat_kwargs = {
                "delta_small": kwargs["delta_small"],
                "delta_large": kwargs["delta_large"],
                "percentage": kwargs["percentage"],
            }

            delta_small = kwargs["delta_small"]
            delta_large = kwargs["delta_large"]

            while True:
                m = 0.89 * np.random.random() + 0.1
                q = 0.89 * np.random.random() + 0.1
                sigma = 0.89 * np.random.random() + 0.1
                if np.square(m) < q + delta_small * q and np.square(m) < q + delta_large * q:
                    initial_condition = [m, q, sigma]
                    break

            var_hat_func = var_hat_func_Huber_double_noise
        else:
            var_hat_kwargs = {"delta": kwargs["delta"]}
            delta = kwargs["delta"]

            while True:
                m = 0.89 * np.random.random() + 0.1
                q = 0.89 * np.random.random() + 0.1
                sigma = 0.89 * np.random.random() + 0.1
                if np.square(m) < q + delta * q:
                    initial_condition = [m, q, sigma]
                    break

            var_hat_func = var_hat_func_Huber_single_noise

    (alphas, errors, lambdas, huber_params,) = no_parallel_optimal_reg_param_and_huber_parameter(
        var_hat_func=var_hat_func,
        alpha_1=kwargs["alpha_min"],
        alpha_2=kwargs["alpha_max"],
        n_alpha_points=kwargs["alpha_pts"],
        initial_cond=initial_condition,
        var_hat_kwargs=var_hat_kwargs,
    )

    kwargs.update(
        {
            "file_path": file_path,
            "alphas": alphas,
            "errors": errors,
            "lambdas": lambdas,
            "huber_params": huber_params,
        }
    )

    save_file(**kwargs)


def reg_param_and_huber_param_experimental_optimal_runner(**kwargs):
    n_pts_theoretical = kwargs["alpha_pts_theoretical"]
    n_pts_experimental = kwargs["alpha_pts_experimental"]

    theoretical_exp_dict = kwargs.copy()
    theoretical_exp_dict.update(
        {
            "experiment_type": "reg_param huber_param optimal",
            "alpha_pts": n_pts_theoretical,
        }
    )
    file_exists, file_path = check_saved(**theoretical_exp_dict)

    if not file_exists:
        raise RuntimeError("The file corresponding to the experiment do not exists.")

    theoretical_exp_dict.update(
        {
            "file_path": file_path,
        }
    )

    alphas_theoretical, _, lambdas_theoretical, huber_params_theoretical = load_file(
        **theoretical_exp_dict
    )

    idxs = _get_spaced_index(alphas_theoretical, n_pts_experimental)
    alphas_experimental = alphas_theoretical[idxs]
    lambdas_experimental = lambdas_theoretical[idxs]
    huber_params_experimental = huber_params_theoretical[idxs]

    print("max alpha", np.max(alphas_experimental))

    double_noise = not float(kwargs.get("percentage", 0.0)) == 0.0
    decorrelated_noise = not (kwargs.get("beta", 1.0) == 1.0)

    if decorrelated_noise:
        measure_fun_kwargs = {
            "delta_small": kwargs["delta_small"],
            "delta_large": kwargs["delta_large"],
            "percentage": kwargs["percentage"],
            "beta": kwargs["beta"],
        }
        error_function = num.measure_gen_decorrelated
    else:
        if double_noise:
            error_function = num.measure_gen_double
            measure_fun_kwargs = {
                "delta_small": kwargs["delta_small"],
                "delta_large": kwargs["delta_large"],
                "percentage": kwargs["percentage"],
            }
        else:
            error_function = num.measure_gen_single
            measure_fun_kwargs = {"delta": kwargs["delta"]}

    find_coefficients_fun_kwargs = [{"a": a} for a in huber_params_experimental]

    _, errors_mean, errors_std = num.no_parallel_generate_different_alpha(
        error_function,
        num.find_coefficients_Huber,
        alpha_1=kwargs["alpha_min"],
        alpha_2=kwargs["alpha_max"],
        n_features=kwargs["n_features"],
        n_alpha_points=n_pts_experimental,
        repetitions=kwargs["repetitions"],
        reg_param=lambdas_experimental,
        measure_fun_kwargs=measure_fun_kwargs,
        find_coefficients_fun_kwargs=find_coefficients_fun_kwargs,
        alphas=alphas_experimental,
    )

    kwargs.update(
        {
            "alpha_pts": n_pts_experimental,
        }
    )

    _, file_path = check_saved(**kwargs)

    kwargs.update(
        {
            "file_path": file_path,
            "alphas": alphas_experimental,
            "errors_mean": errors_mean,
            "errors_std": errors_std,
            "lambdas": lambdas_experimental,
            "huber_params": huber_params_experimental,
        }
    )

    save_file(**kwargs)
