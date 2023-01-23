# Robust Linear Regression

## Code Organization

The idea behind the code is based on having functions that let you preform the simulations that you want to use. The principal files are the following:

- `fpeqs.py` contains all of the functions to run simulations for the fixed point equations
- `numerics.py` contains the functions to run the numerical experiments 
- `numerical_functions.py` mostly contains all the functions to perform numerical integrations and that are called from `fpeqs.py`.
- `utils.py` contains functions to load and save simulations data
- `integration_utils.py` contains functions that are used to perform integrations

All of the functions used to save and load files are contained in `src.py`.

The folder `src_cluster` contains all files that only reference themselves and present a parallel implementation. These files are also working on minimal dependances (`numpy`, `scipy`, `sklearn`) and are meant to be runned in parallel.

## Code usability

An example of code is presented in the file `together_theory_experiment.py`. One can create a dictionary with all the parameters of the simulations and then call the routine in `utils` that runs and saves the experiment. 

## Tests

To run the tests for the code one should run the file `unittests/run_tests.py`.