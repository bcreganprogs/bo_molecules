# Standard library imports
import random

# Local imports (grouped and relabeled for brevity)
from modules.benchmarking.metrics_and_benchmarking import (
    generate_benchmarking_plots, 
    print_benchmarking_metrics)
from modules.utils.read_sample_store import store_buffer
from modules.bayesian_optimisation.bayesian_optimisation import bayesian_optimisation
from modules.utils.yaml_utils import (
    parse_arguments,
    create_results_dir,
    copy_yaml_to_results_dir, 
    load_experiment,
    load_initial_dataset,
    load_surrogate_model,
    load_acquisition_function,
    load_acquisition_function_optimiser,
    load_objective_function,
    load_graph_kernel,
    load_benchmarking_metrics)
import optuna
import pandas as pd
import numpy as np

def mean_of_top_n(array, n ) -> float:
    """
    Calculate the mean of the top n values in the given array.
    
    Parameters:
    - array: List array of observed values.
    - n: The number of top values to consider for calculating the mean.
    
    Returns:
    - The mean of the top n values.
    """
    if n > len(array):
        mean_top_n = 0
    # Sort the array in descending order and take the first n elements
    top_n = np.sort(array)[-n:]
    # Compute the mean of these top n values
    mean_top_n = np.mean(top_n)
    
    return mean_top_n




def objective(trial: optuna.trial.Trial) -> float:
    """Bayesian Optimisation Process (Example)"""
    # Enable debug mode if necessary
    # debug._set_state(True)
    
    # Set random seed for reproducibility
    random.seed(0)
    
    # Parse command line arguments or use default values
    args = parse_arguments()
    main_results_dir = args.results_dir
    experiment_config_filepath = args.config_filepath
    
    # Load experiment configuration
    config = load_experiment(experiment_config_filepath)   #YAML file
    
    config['ga_optimiser']['n_parents'] = trial.suggest_int('n_parents', 1, 1000, log=True)
    config['ga_optimiser']['n_offspring'] = trial.suggest_int('n_offspring', 1, 1000, log=True)
    config['ga_optimiser']['mutation_rate'] = trial.suggest_float('mutation_rate', 0, 1)
    
    max_oracle_calls = config['bayesian_optimisation']['n_trials']   # PMO sets 10000
    oracle_name = config['objective_function']['oracle_name']   # eg.'qm9'


    # Load initial dataset and model components
    intial_graphs = load_initial_dataset(config)   # eg. 1000 ZINC molecules
    surrogate_model = load_surrogate_model(config)   # eg. Gaussian Process
    acquisition_function = load_acquisition_function(config)   # eg. Expected Improvement
    acquisition_function_optimiser = load_acquisition_function_optimiser(config)   # eg. Genetic Algorithm
    objective_function = load_objective_function(config)
    graph_kernel = load_graph_kernel(config)


    # Setup acf using acquisition function
    if config['acquisition_function']['upper_confidence_bound_with_tuning']:
        acf = acquisition_function(epsilon = trial.suggest_float('ucb_epsilon', 1e-8, 1, log=True), asymptote = trial.suggest_float('ucb_asymptote', 0, 1), inital_value = trial.suggest_float('ucb_inital_value', 0, 2))

    # Load benchmarking metrics
    benchmarking_metrics = load_benchmarking_metrics(config, '')

    # Perform Bayesian Optimisation
    observed_oracle_scores, buffer = bayesian_optimisation (
                                        intial_graphs, 
                                        max_oracle_calls,
                                        surrogate_model=surrogate_model, 
                                        acf=acf,
                                        acquisition_function_optimiser=acquisition_function_optimiser, 
                                        objective_function=objective_function,
                                        kernel=graph_kernel, 
                                        benchmarking_metrics=benchmarking_metrics)
    
    results_top_10 = mean_of_top_n(observed_oracle_scores, 10)
    
    return results_top_10

if __name__ == '__main__':

    study_name = 'hp_optimization_studyv1'
    storage = 'sqlite:///hp_optimization_studyv1.db'

    try:
        # Try loading the existing study
        trial = optuna.load_study(study_name=study_name, storage=storage)
    except KeyError:
        # If the study doesn't exist, create it
        trial = optuna.create_study(study_name=study_name, storage=storage, direction='minimize')  # or 'maximize'


    trial.optimize(objective, n_trials=None, timeout=(60*60*8), n_jobs=-1, catch=(Exception,))
    
    
    print(trial.best_trial.value)
    print(trial.best_trial.params)
    print(trial)
    
    df = trial.trials_dataframe()
    df.to_csv('resultsv2.csv')
    