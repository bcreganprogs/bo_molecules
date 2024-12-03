# Standard library imports
import random

# Local imports (grouped and relabeled for brevity)
from modules.benchmarking.metrics_and_benchmarking import (
    generate_benchmarking_plots, 
    print_benchmarking_metrics)
from modules.bayesian_optimisation.bayesian_optimisation import (
    bayesian_optimisation)
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
from modules.utils.read_sample_store import store_buffer


def main():
    """Bayesian Optimisation Process (Example)"""    
    # Set random seed for reproducibility
    random.seed()
    
    # Parse command line arguments or use default values
    args = parse_arguments()
    main_results_dir = args.results_dir
    experiment_config_filepath = args.config_filepath
    
    # Load experiment configuration
    config = load_experiment(experiment_config_filepath)   #YAML file
    max_oracle_calls = config['bayesian_optimisation']['n_trials']
    oracle_name = config['objective_function']['oracle_name']   # eg.'qm9'

    # Setup directories for storing results
    results_dir = create_results_dir(main_results_dir, oracle_name, max_oracle_calls)
    copy_yaml_to_results_dir(experiment_config_filepath, results_dir)
    
    # Load initial dataset and model components
    intial_graphs = load_initial_dataset(config)   # eg. 1000 ZINC molecules
    surrogate_model = load_surrogate_model(config)   # eg. Gaussian Process
    acquisition_function = load_acquisition_function(config)   # eg. Expected Improvement
    acquisition_function_optimiser = load_acquisition_function_optimiser(config)   # eg. Genetic Algorithm
    objective_function = load_objective_function(config)
    graph_kernel = load_graph_kernel(config)

    # Load benchmarking metrics
    benchmarking_metrics = load_benchmarking_metrics(config, results_dir)

    # Perform Bayesian Optimisation
    observed_oracle_scores, buffer = bayesian_optimisation (
        intial_graphs, 
        max_oracle_calls,
        surrogate_model=surrogate_model, 
        acquisition_function=acquisition_function,
        acquisition_function_optimiser=acquisition_function_optimiser, 
        objective_function=objective_function,
        kernel=graph_kernel, 
        benchmarking_metrics=benchmarking_metrics)
    
    # Print and plot benchmarking metrics
    print_benchmarking_metrics(benchmarking_metrics, 
                               best_observed = observed_oracle_scores, 
                               buffer = buffer,
                               current_iteration= max_oracle_calls,
                               save_as_json = True)

    generate_benchmarking_plots(benchmarking_metrics, 
                                best_observed = observed_oracle_scores,
                                interim = False)

    # Store molecules and scores if configured
    if config['storage']['store_oracle_values_and_smiles']:
        store_buffer(buffer, benchmarking_metrics)


if __name__ == '__main__':
    main()