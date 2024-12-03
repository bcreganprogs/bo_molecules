# Standard library imports
import random

# Local imports (grouped and relabeled for brevity)
from modules.benchmarking.benchmarking_pmo import (
    generate_benchmarking_plots, 
    print_benchmarking_metrics, 
    store_molecules_and_scores)
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


def looped_main(orcale_name):
    
    # Enable debug mode if necessary
    # debug._set_state(True)
    
    # Set random seed for reproducibility
    random.seed(0)
    
    # If no arguments are passed via the command line, use the default 
    # experiment configuration file and results directory
    args = parse_arguments()
    main_results_dir = args.results_dir
    experiment_config_filepath = args.config_filepath
    
    # Load the yaml file
    config = load_experiment(experiment_config_filepath)
    
    max_oracle_calls = 2000
    # oracle_names = ["gsk3b", 
    oracle_names = ["mestranol_similarity", 
                    "sitagliptin_mpo", 
                    "troglitazone_rediscovery", 
                    "median1",
                    'median2',
                    "osimertinib_mpo",
                    "celecoxib_rediscovery",
                    "jnk3",
                    "ranolazine_mpo"]
    
    for oracle_name in oracle_names:
        results_dir = create_results_dir(main_results_dir, oracle_name, max_oracle_calls)
        
        copy_yaml_to_results_dir(experiment_config_filepath, results_dir)
        
        intial_graphs = load_initial_dataset(config)
        
        surrogate_model = load_surrogate_model(config)
        
        acquisition_function = load_acquisition_function(config)
        
        acquisition_function_optimiser = load_acquisition_function_optimiser(config)
        
        objective_function = load_objective_function(config)
        
        graph_kernel = load_graph_kernel(config)
        
        benchmarking_metrics = load_benchmarking_metrics(config, results_dir)

        # Bayesian optimisation
        observed_oracle_scores, observed_smiles, buffer = bayesian_optimisation (
                                            intial_graphs, 
                                            max_oracle_calls,
                                            surrogate_model=surrogate_model, 
                                            acquisition_function=acquisition_function,
                                            acquisition_function_optimiser=acquisition_function_optimiser, 
                                            objective_function=objective_function,
                                            kernel=graph_kernel, 
                                            node_label="element", 
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
            store_molecules_and_scores(observed_oracle_scores,
                                    observed_smiles, 
                                    benchmarking_metrics["results_dir"], 
                                    compress_file=config['storage']['compress_file'])


if __name__ == '__main__':
    
    oracle_names = ["mestranol_similarity", 
                    "sitagliptin_mpo", 
                    "troglitazone_rediscovery", 
                    "median1",
                    'median2',
                    "osimertinib_mpo",
                    "celecoxib_rediscovery",
                    "jnk3",
                    "ranolazine_mpo"]
    
    for oracle_name in oracle_names:
        try:
            looped_main(oracle_name)
        except Exception as e:
            print(f"Error in {oracle_name}: {e}")
            continue
    