# Standard library imports
import os
import yaml
import argparse
from datetime import datetime
from typing import List, Callable

# Third-party imports
from tdc import Oracle
from modules.surrogate_models.kernel import ( 
    VertexHistogramKernel, EdgeHistogramKernel,
    WeisfeilerLehmanKernel, NeighborhoodHashKernel,
    RandomWalkKernel, ShortestPathKernel,
    WeisfeilerLehmanOptimalAssignmentKernel)
from modules.DGK_Graphormer import Graphormer_DGK
import networkx as nx

# Local imports
from modules.utils.read_sample_store import sample_graphs_from_smiles_csv
from modules.surrogate_models.gp_model import GraphGP, GPModel
from modules.acquisition_functions.acquisition_functions import (
        GraphExpectedImprovement, 
        GraphProbabilityOfImprovement, 
        GraphProbabilityOfImprovement, 
        GraphUpperConfidenceBound,
        EntropySearch,
        EntropySearchPortfolio,
        RandomSampler,
        GraphUpperConfidenceBoundWithTuning)
from modules.acquisition_function_optimisers.genetic_algorithm import GeneticAlgorithm
from modules.acquisition_function_optimisers.transformer import Transformer
from modules.acquisition_function_optimisers.dataset_sampler import DatasetSampler


def parse_arguments(default_config_filepath: str = 'experiments/configs/experiment.yaml',
                    default_results_dir: str = 'experiments/results'
                    ) -> argparse.Namespace:
    """ 
    Parse the command line arguments.
    
    Args:
        default_config_filepath: path to the experiment configuration file.
        default_results_dir: directory to store the results.
        
    Returns:
        argparse.Namespace: command line arguments.
    """
    parser = argparse.ArgumentParser(
        description='Run Bayesian Optimization experiment.')
    parser.add_argument('--config_filepath',
                        type=str, 
                        default = default_config_filepath, 
                        help='Path to the experiment configuration file.')
    parser.add_argument('--results-dir', 
                        type=str, 
                        default= default_results_dir, 
                        help='Path to save results.')
    return parser.parse_args()

def create_results_dir(main_results_dir: str, oracle_name: str,
                       max_oracle_calls: int) -> str:
    """ 
    Create a directory to store the results of the experiment.
    
    Args:
        main_results_dir: directory to store the results.
        oracle_name: name of the oracle.
        max_oracle_calls: maximum number of oracle calls.
    
    Returns:
        results_dir: directory to store the results of the experiment.
    """
    # Check if the main results directory/oracle_name exists
    main_results_dir = f'{main_results_dir}/{oracle_name}'
    if not os.path.exists(main_results_dir):
        os.makedirs(main_results_dir)
    
    date_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_dir = f'{main_results_dir}/{date_time}_{oracle_name}_{max_oracle_calls}'
    os.makedirs(results_dir)
    surrogate_model_dir = f'{results_dir}/surrogate_model_fitting'
    os.makedirs(surrogate_model_dir)
    
    return results_dir

def copy_yaml_to_results_dir(experiment_config_filepath: str,
                             results_dir: str) -> None:
    """ 
    Copy the experiment configuration file to the results directory.

    Args:
        experiment_config_filepath: path to the experiment configuration file.
        results_dir: directory to store the results.
    
    Returns:
        None
    """
    # Extract the base name of the file (e.g., 'experiment.yaml')
    experiment_name_with_ext = os.path.basename(experiment_config_filepath)
    
    # Copy the yaml file to the results directory
    new_yaml_path = f'{results_dir}/{experiment_name_with_ext}'
    os.system(f'cp {experiment_config_filepath} {new_yaml_path}')

def load_experiment(filepath: str) -> dict:
    """ 
    Load the experiment configuration file.
    
    Args:
        filepath: path to the experiment configuration file.
        
    Returns:
        dict: experiment configuration file as Python dict.
    """
    with open(filepath, 'r') as file:
        return yaml.safe_load(file)

def load_initial_dataset(config: dict) -> List[nx.Graph]:
    """
    Load the initial dataset for the experiment.
    
    Args:
        config: experiment configuration file.
        
    Returns:
        initial_dataset: initial dataset of NetworkX graphs for the experiment.
    """
    # Ensure that exactly one option is selected
    if (config['initial_dataset']['sample_from_zinc'] + 
        config['initial_dataset']['sample_from_transformer'] + 
        config['initial_dataset']['sample_from_custom_dataset']) != 1:
        raise ValueError("Exactly one initial dataset option must be selected.")
    
    # Ensure that the number of initial samples is less than number of trials
    n_initial_samples = config['bayesian_optimisation']['n_initial_samples']
    n_trials = config['bayesian_optimisation']['n_trials']
    if n_trials <= n_initial_samples:
        raise ValueError("The number of initial samples " +
                         "must be less than the number of trials.")
    
    if config['initial_dataset']['sample_from_zinc']:
        initial_dataset, _ = sample_graphs_from_smiles_csv(
            filepath='data/zinc_1000_sample.csv', 
            sample_size=config['bayesian_optimisation']['n_initial_samples'])
    elif config['initial_dataset']['sample_from_transformer']:
        sampler = Transformer(
            n_offspring=config['bayesian_optimisation']['n_initial_samples'])
        initial_dataset = sampler.sample()
    elif config['initial_dataset']['sample_from_custom_dataset']:
        if not config["initial_dataset"]["custom_dataset"].get(
            "filepath_to_custom_csv"):
            raise ValueError("A filepath is required,"+
                             "since custom_sample_csv is selected as True.")
        else:
            initial_dataset_config = config['initial_dataset']['custom_dataset']
            filepath_to_custom_csv = initial_dataset_config['custom_csv_path']
            try:
                initial_dataset, _ = sample_graphs_from_smiles_csv(
                    filepath = filepath_to_custom_csv, 
                    base_configs = config['bayesian_optimisation'],
                    sample_size = config['bayesian_optimisation']['n_initial_samples'])
            except FileNotFoundError:
                raise FileNotFoundError(
                    f"The file {filepath_to_custom_csv} was not found.")
    else:
        raise ValueError('No initial dataset specified')
    return initial_dataset

def load_surrogate_model(config: dict) -> object:
    """ 
    Load the surrogate model for the experiment.
    
    Args:
        config: experiment configuration file.
        
    Returns:
        surrogate_model: surrogate model for the experiment.
    """
    if config['surrogate_model']['model'] == 'graph_gp':
        surrogate_model = GPModel #GraphGP
    else:
        raise ValueError('No surrogate model specified')
    return surrogate_model

def load_acquisition_function(config: dict) -> object:
    """ 
    Load the acquisition function for the experiment.
    
    Args:
        config: experiment configuration file.
        
    Returns:
        acquisition_function: acquisition function for the experiment.
    """
    # Ensure that exactly one option is selected
    if (config['acquisition_function']['expected_improvement'] +
        config['acquisition_function']['probability_of_improvement'] +
        config['acquisition_function']['upper_confidence_bound'] +
        config['acquisition_function']['entropy_search'] +
        config['acquisition_function']['entropy_search_portfolio'] + 
        config['acquisition_function']['random'] +
        config['acquisition_function']['upper_confidence_bound_with_tuning']) != 1:
        raise ValueError(
            "Exactly one acquisition function must be selected as True.")
    
    if config['acquisition_function']['expected_improvement']:
        acquisition_function = GraphExpectedImprovement
    elif config['acquisition_function']['probability_of_improvement']:
        acquisition_function = GraphProbabilityOfImprovement
    elif config['acquisition_function']['upper_confidence_bound']:
        acquisition_function = GraphUpperConfidenceBound
    elif config['acquisition_function']['entropy_search']:
        acquisition_function = EntropySearch
    elif config['acquisition_function']['entropy_search_portfolio']:
        acquisition_function = EntropySearchPortfolio
    elif config['acquisition_function']['random']:
        acquisition_function = RandomSampler
    elif config['acquisition_function']['upper_confidence_bound_with_tuning']:
        acquisition_function = GraphUpperConfidenceBoundWithTuning
    else:
        raise ValueError('No acquisition function specified')
    return acquisition_function

def load_objective_function(config: dict) -> Callable:
    """ 
    Load the objective function for the experiment.
    
    Args:
        config: experiment configuration file.
        
    Returns:
        objective_function: objective function for the experiment.
    """
    oracle_name = config['objective_function']['oracle_name']
    try:
        objective_function = Oracle(name=oracle_name)
    except ValueError:
        raise ValueError(
            f'The objective function {oracle_name} could not be loaded.')

    return objective_function

def load_acquisition_function_optimiser(config: dict) -> object:
    """ 
    Load the acquisition function optimiser for the experiment.
    
    Args:
        config: experiment configuration file.
        
    Returns:
        acquisition_funciton_optimiser: acquisition function optimiser
                                        for the experiment.
    """
    # Ensure that exactly one option is selected
    if (config['acquisition_optimiser']['genetic_algorithm'] +
        config['acquisition_optimiser']['transformer'] +
        config['acquisition_optimiser']['dataset_sampler']) != 1:
        raise ValueError(
            "Exactly one acquisition function optimiser must be selected.")
    
    if config['acquisition_optimiser']['genetic_algorithm']:
        acquisition_funciton_optimiser = GeneticAlgorithm (
            n_parents=config['ga_optimiser']['n_parents'],
            n_offspring=config['ga_optimiser']['n_offspring'], 
            fitness_function=None,
            max_atoms=config['ga_optimiser']['max_atoms'],
            mutation_rate = config['ga_optimiser']['mutation_rate'],
            sampling=config['ga_optimiser']['sampling'])
    elif config['acquisition_optimiser']['transformer']:
        acquisition_funciton_optimiser = Transformer (
            n_offspring=config['transformer_optimiser']['n_offspring'])
    elif config['acquisition_optimiser']['dataset_sampler']:
        acquisition_funciton_optimiser = DatasetSampler (
            n_offspring=config['transformer_optimiser']['n_offspring'])
    else:
        raise ValueError('No acquisition function optimiser specified')
    return acquisition_funciton_optimiser

def load_graph_kernel(config: dict) -> object:
    """ 
    Load the graph kernel for the experiment.
    
    Args:
        config: experiment configuration file.
        
    Returns:
        graph_kernel: graph kernel for the experiment.
    """
    if config['graph_kernel']['kernel_name'] == 'random_walk':
        graph_kernel = RandomWalkKernel()
    elif config['graph_kernel']['kernel_name'] == 'shortest_path':
        graph_kernel = ShortestPathKernel(node_label='element')
    elif config['graph_kernel']['kernel_name'] == 'vertex_histogram':
        graph_kernel = VertexHistogramKernel(node_label='element')
    elif config['graph_kernel']['kernel_name'] == 'neighborhood_hash':
        graph_kernel = NeighborhoodHashKernel(node_label='element')
    elif config['graph_kernel']['kernel_name'] == 'weisfeiler_lehman':
        # optional edge label argument
        graph_kernel = WeisfeilerLehmanKernel(node_label='element', 
                                              edge_label='bond')
    elif config['graph_kernel']['kernel_name'] == 'weisfeiler_lehman_optimal_assignment':
        # optional edge label argument
        graph_kernel = WeisfeilerLehmanOptimalAssignmentKernel(
            node_label='element', edge_label='bond')
    elif config['graph_kernel']['kernel_name'] == 'edge_histogram':
        graph_kernel = EdgeHistogramKernel(edge_label='bond')
    elif config['graph_kernel']['kernel_name'] == 'Graphormer_DGK':
        graph_kernel = Graphormer_DGK.load_new_model()
    else:
        raise ValueError('No valid graph kernel specified')

    return graph_kernel

def load_benchmarking_metrics(config: dict, results_dir: str) -> dict:
    """ 
    Load the benchmarking metrics for the experiment.
    
    Args:
        config: experiment configuration file.
        results_dir: directory to store the results.
        
    Returns:
        benchmarking_metrics: benchmarking metrics for the experiment.
    """
    benchmarking_metrics = {
        "interim_benchmarking": config['benchmarking']['interim_benchmarking'],
        "interim_benchmarking_freq": config['benchmarking']['interim_benchmarking_freq'],
        "top_1_benchmark": config['benchmarking']['top_1_benchmark'],
        "top_10_benchmark": config['benchmarking']['top_10_benchmark'],
        "top_100_benchmark": config['benchmarking']['top_100_benchmark'],
        "plot_oracle_history": config['benchmarking']['plot_oracle_history'],
        "results_dir": results_dir,
        "oracle_name": config['objective_function']['oracle_name'],
        "max_oracle_calls": config['bayesian_optimisation']['n_trials']
    }
    return benchmarking_metrics