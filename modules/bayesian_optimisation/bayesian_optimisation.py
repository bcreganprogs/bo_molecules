# Standard library imports
import time
from typing import List, Tuple, Callable

# Third party imports
import gpytorch
import numpy as np
import torch
from torch.nn import functional as F
from tqdm import tqdm
from botorch.exceptions.errors import ModelFittingError
import networkx as nx
import os

# Local imports
from modules.surrogate_models.gp_model import initialize_model, optimize_mll
from modules.utils.molecular_data_conversion import (
    graph_to_smiles, is_valid_molecule)
from modules.benchmarking.metrics_and_benchmarking import (
    generate_benchmarking_plots, print_benchmarking_metrics)
from modules.utils.read_sample_store import (
    store_buffer, store_surrogate_model, store_offspring)
from modules.DGK_Graphormer.Graphormer_DGK import (
    GraphormerGraphEncoder, get_embedings)


def evaluate_acquisition_function(acquisition_function: object, 
                                  acquisition_function_optimiser: object, 
                                  fitness_function: List[Tuple[nx.Graph, float]]
                                  ) -> Tuple[nx.Graph, List[nx.Graph]]:
    """
    Optimize acquisition function and return selected graph.
    
    Optimizes the acquisition function using the specified optimizer and 
    returns the observation point that maximizes the acquisition function.

    Args:
        acquisition_function: The acquisition function to be optimized.
        acquisition_function_optimiser: The optimizer used to optimize the 
                                        acquisition function. Supports 
                                        GeneticAlgorithm, DataSampler,
                                        and Transformer classes.
        fitness_function: The fitness function used by the optimizer.

    Returns:
        selected_graph: The observation point that maximizes the 
                        acquisition function; is selected for evaluation.
        offspring: List of network_x Graphs
    """
    # Loop over the discrete set of points and evaluate the acquisition 
    # function optimiser produces graphs to evaluate
    acquisition_function_optimiser.fitness_function = sorted(fitness_function, 
                                                             key=lambda x: x[1], 
                                                             reverse=True)
    offspring = acquisition_function_optimiser.sample()

    acquisition_function_values = acquisition_function(offspring).detach()

    ind = torch.argmax(acquisition_function_values)
    selected_graph = offspring[ind]

    return selected_graph, offspring

def bayesian_optimisation(
    x_train: List[nx.Graph],
    n_trials: int,
    kernel: object,
    acf: Callable,
    acquisition_function_optimiser: object,
    surrogate_model: object,
    objective_function: Callable,
    benchmarking_metrics: dict) -> Tuple[List[float], dict]:
    """
    Performs Bayesian optimisation on initial x_train graphs.
    
    The buffer is used to store the observed objective function values and
    the iteration number. This is used to calculate the benchmarking metrics.
    It is adapted from Wenhao Gao's PMO code and the same as the mol_buffer
    in the PMO code.

    Args:
        x_train: List of networkx graphs for training.
        n_trials: Number of random trials to execute.
        kernel: Kernel used for the GP model.
        acf: The acquisition function employed already initalized.
        acquisition_function_optimiser: Optimiser for the acquisition function.
        surrogate_model: GP model utilized.
        objective_function: Objective function applied.
        benchmarking_metrics: Dictionary containing benchmarking metrics.

    Returns:
        obj_fun_scores: Observed objective function values
                                   for each iteration.
        buffer: Dictionary containing observed objective function values 
                mapped to iteration numbers.
    """

    # Used to store the molecules and their oracle score
    obj_fun_scores = []
    molecules_as_graph, molecules_as_smiles = [], []
    
    
    # Calculate the objective function values for the initial training points
    # for the acquisition funciton optimizer
    fitness_function = []
    for mol_as_graph in x_train:
        mol_as_smiles = graph_to_smiles(mol_as_graph)
        oracle_score = objective_function(mol_as_smiles)
        
        # Needed for acquition function evaluation
        fitness_function.append((mol_as_graph, oracle_score))
        
        # Needed for benchmarking and storing
        molecules_as_graph.append(mol_as_graph)
        molecules_as_smiles.append(mol_as_smiles)
        obj_fun_scores.append(oracle_score)
    
    # Calculate initial objective function scores
    y_train = torch.tensor(obj_fun_scores).flatten().float()
    
    # This is a buffer to store the objective function values and the iteration
    # number. This is the same as the mol_buffer in the PMO code. This is used
    # to calculate the benchmarking metrics.
    buffer = dict()  

    # if using DGK, we need to store the graph embeddings
    if isinstance(kernel, GraphormerGraphEncoder):
        isDeepKernel = True
    else:
        isDeepKernel = False
    model = surrogate_model(kernel=kernel)
    
    #move to gpu
    if isDeepKernel: # if deepkernel
        x_train = get_embedings(kernel, x_train)
        x_train = F.normalize(x_train, p=2, dim=-1)
        x_train = x_train.to('cuda' if torch.cuda.is_available() else 'cpu')
        y_train = y_train.to('cuda' if torch.cuda.is_available() else 'cpu')
    else: # if traditional kernel
        x_train = [i for i in molecules_as_graph] # list of networkx graphs
  
    model.fit(x_train, y_train)
    oracle_calls_so_far = len(x_train)
    progress_bar = tqdm(range(oracle_calls_so_far, n_trials), smoothing=0.1)
    
    for iteration in progress_bar:
        
        progress_bar.set_description(f"Running iteration: #{iteration}. Best score: {np.max(obj_fun_scores).round(4)}.")
      
        t0 = time.time()

        # Update acquisition function with new model, best value and iteration count
        # do we need to define acf each iteration?
        # defeintiley not so we can move this out of the loop and pass it as 
        # an argument updating it with iteration count, best value and model


        # acf = acquisition_function(model=model, 
        #                            best_f=y_train.max().to('cuda' if torch.cuda.is_available() else 'cpu'), 
        #                            iteration_count=iteration)
        
        acf.model = model
        acf.best_f = y_train.max().to('cuda' if torch.cuda.is_available() else 'cpu')
        acf.iteration_count = iteration
  
        # Find graph which maximises acquisition function
        new_mol_as_graph, offspring_graphs = evaluate_acquisition_function(
            acquisition_function=acf,
            acquisition_function_optimiser=acquisition_function_optimiser, 
            fitness_function=fitness_function)
   
        # If molecule is valid evaluate and append to history
        if new_mol_as_graph is not None and is_valid_molecule(new_mol_as_graph):
      
            new_mol_as_smiles = graph_to_smiles(new_mol_as_graph)
            new_mol_objective_score = objective_function(new_mol_as_smiles)
            
            fitness_function.append((new_mol_as_graph, new_mol_objective_score))
    
            # Append for benchmarking and storing
            molecules_as_graph.append(new_mol_as_graph)
            molecules_as_smiles.append(new_mol_as_smiles)
            obj_fun_scores.append(new_mol_objective_score)
    
            # Add the new oracle call value and iteration number to the buffer
            buffer[new_mol_as_smiles] = [new_mol_objective_score, iteration]

            # Update x_train_tens and y_train
            if isDeepKernel:
                new_graph = get_embedings(kernel, [new_mol_as_graph])
                new_graph = F.normalize(new_graph, p=2, dim=-1)
                x_train = torch.cat((x_train, new_graph))
                y_train = torch.cat((
                    y_train, 
                    torch.tensor([new_mol_objective_score]).flatten().float().to('cuda' if torch.cuda.is_available() else 'cpu')))
            else: # traditional kernels need to store graph objects not embeddings
                # add new graph to x_train
                x_train.extend([new_mol_as_graph])
                new_graph = new_mol_as_graph

            
            # Update the model
            model.fit(new_graph, torch.tensor([new_mol_objective_score]).flatten().float().to('cuda' if torch.cuda.is_available() else 'cpu'))

        t1 = time.time()

        print(f"Time for iteration {iteration}: {t1-t0} seconds")
        print(f'Current oracle call - score: {float(new_mol_objective_score)}')
        print(f'Current oracle call - molecule: {new_mol_as_smiles}')
        print(f'Current best score: {np.max(obj_fun_scores)}')
        print(f'Current best molecule: {np.argmax(obj_fun_scores)}')
        
        if (benchmarking_metrics['interim_benchmarking'] and
            iteration % benchmarking_metrics['interim_benchmarking_freq'] == 0
            and iteration != 0):
            
            print_benchmarking_metrics(benchmarking_metrics, 
                                       best_observed = obj_fun_scores, 
                                       buffer = buffer, 
                                       current_iteration=iteration,
                                       save_as_json=False)
            
            generate_benchmarking_plots(benchmarking_metrics, 
                                        best_observed = obj_fun_scores,
                                        interim=True)
                
            store_buffer(buffer, benchmarking_metrics)
            
            
    return (obj_fun_scores, buffer)