# Standard library imports
from typing import List, Tuple
import gzip
import shutil
from datetime import datetime
import os
import json
import pickle

# # Third-party imports
import pandas as pd
import random
import networkx as nx
import torch

# Local imports
from modules.utils.molecular_data_conversion import (
    smiles_to_graph, graph_to_smiles)


def read_in_zinc_table(filepath: str) -> pd.DataFrame:
    """
    Reading in ZINC dataset in tab format.
    
    Args:
        filepath: The path to the ZINC dataset.
        
    Returns:
        pd.DataFrame: The ZINC dataset.
    """
    return pd.read_csv(filepath, sep='\t')

def store_zinc_table_as_csv(df: pd.DataFrame, filepath: str, 
                            sample_size: int) -> None:
    """
    Store ZINC dataset in tab format.
    
    Args:
        df: The ZINC dataset.
        filepath: The path to the ZINC dataset.
        sample_size: The number of samples to store.
        
    Returns:
        None
    """
    df = df.sample(n=sample_size)
    df.to_csv(filepath, sep='\t', index=False)

def sample_graphs_from_smiles_csv(filepath: str, sample_size: int
                                  ) -> Tuple[List[nx.Graph], List[str]]:
    """
    Sample a subset of smiles and transform into Networkx graphs.
    
    Args:
        filepath: A filepath to a csv with one column of SMILES stings.
        sample_size: The number of graphs to sample.
        
    Returns:
        graphs: A list of sampled networkx graphs.
        smiles: A list of sampled SMILES strings.
    """
    df = pd.read_csv(filepath)
    raw_smiles = list(df['smiles'].sample(n=sample_size*2))
    graphs = []
    smiles = []
    while len(graphs) < sample_size:
        try:
            s = raw_smiles.pop()
            g = smiles_to_graph(s)
            # TODO: Do we still need this.
            s_reconstructed = graph_to_smiles(g)
            graphs.append(g)
            smiles.append(s)
        except:
            continue
    
    print(f'Sampled {len(graphs)} graphs from {filepath}')
    return graphs, smiles
    
def sample_smiles_from_smiles_csv(filepath: str, 
                                  sample_size: int) -> List[str]:
    """
    Sample a subset of smiles.
    
    Args:
        filepath: A filepath to a csv with one column of SMILES stings.
        sample_size: The number of graphs to sample.
        
    Returns:
        smiles: A list of sampled SMILES strings; the number of samples is
                double the sample_size to account for failed conversions.
    """
    df = pd.read_csv(filepath)
    # Sample twice as many SMILES as needed to account for failed conversions.
    smiles = list(df['smiles'].sample(n=sample_size*2))
    return smiles

def sample_smiles_from_pd_dataframe(df: pd.DataFrame, 
                                    sample_size: int) -> List[str]:
    """
    Sample a subset of smiles.
    
    Args:
        df: A pandas DataFrame with a column of SMILES strings.
        sample_size: The number of graphs to sample.
        
    Returns:
        smiles: A list of sampled SMILES strings.
    """
    smiles = list(df['smiles'].sample(n=sample_size))
    return smiles

def sample_graphs_from_smiles(smiles: List[str], 
                              sample_size: int, 
                              shuffle: bool = True) -> List[nx.Graph]:
    """ 
    Sample a subset of smiles and transform into Networkx graphs.
    
    Args:
        smiles: A list of SMILES strings.
        sample_size: The number of graphs to sample.
        shuffle: Whether to shuffle the list of SMILES strings.
        
    Returns:
        list: A list of sampled networkx graphs.
    """
    if shuffle:
        random.shuffle(smiles)
    
    graphs = [smiles_to_graph(s) for s in smiles[:sample_size]]

    return graphs

def sample_graphs_from_csv_smiles(filepath: str, 
                                  sample_size: int) -> List[nx.Graph]:
    """ 
    Sample a subset of smiles and transform into Networkx graphs.
    
    Args:
        filepath: A filepath to a csv with one column of SMILES stings.
        sample_size: The number of graphs to sample.
        
    Returns:
        graphs: A list of sampled networkx graphs.
    """
    df = pd.read_csv(filepath)
    smiles = list(df['smiles'].sample(n=sample_size))
    graphs = [smiles_to_graph(s) for s in smiles]

    return graphs
 
def store_buffer(buffer: dict, benchmarking_metrics: dict) -> None:
    """ 
    Store buffer and compress if needed.
    
    The format of the buffer is given by the Wenhao Gao's PMO implementation.

    Args:
        buffer: Dictionary with molecule graph as keys and 
                tuples of observed values and iteration numbers as values.
        benchmarking_metrics: Dictionary with benchmarking metrics.
        
    Returns:
        None
    """
    
    # Convert dictionary to JSON for storage
    buffer_json = json.dumps(buffer)

    # Define file paths
    file_path = os.path.join(benchmarking_metrics['results_dir'], 'buffer.json')
    compressed_file_path = file_path + ".gz"

    # Write the JSON data to a file
    with open(file_path, "w") as f:
        f.write(buffer_json)

    with open(file_path, "rb") as f_in:
        with gzip.open(compressed_file_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        
    # Remove the uncompressed file
    os.remove(file_path)
        
def read_compressed_or_regular_json(file_path: str) -> dict:
    """ 
    Read a JSON file that may be compressed or regular.
    
    Args:
        file_path: The path to the file to be read.
        
    Returns:
        dict: The data from the JSON file.
    """
    # Check if the file is compressed based on its extension
    if file_path.endswith('.gz'):
        # Open the gzip file
        with gzip.open(file_path, 'rt', encoding='utf-8') as file:
            data = json.load(file)
    else:
        # Open the regular JSON file
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

    return data

def buffer_to_dataframe(buffer_data: dict) -> pd.DataFrame:
    """ 
    Transform the loaded buffer data into a pandas DataFrame.

    Args:
        buffer_data: Dictionary with SMILES strings as keys and 
                     tuples of oracle scores and iteration numbers as values.

    Returns:
        df: DataFrame with columns: "SMILES", "oracle_score", "iteration".
    """
    # Initialize lists to store the columns data
    smiles_list = []
    oracle_scores = []
    iterations = []

    # Iterate over the buffer_data dictionary and populate the lists
    for smile, (oracle_score, iteration) in buffer_data.items():
        smiles_list.append(smile)
        oracle_scores.append(oracle_score)
        iterations.append(iteration)

    # Create the DataFrame
    df = pd.DataFrame({
        "smiles": smiles_list,
        "oracle_score": oracle_scores,
        "iteration": iterations
    })

    return df

def store_surrogate_model(benchmarking_metrics: dict, 
                          model: object, 
                          iteration: int) -> None:
    """ 
    Store the surrogate model state dictionary.
    
    Args:
        benchmarking_metrics: Dictionary with benchmarking metrics.
        model: The surrogate model to be stored.
        iteration: The iteration number of the optimisation process.
    
    Returns:
        None
    """
    surrogate_results_dir = (f"{benchmarking_metrics['results_dir']}" +
                             f"/surrogate_model_fitting")
    torch.save(model.state_dict(), 
               f'{surrogate_results_dir}/gp_model_state_{iteration}.pth')
    
def store_offspring(offspring_graphs: List[nx.Graph], 
                    benchmarking_metrics: dict, 
                    iteration: int) -> None:
    """ 
    Store the offspring graphs.
    
    Args:
        offspring_graphs: List of offspring graphs.
        benchmarking_metrics: Dictionary with benchmarking metrics.
        iteration: The iteration number of the optimisation process.
    
    Returns:
        None
    """
    
    offspring_smiles = [graph_to_smiles(graph) for graph in offspring_graphs]
    filepath = f"{benchmarking_metrics['results_dir']}/surrogate_model_fitting"
    with open(f"{filepath}/graphs_{iteration}.pickle", 'wb') as f:
        pickle.dump(offspring_smiles, f)