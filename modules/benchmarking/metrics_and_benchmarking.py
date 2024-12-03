# Standard library imports
import heapq
import os
import json
from typing import List

# Third-pary imports
import numpy as np
import matplotlib.pyplot as plt


def accumulate_top_n(array: np.array, n: int) -> List[float]:
    """
    Accumulate the top n values up to index i for each i in the array.
    
    Args:
        array: array of observed values.
        n: number of top values to track.
    
    Returns:
        top_n_accumulated: A list of floats, where each entry is the mean of
                           the top n values up to that index.
    """
    # Initialize an empty min heap to keep track of the top n values
    heap = []
    # Initialize the list that will hold the top n values at each index
    top_n_accumulated = []
    
    for value in array:
        # If the heap size exceeds n, remove the smallest element
        if len(heap) < n:
            heapq.heappush(heap, value)
        else:
            # Add value to the heap and then remove the smallest element
            heapq.heappushpop(heap, value)
        
        top_n_accumulated.append(np.mean(heap))
    
    return top_n_accumulated

def top_auc(buffer: dict,
            top_n: int,
            max_oracle_calls: int,
            finish = True, 
            freq_log = 1) -> float:
    """
    Generates AUC metric for top n values.
    
    This implementation was adapted from the PMO benchmarking Github 
    (https://github.com/wenhao-gao/mol_opt) for the sake of comparability.

    Args:
        buffer: dictionary of observed values where the key is the molecule
                 graph structure and the value is a tuple of the observed 
                 oracle score value and the iteration number.
        top_n: number of top values to consider for the AUC metric.
        max_oracle_calls: maximum number of oracle calls allowed in the
                          optimization process
        finish: whether or not the optimization process has finished;
                 defaults to True.
        freq_log: frequency of logging for the oracle calls, which is used
                  to calculate the AUC metric. Defaults to 1.
        
    Returns:
        auc: AUC metric for the top n values. Will be a value between 0 and 1.
    """
    sum = 0
    prev = 0
    called = 0
    
    # Sort the buffer by the iteration number in ascending order
    ordered_results = list(sorted(
        buffer.items(), key=lambda kv: kv[1][1], reverse=False))
    
    # Start by only considering a batch sized freq_log
    # and then adding new values of size freq_log until the end of the buffer
    for idx in range(freq_log, min(len(buffer), max_oracle_calls), freq_log):
        temp_result = ordered_results[:idx]
        
        # Consider the top n values in the current batch
        temp_result = list(sorted(
            temp_result, key=lambda kv: kv[1][0], reverse=True))[:top_n]
        
        # Mean of the top n values in the current batch
        top_n_now = np.mean([item[1][0] for item in temp_result])
        
        # Add trapzoidal area under the curve for the current batch
        sum += freq_log * (top_n_now + prev) / 2
        
        # Need to keep track of the previous top n mean value
        prev = top_n_now
        
        called = idx
        
    # If the for loop iteration has finished, consider the remaining values
    # for example if the buffer size is 100 and freq_log is 30, then the
    # previous loop would have considered the first 90 values and this
    # loop would consider the last 10 values
    temp_result = list(sorted(
        ordered_results, key=lambda kv: kv[1][0], reverse=True))[:top_n]
    
    top_n_now = np.mean([item[1][0] for item in temp_result])
    sum += (len(buffer) - called) * (top_n_now + prev) / 2
    
    # If the optimization process is shorter than the maximum number of oracle calls
    # then we need to consider the remaining area under the curve
    # for comparison purposes
    if finish and len(buffer) < max_oracle_calls:
        sum += (max_oracle_calls - len(buffer)) * top_n_now
    
    # Normalize the sum by the maximum possible sum. 
    # Hence, the AUC will be a value between 0 and 1.
    auc = sum / max_oracle_calls
    return auc

def mean_of_top_n(array: List[float], n: int) -> float:
    """
    Calculate the mean of the top n values in the given array.
    
    Parameters:
    - array: List array of observed values.
    - n: The number of top values to consider for calculating the mean.
    
    Returns:
    - The mean of the top n values.
    """
    if n > len(array):
        raise ValueError("n cannot be greater than size of the array.")
    # Sort the array in descending order and take the first n elements
    top_n = np.sort(array)[-n:]
    # Compute the mean of these top n values
    mean_top_n = np.mean(top_n)
    
    return mean_top_n
                                                         
def plot_top_n_curve(best_observed: List[float], 
                     oracle_name: str, 
                     results_dir: str,
                     n: int = 1,
                     interim: bool = False) -> None:
    """
    Generate a plot of the mean of the top n observed values over iterations.

    Args:
        best_observed: List of observed values from all oracle 
                       calls of the experiment.
        oracle_name: name of the oracle function
        results_dir: directory to save the plot
        n: Top n values to consider. Defaults to 1.
        interim: whether the plot is interim or final.
    
    Returns:
        None
    """
    array = accumulate_top_n(best_observed, n)

    plt.plot(array)
    plt.xlabel('Iteration')
    plt.ylabel(f'Mean of Top {n} Observed')
    if interim:
        plt.title(f'INTERIM: Best Observed {oracle_name} Value Over Iterations')
    else:
        plt.title(f'Best Observed {oracle_name} Value Over Iterations')
    file_path_suffix = f'Best_observed_top_{n}.png'
    plot_path = os.path.join(results_dir, file_path_suffix)
    print(plot_path)
    plt.savefig(plot_path)
    plt.close()

def plot_oracle_history(best_observed: List[float], 
                     oracle_name: str, 
                     results_dir: str,
                     interim: bool,
                     window_size = 20) -> None:
    """
    Generate a plot of the mean of the top n observed values over iterations.

    Args:
        best_observed: List of observed values from all oracle 
                       calls of the experiment.
        oracle_name: name of the oracle function
        results_dir: directory to save the plot
        n: Top n values to consider. Defaults to 1.
        window_size: size of the window for the moving average. Defaults to 20.
    
    Returns:
        None
    """
    
    def moving_average(arr: np.array,
                       window_size: int) -> np.array:
        """
        Calculate the moving average of an array.

        Args:
            arr: array of observed values.
            window_size: size of the window for the moving average.
        
        Returns:
            np.array: array of the moving average values.
        """
        cumsum = np.cumsum(np.insert(arr, 0, 0)) 
        diff = cumsum[window_size:] - cumsum[:-window_size]
        return (diff) / float(window_size)
    
    mov_avg = moving_average(best_observed, window_size)

    plt.plot(best_observed)
    plt.plot(np.arange(window_size - 1, len(best_observed)), mov_avg, 
             label=f'Moving Average, Window Size {window_size}', color='red')
    
    plt.xlabel('Iteration')
    plt.ylabel(f'{oracle_name}')
    if interim:
        plt.title(f'INTERIM: Observed {oracle_name} Over Iterations')
    else:
        plt.title(f'Observed {oracle_name} Over Iterations')
    file_path_suffix = 'Oracle_history.png'
    plot_path = os.path.join(results_dir, file_path_suffix)
    print(plot_path)
    plt.legend()
    plt.savefig(plot_path)
    plt.close()

def print_benchmarking_metrics(benchmarking_metrics: dict,
                                        best_observed: List[float],
                                        buffer: dict,
                                        current_iteration: int,
                                        save_as_json: bool) -> None:
    """
    Printing benchmarking metrics for optimization process.

    Args:
        benchmarking_metrics: benchmarking metrics from config file.
        best_observed: List of all observed oracle call values.
        buffer: dictionary of observed values where the key is the 
                molecule graph structure and the value is a tuple
                of the observed oracle score value and the iteration 
                number.
        current_iteration: current iteration number (oracle calls).
        save_as_json: whether to save the benchmarking results as a json file.
    
    Returns:
        None
    """
    print("---------------------------")
    print("START: Benchmarking Metrics")
    print("---------------------------")
    
    benchmarking_results = dict()
    
    if benchmarking_metrics['top_1_benchmark']:
        
        results_top_1 = mean_of_top_n(best_observed, 1)
        print(f'The final top 1 is {results_top_1}')
        benchmarking_results['results_top_1'] = results_top_1
        
        results_auc_top_1 = top_auc(buffer, top_n=1, finish=True, freq_log=1, 
                        max_oracle_calls=current_iteration)
        print(f'The top 1 AUC is {results_auc_top_1}')
        benchmarking_results['results_auc_top_1'] = results_auc_top_1
    
    if benchmarking_metrics['top_10_benchmark']:
        if len(best_observed) < 10:
            print("Not enough oracle calls to calculate top 10 AUC.")
        else:
            results_top_10 = mean_of_top_n(best_observed, 10)
            print(f'The final top 10 is {results_top_10}')
            benchmarking_results['results_top_10'] = results_top_10
            
            results_auc_top_10 = top_auc(buffer, top_n=10, finish=True,
                                         freq_log=1, 
                                         max_oracle_calls=current_iteration)
            print(f'The top 10 AUC is {results_auc_top_10}')
            benchmarking_results['results_auc_top_10'] = results_auc_top_10
        
    if benchmarking_metrics['top_100_benchmark']:
        if len(best_observed) < 100:
            print("Not enough oracle calls to calculate top 100 AUC.")
        else:
            results_top_100 = mean_of_top_n(best_observed, 100)
            print(f'The final top 100 is {results_top_100}')
            benchmarking_results['results_top_100'] = results_top_100
            
            results_auc_top_100 = top_auc(buffer, top_n=100, finish=True,
                                          freq_log=1, 
                                          max_oracle_calls=current_iteration)
            print(f'The top 100 AUC is {results_auc_top_10}')
            benchmarking_results['results_auc_top_100'] = results_auc_top_100
    
    if save_as_json:
        with open(os.path.join( benchmarking_metrics['results_dir'], 
            'benchmarking_results.json'), 'w') as f:
            
            json.dump(benchmarking_results, f)
        
        
    print("-------------------------")
    print("END: Benchmarking Metrics")
    print("-------------------------")
                              
def generate_benchmarking_plots(benchmarking_metrics: dict,
                                best_observed: List[float],
                                interim: bool) -> None:
    """
    Saving final benchmarking plots for optimization process.

    Args:
        benchmarking_metrics: benchmarking metrics from config file.
        best_observed: List of all observed oracle call values.
        interim: whether the plots are interim or final.
    
    Returns:
        None
    """
    if benchmarking_metrics['top_1_benchmark']:
        plot_top_n_curve(best_observed, 
                        benchmarking_metrics['oracle_name'],
                        benchmarking_metrics['results_dir'], 
                        n=1,
                        interim=interim)
    if benchmarking_metrics['top_10_benchmark']:
        plot_top_n_curve(best_observed, 
                        benchmarking_metrics['oracle_name'],
                        benchmarking_metrics['results_dir'], 
                        n=10,
                        interim=interim)
    if benchmarking_metrics['top_100_benchmark']:
        plot_top_n_curve(best_observed, 
                        benchmarking_metrics['oracle_name'],
                        benchmarking_metrics['results_dir'], 
                        n=100,
                        interim=interim)
    if benchmarking_metrics['plot_oracle_history']:
        plot_oracle_history(best_observed, 
                            benchmarking_metrics['oracle_name'],
                            benchmarking_metrics['results_dir'], 
                            interim=interim)