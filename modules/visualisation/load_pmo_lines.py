from typing import List

import pandas as pd 
import numpy as np
import json

def load_pmo_lines(file_path: str,
                        storage_path: str,
                        storage_file_name: str,
                        start_iteration: int = 1,
                        start_value: float = 0,
                        save_results: bool = True) -> List[float]:
    """ 
    Loads CSV file with the data for a single line of the PMO data.
    
    Args:
        file_path: Path to the CSV file.
        results_array: Array to store the results.
        storage_path: Path to store the results.
        
    Returns:
        results_array: Array with the results.
    """
    # Initialize the raw data array as an empty list of tuples
    raw_data = []
    
    # Load the data as csv
    with open(file_path, 'r') as file:
        for line in file:
            single_entry = line.strip().split(',')
            raw_data.append((int(float(single_entry[0])), float(single_entry[1])))
            
    # Sort raw data by iteration
    raw_data.sort(key=lambda x: x[0])
    
    cleaned_array = []
    
    relevant_oracle_score = start_value
        
    for iteration in range(1, start_iteration, 1):
        cleaned_array.append(np.nan)
    for iteration in range(start_iteration, 10001, 1):
        if raw_data == []:
            cleaned_array.append(relevant_oracle_score)
            continue
        elif raw_data[0][0] > iteration:
            cleaned_array.append(relevant_oracle_score)
        else:
            relevant_oracle_score = round((raw_data[0][1]), 2)
            raw_data.pop(0)
            
            if relevant_oracle_score > 0.995:
                relevant_oracle_score = 1.0
            elif relevant_oracle_score < 0.0:
                relevant_oracle_score = 0.0
                
            cleaned_array.append(relevant_oracle_score)
    
    # Save the results
    if save_results:
        storage_file = storage_path + storage_file_name
        with open(storage_file, 'w') as f:
            json.dump(cleaned_array, f)
    
    return cleaned_array
    