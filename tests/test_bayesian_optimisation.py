import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from modules.bayesian_optimisation.bayesian_optimisation import evaluate_acquisition_function, bayesian_optimisation
import torch
import networkx as nx
import unittest
from modules.bayesian_optimisation.bayesian_optimisation import bayesian_optimisation
from modules.utils.read_sample_store import sample_graphs_from_smiles_csv
from modules.surrogate_models.kernel import WeisfeilerLehmanKernel
from modules.surrogate_models.gp_model import GraphGP
from modules.acquisition_functions.acquisition_functions import GraphExpectedImprovement
from modules.acquisition_function_optimisers.genetic_algorithm import GeneticAlgorithm
from tdc import Oracle

class TestEvaluateAcquisitionFunction(unittest.TestCase):
    """
    A unit test class for testing the evaluate_acquisition_function method. This class
    specifically tests the functionality and integration with a custom Genetic Algorithm class,
    ensuring that the acquisition function is evaluated correctly across given fitness function outputs.
    """

    def test_with_genetic_algorithm(self):
        
        """
        Tests the evaluate_acquisition_function's ability to handle and process outputs generated
        by a Genetic Algorithm (GA). This test ensures that the acquisition function correctly processes
        inputs from the GA and that the output is as expected based on a predefined fitness function.
        
        Inner components:
            GeneticAlgorithm: A mock-up class mimicking a genetic algorithm, specifically tailored to
                              produce predictable outputs for testing the evaluate_acquisition_function.
            
            test_acq_func: A mock-up acquisition function designed to convert input features into
                           a tensor of indices representing each feature for simplicity in testing.
        
        The test:
        1. Initializes the mock GeneticAlgorithm and defines a simple acquisition function.
        2. Applies a fitness function, runs the acquisition function evaluation, and checks the output.
        3. Asserts that the outputs are instances of nx.Graph (expected data type) and checks for equality
           against a predefined expected result.
        """
        dataset, _ = sample_graphs_from_smiles_csv(
            filepath='data/zinc_1000_sample.csv', 
            sample_size=10)

        class GeneticAlgorithm:
            "Custom genetical algorithm class for testing"

            def __init__(self):
                pass

            def sample(self):
                return [dataset[1], dataset[2], dataset[3]]

        def test_acq_func(x):
            "testing acquisition function"
            return torch.tensor([i for i in range(len(x))])

        test_fitness_function = [(dataset[4],1),(dataset[5],2),(dataset[6],3)]
        test_genetic_algorithm  = GeneticAlgorithm()
        ga_output = evaluate_acquisition_function(test_acq_func, test_genetic_algorithm, test_fitness_function)
        expected = dataset[3]

        self.assertIsInstance(ga_output[0], nx.classes.graph.Graph)
        self.assertEqual(ga_output[0], expected)

class TestBayesianOptimisaton(unittest.TestCase):

    """
    This class tests the Bayesian optimization process on a dataset of molecularstructures.
    It verifies the Bayesian optimization algorithm's ability to maintain expected data types and sizes throughout the process,
    ensuring that all aspects of the optimization output correctly reflect the configured parameters and constraints.
    
    Methods:
        test_bayesian_optimisation: Tests various aspects of the Bayesian optimization output, including the length
                                    and type of results lists, and checks constraints on numerical outputs.
    """

        
    def test_bayesian_optimisation(self):

        """
        Tests the Bayesian optimization pipeline for consistency and correctness of the output. It ensures that the pipeline:
        
        - Produces an expected number of results.
        - Maintains output values within the specified range (0 to 1 for objective function scores).
        - Keeps the output data types consistent (graphs are networkx.Graph objects, SMILES strings are strings).
        
        The test simulates the Bayesian optimization process using a genetic algorithm as the acquisition function optimizer and
        a simple molecular similarity function as the objective. The test checks:
        
        - The lengths of the output lists (objective_function_scores)
        - The value range of the objective function scores.
        """

        initial_dataset, _ = sample_graphs_from_smiles_csv(
            filepath='data/zinc_1000_sample.csv', 
            sample_size=10)
    
        objective_function = Oracle(name='albuterol_similarity')
        acquisition_function_optimiser = GeneticAlgorithm (
            n_parents=15,
            n_offspring=30, 
            fitness_function=None,
            max_atoms=60,
            mutation_rate = 0.2,
            sampling='uniform')
        
        benchmarking_metrics = {
        "interim_benchmarking": 'false',
        "interim_benchmarking_freq": 100,
        "top_1_benchmark": 'false',
        "top_10_benchmark": 'false',
        "top_100_benchmark": 'false',
        "plot_oracle_history": 'false',
        "results_dir": None,
        "oracle_name": 'false',
        "max_oracle_calls": 'false'
    }

        obj_fun_scores, buffer = bayesian_optimisation(
        initial_dataset,
        15,
        WeisfeilerLehmanKernel(node_label='element', edge_label='bond'),
        GraphExpectedImprovement,   # should set type constraint here
        acquisition_function_optimiser,
        GraphGP,
        objective_function,
        benchmarking_metrics)

        self.assertEqual(len(obj_fun_scores), 15, "The length of the objective function scores should be 15")
        self.assertTrue(torch.all((torch.Tensor(obj_fun_scores)>= 0) & (torch.Tensor(obj_fun_scores)<= 1)), 
                        "All tensor values should be between 0 and 1")

if __name__ == '__main__':
    unittest.main()
