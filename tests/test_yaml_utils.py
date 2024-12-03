import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Standard library imports
import unittest
import shutil

# Local imports
from modules.utils.yaml_utils import *


class TestYamlUtils(unittest.TestCase):
    """Tests for the YAML utility functions."""
    def setUp(self):
        """Set up the configuration dictionary for the tests."""
        self.config = {
            'surrogate_model': {'model': 'graph_gp'},
            'acquisition_function': {'expected_improvement': True},
            'acquisition_optimiser': {'genetic_algorithm': True},
            'graph_kernel': {'kernel_name': 'random_walk'},
            'initial_dataset': {'sample_from_zinc': True, 
                                'sample_from_transformer': False, 
                                'sample_from_custom_dataset': False},
            'bayesian_optimisation': {'n_initial_samples': 10, 'n_trials': 20}
        }

    def test_less_trials_than_initial_samples(self):
        """Test when there are less trials than initial samples."""
        experiment_config_filepath = 'tests/test_experiment_yaml/' + \
                                    'test_less_trials_than_initial_samples.yaml'
        config = load_experiment(experiment_config_filepath)   #YAML file

        # Test the condition that should raise a ValueError
        with self.assertRaises(ValueError) as context:
            load_initial_dataset(config)

        # Check the specific error message
        self.assertTrue(
            "The number of initial samples must be less than the number of trials." 
                        in str(context.exception), 
                        "Expected a specific ValueError message when initial \
                        samples exceed the number of trials.")

    def test_create_results_dir(self):
        """Test the creation of a results directory."""
        main_dir = 'test_results'
        oracle_name = 'test_oracle'
        max_calls = 10
        
        # Create a results directory
        results_dir = create_results_dir(main_dir, oracle_name, max_calls)
        # Check if the directory exists
        assert os.path.exists(results_dir), "The results directory was not created."
        # Cleanup after test
        shutil.rmtree(main_dir)  # This removes the directory and its contents

    def test_load_surrogate_model(self):
        """Test the loading of the surrogate model."""
        surrogate_model = load_surrogate_model(self.config)
        self.assertEqual(surrogate_model.__name__, 'GraphGP', 
                         "The surrogate model did not load correctly.")

    def test_load_graph_kernel(self):
        """Test the loading of the graph kernel."""
        graph_kernel = load_graph_kernel(self.config)
        self.assertIsInstance(graph_kernel, RandomWalkKernel, 
                              "The graph kernel did not load correctly.")

    def test_copy_yaml_to_results_dir(self):
        """Test the copying of a YAML file to a results directory."""
        os.makedirs('test_results_dir', exist_ok=True)
        with open('test_config.yaml', 'w') as file:
            file.write("test: config")


if __name__ == '__main__':
    unittest.main()
