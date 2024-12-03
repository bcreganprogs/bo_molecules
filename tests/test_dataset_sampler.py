import unittest
import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
import networkx as nx
from modules.acquisition_function_optimisers.dataset_sampler import (
    DatasetSampler)


class TestDatasetSampler(unittest.TestCase):
    def setUp(self) -> None:
        self.sampler = DatasetSampler(n_offspring=3, filepath='data/zinc.csv')    

    def test_sample_output_type(self):
        # Test to check if the output is a list of nx.Graph
        graphs = self.sampler.sample()
        self.assertIsInstance(graphs, list, "Output should be a list")
        for graph in graphs:
            self.assertIsInstance(graph, nx.Graph, 
                                  "Each item in list should be an nx.Graph")

    def test_sample_output_length(self):
        # Test to check if the output list length matches n_offspring
        n_offspring = 5
        sampler = DatasetSampler(n_offspring=n_offspring, 
                                 filepath='data/zinc.csv')
        graphs = sampler.sample()
        self.assertEqual(len(graphs), n_offspring, 
                         "Output length should match n_offspring")

    def test_sample_empty_case(self):
        # Test to check behavior when n_offspring is 0
        sampler = DatasetSampler(n_offspring=0, filepath='data/zinc.csv')
        graphs = sampler.sample()
        self.assertEqual(graphs, [], 
                         "Output should be an empty list when n_offspring is 0")

if __name__ == '__main__':
    unittest.main()
