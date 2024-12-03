import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
from unittest.mock import patch
from modules.acquisition_function_optimisers.transformer import Transformer


class TestTransformer(unittest.TestCase):

    def test_sample_output_type(self):
        # Test to check if the output is a list
        transformer = Transformer(n_offspring=3)
        offspring = transformer.sample()
        self.assertIsInstance(offspring, list, "Output should be a list")

    def test_sample_output_length(self):
        # Test to check if the output list length matches n_offspring
        n_offspring = 5
        transformer = Transformer(n_offspring=n_offspring)
        offspring = transformer.sample()
        self.assertEqual(len(offspring), n_offspring, "Output length should match n_offspring")

    def test_sample_empty_case(self):
        # Test to check behavior when n_offspring is 0
        transformer = Transformer(n_offspring=0)
        offspring = transformer.sample()
        self.assertEqual(offspring, [], "Output should be an empty list when n_offspring is 0")


if __name__ == '__main__':
    unittest.main()
