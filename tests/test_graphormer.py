import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
import torch
import networkx as nx
from typing import List
from modules.utils.molecular_data_conversion import smiles_to_graph
from modules.DGK_Graphormer.DGK_model_utils import GraphormerGraphEncoder
from modules.DGK_Graphormer.Graphormer_DGK import *

# Assuming the load_new_model and other functions are imported from the module as per your structure

class TestGraphProcessing(unittest.TestCase):
    def test_load_new_model_structure(self):
        """ Test if the GraphormerGraphEncoder is loaded with correct hyperparameters """
        model = load_new_model()
        self.assertIsInstance(model, GraphormerGraphEncoder, "Model should be an instance of GraphormerGraphEncoder.")
        # Check for a specific layer or parameter to confirm structure
        self.assertTrue(hasattr(model, 'forward'), "Model should have a forward method.")

    def test_get_ogbg_from_data(self):
        """ Test conversion from networkx graphs to ogbg format. """
        # Create a simple networkx graph
        G = smiles_to_graph('CCO')
        
    
        data = NonTensorialInputs(G)
        # Convert this to a mock smiles2graph function
        ogbg_data = get_ogbg_from_data(data)
        self.assertIsInstance(ogbg_data, List, "Output should be a list.")
        # print(ogbg_data)
        self.assertEqual(len(ogbg_data), 1, "There should be one graph processed.")

    def test_get_embeddings_output_type(self):
        """ Test embeddings output type from a dummy Graphormer model and a dummy graph. """
        class DummyModel:
            def eval(self):
                pass
            
            def __call__(self, **kwargs):
                return (None, torch.randn(1, 128))  # Mocked output

        G = smiles_to_graph('CCO')
        data = NonTensorialInputs(G)
        model = DummyModel()
        embeddings = get_embedings(model, data)
        self.assertIsInstance(embeddings, torch.Tensor, "Embeddings should be a torch tensor.")
        self.assertEqual(embeddings.shape[1], 128, "Embeddings should have the correct shape.")

    def test_get_covariance_matrix_output_type(self):
        """ Test covariance matrix calculation. """
        class DummyModel:
            def eval(self):
                pass
            
            def __call__(self, **kwargs):
                return (None, torch.randn(1, 128))  # Mocked output

        G = smiles_to_graph('CCO')
        data = NonTensorialInputs(G)
        model = DummyModel()
        cov_matrix = get_covariance_matrix(model, data)
        self.assertIsInstance(cov_matrix, torch.Tensor, "Covariance matrix should be a torch tensor.")
        self.assertEqual(cov_matrix.shape[0], cov_matrix.shape[1], "Covariance matrix should be square.")


if __name__ == '__main__':
    unittest.main()
