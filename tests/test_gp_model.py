import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from modules.surrogate_models.gp_model import GraphGP, initialize_model
from modules.utils.read_sample_store import sample_graphs_from_smiles_csv
from modules.surrogate_models.kernel import WeisfeilerLehmanKernel
from tdc import Oracle
from gpytorch.likelihoods import GaussianLikelihood
import unittest
import torch
from gauche import NonTensorialInputs
import gpytorch

class TestGraphGP(unittest.TestCase):
    """
    A test suite for verifying the functionality and correctness of the GraphGP model,
    which integrates Gaussian Process modeling with graph-based input data. The tests
    cover model training, evaluation, and initialization scenarios ensuring the model's
    output consistency and correctness.
    """

    def setUp(self):
        """
        Set up the test environment for GraphGP tests by initializing training and testing datasets
        from a given SMILES CSV file. It also initializes the objective function, model,
        likelihood, and kernel used in all tests.
        """
        initial_dataset_graphs, initial_dataset_smiles = sample_graphs_from_smiles_csv(
                    filepath='data/zinc_1000_sample.csv', 
                    sample_size=10)

        test_dataset_graph, test_dataset_smiles = sample_graphs_from_smiles_csv(
                    filepath='data/zinc_1000_sample.csv', 
                    sample_size=5)
            
        objective_function = Oracle(name='albuterol_similarity')

        self.train_x = NonTensorialInputs(initial_dataset_graphs)
        self.train_y = torch.tensor(objective_function(initial_dataset_smiles)).flatten().float()
        self.test_x = NonTensorialInputs(test_dataset_graph)

        self.likelihood = GaussianLikelihood()
        self.kernel = WeisfeilerLehmanKernel(node_label='element', edge_label='bond')

        self.model = GraphGP(self.train_x, self.train_y, self.likelihood, self.kernel)

    def test_train_call(self):
        """
        Test the training procedure of the GraphGP model. Verifies that the mean and variance of the output
        distribution are consistent across all test samples within a specified tolerance and that their lengths
        match the number of test inputs.
        """
        self.model.train()
        self.model.likelihood.train()
        output_distribution = self.model(self.test_x)

        mean_reference = output_distribution.mean[0]
        mean_all_same = torch.allclose(output_distribution.mean, 
            mean_reference.expand_as(output_distribution.mean), atol=1e-4)
        self.assertTrue(mean_all_same, "Not all elements in the tensor are the same.")
        self.assertEqual(len(output_distribution.mean), 
                        len(self.test_x), 'length of mean is not accurate')

        variance_reference = output_distribution.variance[0]
        variance_all_same = torch.allclose(output_distribution.variance, 
            variance_reference.expand_as(output_distribution.variance), atol=1e-4)
        self.assertTrue(variance_all_same, "Not all elements in the tensor are the same.")
        self.assertEqual(len(output_distribution.variance), 
                        len(self.test_x), 'length of variance is not accurate')
        
    def test_evaluation_call(self):
        """
        Test the evaluation mode of the GraphGP model to ensure that the model's outputs (mean and variance)
        are not uniform and their lengths match the test input size, reflecting correct behavior under model evaluation conditions.
        """
        self.model.eval()
        self.model.likelihood.eval()
        output_distribution = self.model(self.test_x)

        mean_reference = output_distribution.mean[0]
        mean_all_same = torch.allclose(output_distribution.mean, 
                                       mean_reference.expand_as(output_distribution.mean), atol=1e-4)
        self.assertFalse(mean_all_same, "All elements in the mean tensor are the same.")
        self.assertEqual(len(output_distribution.mean), 
                        len(self.test_x), 'length of mean is not accurate')

        variance_reference = output_distribution.variance[0]
        variance_all_same = torch.allclose(output_distribution.variance, 
                                       variance_reference.expand_as(output_distribution.variance), atol=1e-4)
        self.assertFalse(variance_all_same, "All elements in the variance tensor are the same.")
        self.assertEqual(len(output_distribution.variance), 
                         len(self.test_x), 'length of variance is not accurate')
        
    def test_initialise_model(self):
        """
        Test the initialization of the GraphGP model using helper functions, ensuring that the created
        model and its marginal log likelihood (MLL) are of the correct types, crucial for subsequent training and inference.
        """
        mll, model = initialize_model(self.train_x, self.train_y, GraphGP, GaussianLikelihood(), WeisfeilerLehmanKernel)
        self.assertIsInstance(mll, gpytorch.mlls.VariationalELBO, 'mll type is not correct')
        self.assertIsInstance(model, GraphGP, 'model type is not correct')

if __name__ == '__main__':
    unittest.main()
