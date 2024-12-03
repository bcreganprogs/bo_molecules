import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from modules.acquisition_functions.acquisition_functions import (GraphExpectedImprovement, GraphProbabilityOfImprovement, 
                                                                 GraphUpperConfidenceBound, RandomSampler,
                                                                 EntropySearch)
from modules.surrogate_models.gp_model import GraphGP
from gpytorch.distributions import MultivariateNormal
import torch
import unittest

class TestAcquisitionFunctions(unittest.TestCase):
    """
    Class to test aquisition functions
    """

    def test_expected_improvement(self):
        """Test that the graph expected improvement returns accurate value."""

        distribution = MultivariateNormal(torch.tensor([0.0, 0.2]), torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
        acf = GraphExpectedImprovement(model=GraphGP, best_f=0.5)
        output = acf.calculate_expected_improvement(distribution.mean, torch.sqrt(distribution.variance), 0.01)
        self.assertTrue(torch.allclose(output, torch.tensor([0.1978, 0.2667]), atol = 1e-3), "Graph Expected Improvement test failed.")

    def test_probability_of_improvement(self):
        """Test that the graph probability of improvement returns accurate value"""

        distribution = MultivariateNormal(torch.tensor([0.0, 0.2]), torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
        acf = GraphProbabilityOfImprovement(model=GraphGP, best_f=0.5)
        output = acf.calculate_prob_of_improvement(distribution.mean, torch.sqrt(distribution.variance), 0.01)
        self.assertTrue(torch.allclose(output, torch.tensor([0.3050, 0.3783]), atol=1e-3), "Graph Probability of Improvement test failed.")

    def test_upper_confidence_bound(self):
        """Test that the graph upper confidence bound returns accurate value"""

        distribution = MultivariateNormal(torch.tensor([0.0, 0.2]), torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
        acf = GraphUpperConfidenceBound(model=GraphGP, best_f=0.5)
        output = acf.calculate_upper_confidence_bound(distribution.mean, torch.sqrt(distribution.variance), epsilon=0.15)
        self.assertTrue(torch.allclose(output, torch.tensor([1.74927435757, 1.94927435757]), atol=1e-3), "Graph Upper confidence bound test failed.")

    def test_random(self):
        """Test that the graph upper confidence bound returns accurate value"""

        acf = RandomSampler(model=GraphGP, best_f=0.5)
        output = acf.calculate_random_values(7)
        self.assertEqual(len(output), 7, "The tensor should have a length of 7")
        self.assertTrue(torch.all((output>= 0) & (output <= 1)), "All tensor values should be between 0 and 1")

    def test_entropy_calculation(self):
        """Test that the entropy of a model in Entropy Search Acquisition Function
           is being calculated correctly"""

        acf = EntropySearch(model=GraphGP, best_f=0.5, iteration_count=1)
        covariance_matrix = torch.Tensor([[1.2, 0.8, 0.6, 0.4],
                                        [0.8, 1.5, 0.9, 0.7],
                                        [0.6, 0.9, 1.3, 0.5],
                                        [0.4, 0.7, 0.5, 1.1]])
        output = acf.calculate_entropy(covariance_matrix)
        self.assertTrue(torch.allclose(output, torch.Tensor([5.4662]), atol=1e-3), "The entropy calculation is incorrect.")

if __name__ == "__main__":
    unittest.main()