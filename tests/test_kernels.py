import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from modules.surrogate_models.kernel import ( 
    VertexHistogramKernel, EdgeHistogramKernel,
    WeisfeilerLehmanKernel, NeighborhoodHashKernel,
    RandomWalkKernel, ShortestPathKernel,
    WeisfeilerLehmanOptimalAssignmentKernel)
from modules.utils.read_sample_store import sample_graphs_from_smiles_csv
import unittest
import torch

initial_dataset, _ = sample_graphs_from_smiles_csv(
            filepath='data/zinc_1000_sample.csv', 
            sample_size=10)
n = len(initial_dataset)

class TestKernels(unittest.TestCase):
    """Test all graph kernels working as expected."""

    def test_VertexHistogramKernel(self):
        """
        Tests the VertexHistogramKernel.
        Confirms:
        - Correct matrix dimensions.
        - Diagonal elements approximately equal to 1.
        - All elements are bounded between 0 and 1.
        """

        graph_kernel = VertexHistogramKernel(node_label='element')
        covariance_matrix = graph_kernel(initial_dataset)
        expected_shape = (n,n)
        self.assertEqual(covariance_matrix.shape, 
                         expected_shape, f"Expected tensor shape {expected_shape}, but got {covariance_matrix.shape}")
        
        diagonal = covariance_matrix.diag()
        expected_diagonal = torch.ones(n)
        diagonal_condition = torch.allclose(diagonal, expected_diagonal, atol=1e-3)
        self.assertTrue(diagonal_condition, "Graph Upper confidence bound test failed.")
        
        numeric_condition = torch.all((covariance_matrix >= 0) & (covariance_matrix <= 1))
        self.assertTrue(numeric_condition, "Not all elements are between 0 and 1")

    def test_EdgeHistogramKernel(self):
        """
        Tests the EdgeHistogramKernel.
        Confirms:
        - Correct matrix dimensions.
        - Diagonal elements approximately equal to 1.
        - All elements are bounded between 0 and 1.
        """

        graph_kernel = EdgeHistogramKernel(edge_label='bond')
        covariance_matrix = graph_kernel(initial_dataset)
        expected_shape = (n,n)
        self.assertEqual(covariance_matrix.shape, 
                         expected_shape, f"Expected tensor shape {expected_shape}, but got {covariance_matrix.shape}")
        
        diagonal = covariance_matrix.diag()
        expected_diagonal = torch.ones(n)
        diagonal_condition = torch.allclose(diagonal, expected_diagonal, atol=1e-3)
        self.assertTrue(diagonal_condition, "Graph Upper confidence bound test failed.")
        
        numeric_condition = torch.all((covariance_matrix >= 0) & (covariance_matrix <= 1))
        self.assertTrue(numeric_condition, "Not all elements are between 0 and 1")

    def test_WeisfeilerLehmanKernel(self):
        """
        Tests the WeisfeilerLehmanKernel.
        Confirms:
        - Correct matrix dimensions.
        - Diagonal elements approximately equal to 1.
        - All elements are bounded between 0 and 1.
        """

        graph_kernel = WeisfeilerLehmanKernel(node_label='element', 
                                              edge_label='bond')
        covariance_matrix = graph_kernel(initial_dataset)
        expected_shape = (n,n)
        self.assertEqual(covariance_matrix.shape, 
                         expected_shape, f"Expected tensor shape {expected_shape}, but got {covariance_matrix.shape}")
        
        diagonal = covariance_matrix.diag()
        expected_diagonal = torch.ones(n)
        diagonal_condition = torch.allclose(diagonal, expected_diagonal, atol=1e-3)
        self.assertTrue(diagonal_condition, "Graph Upper confidence bound test failed.")
        
        numeric_condition = torch.all((covariance_matrix >= 0) & (covariance_matrix <= 1))
        self.assertTrue(numeric_condition, "Not all elements are between 0 and 1")

    def test_NeighborhoodHashKernel(self):
        """
        Tests the NeighborhoodHashKernel.
        Confirms:
        - Correct matrix dimensions.
        - Diagonal elements approximately equal to 1.
        - All elements are bounded between 0 and 1.
        """

        graph_kernel = NeighborhoodHashKernel(node_label='element')
        covariance_matrix = graph_kernel(initial_dataset)
        expected_shape = (n,n)
        self.assertEqual(covariance_matrix.shape, 
                         expected_shape, f"Expected tensor shape {expected_shape}, but got {covariance_matrix.shape}")
        
        diagonal = covariance_matrix.diag()
        expected_diagonal = torch.ones(n)
        diagonal_condition = torch.allclose(diagonal, expected_diagonal, atol=1e-3)
        self.assertTrue(diagonal_condition, "Graph Upper confidence bound test failed.")
        
        numeric_condition = torch.all((covariance_matrix >= 0) & (covariance_matrix <= 1))
        self.assertTrue(numeric_condition, "Not all elements are between 0 and 1")

    def test_RandomWalkKernel(self):
        """
        Tests the RandomWalkKernel.
        Confirms:
        - Correct matrix dimensions.
        - Diagonal elements approximately equal to 1.
        - All elements are bounded between 0 and 1.
        """

        graph_kernel = RandomWalkKernel()
        covariance_matrix = graph_kernel(initial_dataset)
        expected_shape = (n,n)
        self.assertEqual(covariance_matrix.shape, 
                         expected_shape, f"Expected tensor shape {expected_shape}, but got {covariance_matrix.shape}")
        
        diagonal = covariance_matrix.diag()
        expected_diagonal = torch.ones(n)
        diagonal_condition = torch.allclose(diagonal, expected_diagonal, atol=1e-3)
        self.assertTrue(diagonal_condition, "Graph Upper confidence bound test failed.")
        
        numeric_condition = torch.all((covariance_matrix >= 0) & (covariance_matrix <= 1))
        self.assertTrue(numeric_condition, "Not all elements are between 0 and 1")

    def test_ShortestPathKernel(self):
        """
        Tests the ShortestPathKernel.
        Confirms:
        - Correct matrix dimensions.
        - Diagonal elements approximately equal to 1.
        - All elements are bounded between 0 and 1.
        """

        graph_kernel = ShortestPathKernel(node_label='element')
        covariance_matrix = graph_kernel(initial_dataset)
        expected_shape = (n,n)
        self.assertEqual(covariance_matrix.shape, 
                         expected_shape, f"Expected tensor shape {expected_shape}, but got {covariance_matrix.shape}")
        
        diagonal = covariance_matrix.diag()
        expected_diagonal = torch.ones(n)
        diagonal_condition = torch.allclose(diagonal, expected_diagonal, atol=1e-3)
        self.assertTrue(diagonal_condition, "Graph Upper confidence bound test failed.")
        
        numeric_condition = torch.all((covariance_matrix >= 0) & (covariance_matrix <= 1))
        self.assertTrue(numeric_condition, "Not all elements are between 0 and 1")

    def test_WeisfeilerLehmanOptimalAssignmentKernel(self):
        """
        Tests the WeisfeilerLehmanOptimalAssignmentKernel.
        Confirms:
        - Correct matrix dimensions.
        - Diagonal elements approximately equal to 1.
        - All elements are bounded between 0 and 1.
        """

        graph_kernel = WeisfeilerLehmanOptimalAssignmentKernel(
            node_label='element', edge_label='bond')
        covariance_matrix = graph_kernel(initial_dataset)
        expected_shape = (n,n)
        self.assertEqual(covariance_matrix.shape, 
                         expected_shape, f"Expected tensor shape {expected_shape}, but got {covariance_matrix.shape}")
        
        diagonal = covariance_matrix.diag()
        expected_diagonal = torch.ones(n)
        diagonal_condition = torch.allclose(diagonal, expected_diagonal, atol=1e-3)
        self.assertTrue(diagonal_condition, "Graph Upper confidence bound test failed.")
        
        numeric_condition = torch.all((covariance_matrix >= 0) & (covariance_matrix <= 1))
        self.assertTrue(numeric_condition, "Not all elements are between 0 and 1")

if __name__ == '__main__':
    unittest.main()
