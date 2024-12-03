import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Standard library imports
import unittest

# Third-party imports
import networkx as nx
import graphein.molecule as gm

# Local module imports
from modules.utils.molecular_data_conversion import (
    graph_to_smiles,
    smiles_to_graph,
    is_valid_molecule
)

class TestMolecularDataConversion(unittest.TestCase):
    
    def setUp(self):
        self.valid_smiles = 'O=C=O'
        self.valid_graph = gm.construct_graph(smiles=self.valid_smiles)

        self.invalid_smile = 'ABC'
        self.invalid_graph = nx.Graph()

    def test_valid_smiles_to_graph(self):
        """Test SMILES to graph conversion for a valid SMILES string."""
        # Convert to graph
        graph = smiles_to_graph(self.valid_smiles)

        # Check type and number of nodes
        self.assertIsInstance(graph, nx.Graph)
        self.assertEqual(graph.number_of_nodes(), 3)

    def test_valid_graph_to_smiles(self):
        """Test graph to SMILES conversion for a valid graph."""
        smile = graph_to_smiles(self.valid_graph)

        # Check type and value
        self.assertIsInstance(smile, str)
        self.assertEqual(smile, self.valid_smiles)

    def test_invalid_graph_to_smiles(self):
        """Test graph to SMILES conversion for an invalid graph."""
        with self.assertRaises(KeyError):
            smile = graph_to_smiles(self.invalid_graph)

    def test_smile_to_graph_to_smile(self):
        """Test SMILES to graph to SMILES conversion."""
        # Convert to graph
        graph = smiles_to_graph(self.valid_smiles)

        # Convert back to SMILE
        smile = graph_to_smiles(graph)

        # Check type and value
        self.assertIsInstance(smile, str)
        self.assertEqual(smile, self.valid_smiles)

    def test_is_valid_molecule_for_valid(self):
        """Test that the validity check correctly identifies a valid molecule."""
        self.assertTrue(is_valid_molecule(self.valid_graph))

    def test_is_valid_molecule_for_invalid(self):
        """Test that the validity check correctly identifies an invalid molecule."""
        self.assertFalse(is_valid_molecule(self.invalid_graph))
        
        
if __name__ == '__main__':
    unittest.main()
