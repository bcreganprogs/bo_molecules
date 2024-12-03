import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Standard library imports
import unittest
from tempfile import NamedTemporaryFile

# Third party imports
import pandas as pd
import networkx as nx

# Local imports
from modules.utils.read_sample_store import *

class TestReadSampleStore(unittest.TestCase):
    """Test cases for reading, sampling, and storing molecular data."""
    def setUp(self):
        """Create a temporary CSV file with SMILES strings for testing."""
        self.temp_file = NamedTemporaryFile(delete=False, mode='w+', suffix='.csv')
        pd.DataFrame({'smiles': ['C(C(=O)O)N', 'CCO', 'CCCN', 'CNCCCC']})\
                                    .to_csv(self.temp_file.name, index=False)
        self.temp_file.close()

    def tearDown(self):
        """Remove the temporary CSV file after testing."""
        os.remove(self.temp_file.name)

    def test_read_in_zinc_table(self):
        """Test reading in a ZINC table as a DataFrame."""
        df = pd.read_csv(self.temp_file.name)
        self.assertTrue(not df.empty, "DataFrame should not be empty.")

    def test_store_zinc_table_as_csv(self):
        """Test storing a ZINC table as a CSV file."""
        output_path = self.temp_file.name.replace('.csv', '_output.csv')
        df = pd.read_csv(self.temp_file.name)
        store_zinc_table_as_csv(df, output_path, sample_size=2)
        result_df = pd.read_csv(output_path, sep='\t')
        self.assertEqual(len(result_df), 2, 
                         "DataFrame should contain exactly 2 samples.")
        os.remove(output_path)

    def test_sample_graphs_from_smiles_csv(self):
        """Test sampling graphs from a CSV file with SMILES strings."""
        graphs, smiles = sample_graphs_from_smiles_csv(self.temp_file.name, 
                                                       sample_size=2)
        print('length of graphs:', len(graphs))
        print('length of smiles:', len(smiles))
        self.assertEqual(len(graphs), 2, "Should have sampled 2 graphs.")
        self.assertEqual(len(smiles), 2, "Should have sampled 2 SMILES strings.")

    def test_sample_smiles_from_smiles_csv(self):
        """Test sampling SMILES strings from a CSV file with SMILES strings."""
        smiles = sample_smiles_from_smiles_csv(self.temp_file.name, 
                                               sample_size=2)
        print('length of smiles:', len(smiles))
        self.assertEqual(len(smiles), 4, "Should have sampled 4 SMILES strings \
                                            to account for possible failures.")

    def test_sample_smiles_from_pd_dataframe(self):
        """Test sampling SMILES strings from a Pandas DataFrame."""
        df = pd.read_csv(self.temp_file.name)
        smiles = sample_smiles_from_pd_dataframe(df, sample_size=2)
        self.assertEqual(len(smiles), 2, 
                         "Should have sampled exactly 2 SMILES strings.")

    def test_sample_graphs_from_smiles(self):
        """Test sampling graphs from SMILES strings."""
        smiles = ['C(C(=O)O)N', 'CCO']
        graphs = sample_graphs_from_smiles(smiles, sample_size=2)
        self.assertEqual(len(graphs), 2, 
                         "Should have sampled 2 graphs from SMILES strings.")

    def test_sample_graphs_from_csv_smiles(self):
        """Test sampling graphs from a CSV file with SMILES strings."""
        graphs = sample_graphs_from_csv_smiles(self.temp_file.name, 
                                               sample_size=2)
        self.assertEqual(len(graphs), 2, 
                         "Should have sampled 2 graphs from CSV SMILES.")

    def test_store_buffer(self):
        """Test storing a buffer as a JSON file."""
        buffer = {'C(C(=O)O)N': (0.98, 1), 'CCO': (0.75, 2)}
        results_dir = 'test_results'
        os.makedirs(results_dir, exist_ok=True)
        benchmarking_metrics = {'results_dir': results_dir, 
                                'compress_file': False}
        
        # Store buffer as JSON file
        store_buffer(buffer, benchmarking_metrics)
        file_path = os.path.join(results_dir, 'buffer.json.gz')
        self.assertTrue(os.path.exists(file_path), 
                        "Buffer file should exist after storing.")
        os.remove(file_path)
        os.rmdir(results_dir)

    def test_read_compressed_or_regular_json(self):
        """Test reading a compressed or regular JSON file."""
        import json
        test_data = {'key': 'value'}
        file_path = os.path.join('test_results', 'data.json')
        os.makedirs('test_results', exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(test_data, f)
        
        read_data = read_compressed_or_regular_json(file_path)
        self.assertEqual(test_data, read_data, 
                         "Read data should match the original data.")
        os.remove(file_path)
        os.rmdir('test_results')

    def test_buffer_to_dataframe(self):
        """Test converting a buffer to a DataFrame."""
        buffer_data = {'C(C(=O)O)N': (0.95, 1), 'CCO': (0.88, 2)}
        df = buffer_to_dataframe(buffer_data)
        self.assertEqual(len(df), 2, "DataFrame should have 2 rows.")
        self.assertIn('smiles', df.columns, 
                      "DataFrame should have 'smiles' as a column.")
        self.assertIn('oracle_score', df.columns, 
                      "DataFrame should have 'oracle_score' as a column.")
        self.assertIn('iteration', df.columns, 
                      "DataFrame should have 'iteration' as a column.")


if __name__ == '__main__':
    unittest.main()
