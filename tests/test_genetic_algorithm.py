import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
from modules.acquisition_function_optimisers.genetic_algorithm import GeneticAlgorithm
from modules.utils.molecular_data_conversion import smiles_to_graph
import networkx as nx


class TestGeneticAlgorithm(unittest.TestCase):
    """Test the GeneticAlgorithm class."""

    def setUp(self):
        """Initialise common parameters and objects for the tests."""
        # Setup common parameters and objects used across multiple tests
        fitness_function = [('CCO', 0.5), ('CCCCCCF', 0.8), ('CCCCCO', 0.3)]
        self.population= [(smiles_to_graph(smile), score) for smile, score in fitness_function]
        self.ga = GeneticAlgorithm(n_parents=5, n_offspring=25,
                                   fitness_function=self.population, max_atoms=6)

    def test_has_allowed_size(self):
        """Test if the method correctly determines if a molecule is within the size limit."""
        acceptable_smiles = 'CCO'  # Ethanol, should be acceptable.
        unacceptable_smiles = 'C1=CC=C(C=C1)C2=CC=CC=C2'  # Biphenyl, too large if max_atoms is set low.

        self.assertTrue(self.ga.has_allowed_size(acceptable_smiles, self.ga.max_atoms),
                        f"Molecule {acceptable_smiles} unexpectedly deemed unacceptable.")
        self.assertFalse(self.ga.has_allowed_size(unacceptable_smiles, self.ga.max_atoms),
                         f"Molecule {unacceptable_smiles} unexpectedly deemed acceptable.")

    def test_sample(self):
        """Test if the method correctly samples offspring."""
        offspring = self.ga.sample()  # This method needs to be implemented in GeneticAlgorithm
        self.assertEqual(len(offspring), len(set(offspring)), "Offspring contains repeated molecules.")
        self.assertEqual(len(offspring), self.ga.n_offspring, "Incorrect number of offspring generated.")
        self.assertIsInstance(offspring, list, "Offspring not returned as a list.")
        for individual in offspring:
            self.assertIsInstance(individual, nx.Graph, "Offspring does not contain graphs.")

    def test_elite_selection(self):
        """Test if the method correctly selects the elite individuals."""
        n_elites = 2
        dict_pop = dict(self.population)
        elites = self.ga.elite_selection(dict_pop, n_elites)
        self.assertIsInstance(elites, dict, "Elite selection does not return a dictionary.")
        self.assertEqual(len(elites), n_elites, "Elite selection does not return the correct number of elites.")
        # check if the keys are nx.Graph and values are float
        for key, value in elites.items():
            self.assertIsInstance(key, nx.Graph, "Elite selection returns incorrect keys (expected nx.Graph).")
            self.assertIsInstance(value, float, "Elite selection returns incorrect values (expected float).")

    def test_uniform_qualitative_sampling(self):
        """Test if the method correctly samples individuals uniformly."""
        dict_pop = dict(self.population)
        selected_pop = self.ga.uniform_qualitative_sampling(dict_pop, 2, self.ga.rng)
        self.assertIsInstance(selected_pop, dict, "Uniform qualitative sampling does not return a dictionary.")
        # check if the keys are nx.Graph and values are float
        for key, value in selected_pop.items():
            self.assertIsInstance(key, nx.Graph, "Uniform qualitative sampling returns incorrect keys (expected nx.Graph).")
            self.assertIsInstance(value, float, "Uniform qualitative sampling returns incorrect values (expected float).")


if __name__ == '__main__':
    unittest.main()
    