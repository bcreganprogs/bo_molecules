# Standard library imports
from random import Random
import math
from typing import Dict, List, Callable

# Third-party imports
import numpy as np
from rdkit import Chem
from mol_ga.graph_ga.gen_candidates import graph_ga_blended_generation
import networkx as nx

# Local imports
from modules.utils.molecular_data_conversion import graph_to_smiles, smiles_to_graph

class GeneticAlgorithm():
    """
    A genetic algorithm for generating molecular structures as graphs.
    """
    
    def __init__(self, n_parents: int, n_offspring: int,
                 fitness_function: Callable, sampling: str = 'uniform', 
                 max_atoms: int = 45, 
                 mutation_rate: float = 0.05, rng_seed: int = 42) -> None:
        """
        Initialise genetic algorithm class.
        
        Args:
            n_parents: number of parents to select from population.
            n_offspring: number of offspring to generate from each parent.
            fitness_function: function to evaluate fitness of individual.
            sampling: method for selecting parents from population;
                      options are 'uniform', 'rank', 'elite'.
            max_atoms: maximum number of atoms allowed in a molecule.
            mutation_rate: mutation rate for generating offspring.
            rng_seed: seed for the random number generator; needed for
                      reproducibility.
                      
        Returns:
            None
        """
        self.n_parents = n_parents
        self.n_offspring = n_offspring
        self.fitness_function = fitness_function
        self.sampling = sampling
        self.max_atoms = max_atoms
        self.mutation_rate = mutation_rate
        self.rng = Random(rng_seed)

    @staticmethod
    def elite_selection(population: Dict[nx.Graph, float], 
                        n_elites: int) -> Dict[nx.Graph, float]:
        """
        Select top-N individuals from population.

        Equivalent to the 'elite' selection method in the genetic algorithm.
        
        Args:
            population: dictionary containing graphs and their fitness scores.
            n_elites: number of elites to select.
            
        Returns:
            filtered_dict: containing the selected elites.
        """
        # Filter the key value pairs by fitness score
        filtered_pop = sorted(population.items(), 
                                   key=lambda item: item[1], 
                                   reverse=True)[:n_elites]
        
        filtered_dict = {k: v for k, v in filtered_pop}
        
        return filtered_dict
    
    @staticmethod
    def uniform_qualitative_sampling(population: Dict[nx.Graph, float],
                                     n_sample: int, 
                                     rng: Random,
                                     shuffle: bool = True, 
                                     min_log: int = -3,
                                     max_log: int = 0, 
                                     n_quantiles: int = 25
                                     ) -> Dict[nx.Graph, float]:
        """
        Sample SMILES by sampling uniformly from log spaced top-N.
        
        Adapted from:
        https://github.com/AustinT/mol_ga/blob/main/mol_ga/sample_population.py 
        
        This function samples following a logarithmic distribution of fitness
        scores. Higher fitness scores are sampled more frequently. The 
        distribution can be adjusted by changing the quantiles.

        Args:
            population: dictionary containing graphs and their fitness scores.
            n_sample: number of samples to select.
            rng: random number generator.
            shuffle: whether to shuffle the samples.
            min_log: minimum value for the logspace.
            max_log: maximum value for the logspace.
            n_quantiles: number of quantiles to use.
            
        Returns:
            samples: dictionary containing the selected samples.
        """
        samples = {}
        quantiles = 1 - np.logspace(min_log, max_log, n_quantiles)
        n_samples_per_quantile = int(math.ceil(n_sample / len(quantiles)))
        
        # Sort the population by score in descending order
        sorted_population = sorted(population.items(), key=lambda x: x[1], 
                                   reverse=True)
        
        for q in quantiles:
            # Only include samples above the score threshold
            score_threshold = np.quantile(
                [score for _, score in sorted_population], q)
            eligible_population = {graph: score for graph, score 
                                   in sorted_population 
                                   if score >= score_threshold}
            
            # Choose samples from the eligible population
            for _ in range(n_samples_per_quantile):
                if not eligible_population:
                    break  # Break if the eligible population is empty
                chosen_graphs = rng.choice(list(eligible_population.keys()))
                samples[chosen_graphs] = eligible_population[chosen_graphs]
                del eligible_population[chosen_graphs]
                
        if shuffle:
            # Shuffle the samples if required
            shuffled_samples = list(samples.items())
            rng.shuffle(shuffled_samples)
            samples = dict(shuffled_samples)

        # Trim the samples to ensure they don't exceed n_sample
        if len(samples) > n_sample:
            samples = dict(list(samples.items())[:n_sample])
        
        return samples

    def sample(self) -> List[nx.Graph]:
        """
        Generate offspring using the genetic algorithm.

        Function name follows the standard for acquisition function optimisers.
        
        The genetic algorithm generates offspring by selecting parents from the 
        population, applying genetic operators (crossover and mutation), and 
        ensures the offspring are within acceptable limits based on atom count. 

        Returns:
            offspring: list of offspring generated.
        """
        offspring = []

        # Sort the population by fitness and keep the best ones
        population_with_fitness = dict(self.fitness_function)

        if self.sampling == 'elite':
            selected_pop = self.elite_selection(population_with_fitness,
                                                   self.n_parents)
        elif self.sampling == 'uniform':
            selected_pop = self.uniform_qualitative_sampling(
                population_with_fitness, self.n_parents, self.rng)
            
        selected_smiles = [graph_to_smiles(graph) for graph in selected_pop.keys()]

        while len(offspring) < self.n_offspring:
            # Generate offspring using the function imported from mol_ga
            offspring_smiles = graph_ga_blended_generation(
                samples=selected_smiles, 
                n_candidates=self.n_offspring,
                rng=self.rng,
                frac_graph_ga_mutate=self.mutation_rate)

            # Filter population
            filtered_pop = [smiles for smiles
                                   in offspring_smiles 
                                   if self.has_allowed_size(smiles, 
                                                            self.max_atoms)]

            # Convert offspring to graphs
            offspring += [smiles_to_graph(smiles) for smiles in filtered_pop]
            offspring = list(set(offspring))  # Remove duplicates

        # trim offspring to pop_size
        offspring = offspring[:self.n_offspring]

        return offspring
    
    def has_allowed_size(self, smiles: str, 
                         max_atoms: int = 50) -> bool:
        """
        Check if the molecule size is within acceptable limits.
        
        Args:
            smiles: SMILES representation of the molecule.
            max_atoms: maximum number of atoms allowed in a molecule.
            
        Returns:
            bool: True if the molecule size is within acceptable limits;
                  False otherwise.
        """
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None and mol.GetNumAtoms() <= max_atoms
