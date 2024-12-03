# Standard library imports
from typing import List

# Third-party imports
import networkx as nx

# Local imports
from modules.utils.read_sample_store import (
    sample_graphs_from_csv_smiles)


class DatasetSampler():
    """
    A class for generating molecular structures using random sampling.

    This class is used to generate molecular structures in graphs form by
    randomly sampling from a dataset of molecular structures. The dataset is
    typically a CSV file containing SMILES strings, which are then converted
    into molecular graph representations.
    """

    def __init__(self,
                 n_offspring: int,
                 filepath: str = 'data/zinc.csv') -> None:
        """ Initialises the DatasetSampler class.
        
        Args:
            n_offspring: The number of molecular structures to generate.
            fliepath: The path to the CSV file containing the dataset of
                        molecular structures.

        Returns:
            None
        """
        self.n_offspring = n_offspring
        self.filepath = filepath   

    def sample(self) -> List[nx.Graph]:
        """
        Generates molecular structures based on the provided inputs.

        This function utilizes a text-generation model to create
        SMILES strings, which are then converted into molecular
        graph representations.

        Args:
            None
        
        Returns:
            graphs: A list of NetworkX graph objects representing
                    the molecular structures.
        """
        graphs = sample_graphs_from_csv_smiles(self.filepath, self.n_offspring)

        return graphs