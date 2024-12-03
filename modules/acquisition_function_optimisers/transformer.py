# Standard library imports
from random import Random
import math
from typing import List

# Third-party imports
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from rdkit import Chem
from torch_geometric.data import DataLoader, Data
from transformers import RobertaTokenizer, RobertaForCausalLM, pipeline
import networkx as nx

# Local modules
from modules.utils.molecular_data_conversion import smiles_to_graph

class Transformer():
    """
    A transformer model for generating molecular structures as graphs.
    """

    def __init__(self,
                 n_offspring: int) -> None:
        """
        Constructs all the necessary attributes for the Transformer object.

        Args:
            n_offspring: number of offspring to generate from each parent.
        
        Returns:
            None
        """
        # Load the pre-trained ChemBERTaLM model and tokenizer
        tokenizer = RobertaTokenizer.from_pretrained("gokceuludogan/ChemBERTaLM")
        model = RobertaForCausalLM.from_pretrained("gokceuludogan/ChemBERTaLM")
        generator = pipeline("text-generation",
                             model=model, 
                             tokenizer=tokenizer)

        if torch.cuda.is_available():
            model.cuda()

        self.generator = generator
        self.n_offspring = n_offspring


    def sample(self) -> List[nx.Graph]:
        """
        Generates molecular structures offspring.

        This function utilizes a text-generation model to create SMILES strings,
        which are then converted into molecular graph representations.

        Args:
            None
        
        Returns:
            offspring: a list of molecular graph representations.
        """
        offspring = []

        while len(offspring) < self.n_offspring:
            smile = self.generator("", 
                                   max_length=128, 
                                   do_sample=True)[0]['generated_text']
            
            try:
                graph = smiles_to_graph(smile)
            except:
                print('Kekulization error. Skipping this molecule.')
                continue

            offspring.append(graph)

        return offspring
 