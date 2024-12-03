# Standard library imports
from typing import Callable, List, Optional, Union

# Related third party imports
import networkx as nx
from rdkit import Chem
import graphein.molecule as gm
from grakel import Graph
from grakel.utils import graph_from_networkx


def graph_to_smiles(graph: nx.Graph) -> str:
    """
    Converts a networkx graph to a SMILES string.
    
    Args:
        graph (nx.Graph): networkx graph object

    Returns: 
        str: corresponding SMILES string
    """
    return graph.graph['smiles']

def smiles_to_graph(smiles: str) -> nx.Graph:
    """
    Convert a SMILES string to a networkx graph object.

    Args:
        smiles (str): SMILES string

    Returns: 
        nx.Graph: corresponding networkx graph object
    """
    return gm.construct_graph(smiles=smiles)

def is_valid_molecule(molecule: nx.Graph) -> bool:
    """
    Check validity of the molecule using RDKit's SanitizeMol function.

    Args:
        molecule (object): Molecule to check 

    Returns:
        bool: True if molecule is valid, False otherwise
    """
    try:
        rdmol = molecule.graph['rdmol']
        Chem.SanitizeMol(rdmol)
        return True
    except:
        return False


def convert_nx_to_grakel(graphs: Union[nx.Graph, List[nx.Graph]]) -> List:
    """
    Safely convert NetworkX graph(s) to GraKeL format.
    
    Args:
        graphs: Either a single NetworkX graph or a list of NetworkX graphs
        
    Returns:
        List of graphs in GraKeL format
    """
    # single graph
    if isinstance(graphs, nx.Graph):                
        edges = list(graphs.edges())
    
        # Extract node labels (optional)
        if nx.get_node_attributes(graphs, 'label'):
            node_labels = nx.get_node_attributes(graphs, 'label')
        else:
            node_labels = {n: str(n) for n in graphs.nodes()}  # Default: node indices as labels

        # Create the Grakel graph
        grakel_graph = Graph(edges, node_labels=node_labels)

        return grakel_graph
     
    # list of graphs
    elif isinstance(graphs, list):
        grakel_graphs = []
        for g in graphs:
            edges = list(g.edges())
    
            # Extract node labels (optional)
            if nx.get_node_attributes(g, 'label'):
                node_labels = nx.get_node_attributes(g, 'label')
            else:
                node_labels = {n: str(n) for n in g.nodes()}  # Default: node indices as labels

            # Create the Grakel graph
            grakel_graphs.append(Graph(edges, node_labels=node_labels))
        
        return grakel_graphs
    
    else:
        raise TypeError(f"Expected nx.Graph or list of nx.Graph, got {type(graphs)}")