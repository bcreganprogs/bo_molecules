# Standard library imports
from typing import List
from collections import OrderedDict

# Third-party imports
import torch
import networkx as nx
from torch.nn import functional as F
from transformers.models.graphormer.collating_graphormer import (
    preprocess_item, GraphormerDataCollator)
from gauche import NonTensorialInputs

# Local imports
from modules.utils.molecular_data_conversion import graph_to_smiles
from modules.DGK_Graphormer.DGK_model_utils import (GraphormerGraphEncoder, 
                                                    smiles2graph)


def load_new_model() -> GraphormerGraphEncoder:
    """
    Return an intialised GraphormerGraphEncoder model.
    
    Returns:
        GraphormerGraphEncoder
    """
    # Define the model hyperparameters
    num_atoms = 4608
    num_in_degree = 512
    num_out_degree = 512
    num_edges = 1536
    num_spatial = 512
    num_edge_dis = 128
    edge_type = "multi_hop"
    multi_hop_max_dist = 5
    num_encoder_layers = 12

    # Load the model
    model = GraphormerGraphEncoder(num_atoms = num_atoms,
        num_in_degree = num_in_degree,
        num_out_degree = num_out_degree,
        num_edges = num_edges,
        num_spatial = num_spatial,
        num_edge_dis = num_edge_dis,
        edge_type = edge_type,
        multi_hop_max_dist = multi_hop_max_dist,
        num_encoder_layers = num_encoder_layers)
    
    # Load the pre-trained weights
    pre_trained = torch.load(
        'modules/DGK_Graphormer/HF_state_dict_pcqm4mv2_graphormer_base.pt')
    
    # Remove the encoder.graph_encoder. prefix from the pre-trained weights
    re_label = OrderedDict()
    for key in pre_trained.keys():
        if key not in  ('encoder.lm_head_transform_weight.weight', 
                        'encoder.lm_head_transform_weight.bias', 
                        'encoder.layer_norm.weight', 
                        'encoder.layer_norm.bias', 
                        'classifier.lm_output_learned_bias', 
                        'classifier.classifier.weight',
                        'encoder.graph_encoder.emb_layer_norm.weight',
                        'encoder.graph_encoder.emb_layer_norm.bias'):
            
            re_label.update(
                {key.replace('encoder.graph_encoder.',''):pre_trained[key]})
        
    model.load_state_dict(re_label)

    # Move to gpu if available
    if torch.cuda.is_available():
        model = model.cuda()
    
    elif torch.backends.mps.is_available():
        model = model.to('mps')
    
    else:
        model = model.to('cpu')

    
    return model

def get_ogbg_from_data(data: List) -> List[torch.Tensor]:
    """
    Convert the input data to ogbg format.
    
    Args:
        data: List of network x graphs to be converted wraped by
        NonTensorialInputs.
        
    Returns:
        ogbg_list: list of ogbg formatted graphs.
    """
    
    ogbg_list = []
    net_x_list = data
    if not isinstance(net_x_list, list):
        net_x_list = [net_x_list]
    
    # Convert the nx graphs to ogbg format
    for net_x_graph in net_x_list:
        smile = graph_to_smiles(net_x_graph)
        ogbg_list.append(smiles2graph(smile))
        
    return ogbg_list
      
def get_embedings(model: GraphormerGraphEncoder, data_nx) -> torch.Tensor:
    """
    Get the embedings of the input data.
    
    Args:
        model: GraphormerGraphEncoder model
        data_nx: nx graph data
        
    Returns:
        x: Embedings of the input data 
    """
    
    # Convert the input data to ogbg format
    data = get_ogbg_from_data(data_nx)
    
    # Preprocess the data
    dataset_processed = map(preprocess_item, data)

    # Collate the data
    collator = GraphormerDataCollator()
    data = collator(list(dataset_processed))



    if torch.cuda.is_available():
        labless_data = {k: v.to('cuda')
                        for k, v in data.items() if k != 'labels'}
        
    elif torch.backends.mps.is_available(): 
        labless_data = {k: v.to('mps')
                        for k, v in data.items() if k != 'labels'}
    
    else:
        labless_data = {k: v.to('cpu') for k, v in data.items() if k != 'labels'}

    labless_data = {k: v.to('cuda' if torch.cuda.is_available() else 'cpu')
                                    for k, v in data.items() if k != 'labels'}

        
    # Get the embedings
    with torch.no_grad(): 
        model.eval()
        x = model(**labless_data)
    
    return x[1]

def get_covariance_matrix(model: GraphormerGraphEncoder, 
                          data: List[nx.Graph]) -> torch.Tensor:
    """
    Get the covariance matrix of the input data.
    
    Args:
        model: GraphormerGraphEncoder model
        data: nx graph data
        
    Returns:
        conv: Covariance matrix of the input data
    """
    x = get_embedings(model, data)
    
    # Normalize the embedings
    z = F.normalize(x, p=2, dim=-1)
    
    conv = torch.matmul(z, z.T)
    return conv