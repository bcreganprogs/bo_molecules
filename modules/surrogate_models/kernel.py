import torch
from gpytorch import Module
from typing import Optional
import networkx as nx
from grakel import graph_from_networkx
from grakel.kernels import (
    VertexHistogram, 
    EdgeHistogram, 
    WeisfeilerLehman,
    NeighborhoodHash,
    RandomWalk,
    ShortestPath,
    WeisfeilerLehmanOptimalAssignment,
)

class _GraphKernel(Module):
    """
    Base class for graph kernel computation using specific graph kernels from GraKeL.

    Attributes:
        node_label (Optional[str]): The tag used to retrieve node labels from input graphs.
        edge_label (Optional[str]): The tag used to retrieve edge labels from input graphs.
        kernel (grakel.Kernel): Instance of a GraKeL graph kernel.
    """
    def __init__(self):
        """Initializes the _GraphKernel with default values."""
        super().__init__()
        self.node_label = None
        self.edge_label = None
        self.kernel = None
        self.grakel_cache = {}
        self.test_cache = {}	# cache for predictions

    def __call__(self, x):
        """
        Converts networkx graphs to GraKeL format and computes the kernel matrix.

        Args:
            x (Iterable[networkx.Graph]): An iterable of networkx graphs.

        Returns:
            torch.Tensor: A PyTorch tensor of the computed kernel matrix, in float format.
        """
        graphs = self.convert_to_grakel(x, cache='train')

        return torch.tensor(self.kernel.fit_transform(graphs)).float()
    
    def fit(self, x):
        """
        Fits the kernel to the input graphs.

        Args:
            x (Iterable[networkx.Graph]): An iterable of networkx graphs.
        """
        graphs = self.convert_to_grakel(x, cache='train')

        self.kernel.fit(graphs)

    def transform(self, x):
        """
        Transforms the input graphs into the kernel space.

        Args:
            x (Iterable[networkx.Graph]): An iterable of networkx graphs.

        Returns:
            torch.Tensor: A PyTorch tensor of the transformed graphs, in float format.
        """
        graphs = self.convert_to_grakel(x, cache='test')

        return torch.tensor(self.kernel.transform(graphs)).float()
    
    def transform_self(self, x):
        """Compute kernel values between X and itself using training fit"""
        # Use same parsing/processing as regular transform
        graphs = self.convert_to_grakel(x, cache='test')

        saved_params = self.kernel.get_params()
    
        # Get self similarities
        k2 = self.kernel.fit_transform(graphs)      # fit and transform on test points. Ideally would do with kernel fitted on training data
                                                    # but this is difficult with grakel as pairwise kernel computation is not yet implemented
        
        # Restore kernel to original state
        self.kernel.set_params(**saved_params)

        # But compute X with X instead of X with training data
        return torch.Tensor(k2)

    def convert_to_grakel(self, x, cache='train'):
        """
        Converts networkx graphs to GraKeL format.

        Args:
            x (Iterable[networkx.Graph]): An iterable of networkx graphs.

        Returns:
            List: A list of GraKeL graphs.
        """
        graphs = []
        # select correct cache, test cache is regularly reset to prevent memory issues
        if cache == 'train':
            cache = self.grakel_cache
        elif cache == 'test':
            cache = self.test_cache

        uncached = [g for g in x if g.name not in cache] # isolate uncached graphs
        if uncached:
            # convert uncached to grakel format
            grakel_versions = list(graph_from_networkx(
                uncached, 
                node_labels_tag=self.node_label, 
                edge_labels_tag=self.edge_label
            ))
            # update cache with new graphs
            cache.update({
                g.name: gk for g, gk in zip(uncached, grakel_versions)
            })
          
        # get all graphs in correct order
        graphs = [cache[g.name] for g in x]
  
        return graphs

    def clear_test_cache(self):
        """Clears the test cache."""
        self.test_cache.clear()

class VertexHistogramKernel(_GraphKernel):
    """
    A graph kernel for computing the vertex histogram kernel.

    Attributes:
        node_label (str): The tag used to retrieve node labels from input graphs.
    """
    def __init__(self, node_label: str):
        """
        Initializes the VertexHistogramKernel with the specified node label.

        Args:
            node_label (str): The tag to use for retrieving node labels from graphs.
        """
        super().__init__()
        self.kernel = VertexHistogram(normalize=True)
        self.node_label = node_label

class EdgeHistogramKernel(_GraphKernel):
    """
    A graph kernel for computing the edge histogram kernel.

    Attributes:
        edge_label (str): The tag used to retrieve edge labels from input graphs.
    """
    def __init__(self, edge_label: str):
        """
        Initializes the EdgeHistogramKernel with the specified edge label.

        Args:
            edge_label (str): The tag to use for retrieving edge labels from graphs.
        """
        super().__init__()
        self.kernel = EdgeHistogram(normalize=True)
        self.edge_label = edge_label

class WeisfeilerLehmanKernel(_GraphKernel):
    """
    A graph kernel for computing the Weisfeiler-Lehman kernel.

    Attributes:
        node_label (str): The tag used to retrieve node labels.
        edge_label (Optional[str]): The tag used to retrieve edge labels, if applicable.
    """
    def __init__(self, node_label: str, edge_label: Optional[str] = None):
        """
        Initializes the WeisfeilerLehmanKernel with specified node and optional edge labels.

        Args:
            node_label (str): The tag to use for node labels.
            edge_label (Optional[str]): The tag to use for edge labels, if any.
        """
        super().__init__()
        self.kernel = WeisfeilerLehman(n_jobs=4, n_iter=5, normalize=True)
        self.node_label = node_label
        self.edge_label = edge_label

class NeighborhoodHashKernel(_GraphKernel):
    """
    A graph kernel for computing the neighborhood hash kernel.

    Attributes:
        node_label (str): The tag used to retrieve node labels from input graphs.
    """
    def __init__(self, node_label: str):
        """
        Initializes the NeighborhoodHashKernel with the specified node label.

        Args:
            node_label (str): The tag to use for retrieving node labels from graphs.
        """
        super().__init__()
        self.kernel = NeighborhoodHash(normalize=True)
        self.node_label = node_label

class RandomWalkKernel(_GraphKernel):
    """
    A graph kernel for computing the random walk kernel.
    """
    def __init__(self):
        """Initializes the RandomWalkKernel with default settings."""
        super().__init__()
        self.kernel = RandomWalk(normalize=True)

class ShortestPathKernel(_GraphKernel):
    """
    A graph kernel for computing the shortest path kernel.

    Attributes:
        node_label (Optional[str]): The tag used to retrieve node labels, if needed for labeled paths.
    """
    def __init__(self, node_label: Optional[str] = None):
        """
        Initializes the ShortestPathKernel with an optional node label.

        Args:
            node_label (Optional[str]): The tag to use for node labels, if paths are labeled.
        """
        super().__init__()
        self.node_label = node_label
        self.kernel = ShortestPath(normalize=True, with_labels=True)

class WeisfeilerLehmanOptimalAssignmentKernel(_GraphKernel):
    """
    A graph kernel for computing the Weisfeiler-Lehman kernel.

    Attributes:
        node_label (str): The tag used to retrieve node labels.
        edge_label (Optional[str]): The tag used to retrieve edge labels, if applicable.
    """
    def __init__(self, node_label: str, edge_label: Optional[str] = None):
        """
        Initializes the WeisfeilerLehmanKernel with specified node and optional edge labels.

        Args:
            node_label (str): The tag to use for node labels.
            edge_label (Optional[str]): The tag to use for edge labels, if any.
        """
        super().__init__()
        self.kernel = WeisfeilerLehmanOptimalAssignment(n_jobs=4, n_iter=5, normalize=True)
        self.node_label = node_label
        self.edge_label = edge_label

