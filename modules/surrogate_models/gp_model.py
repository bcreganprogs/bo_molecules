# Standard library imports
from typing import Tuple, Union, List

# Third party imports
import gpytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from gpytorch.constraints import GreaterThan
from gpytorch.means import ConstantMean
from gpytorch.distributions import MultivariateNormal
from sklearn.cluster import KMeans
from scipy.linalg import cholesky, solve

# Local Imports
from modules.DGK_Graphormer.Graphormer_DGK import (
    GraphormerGraphEncoder, get_covariance_matrix, get_embedings)


def initialize_model(x_train: torch.Tensor, 
                     y_train: torch.Tensor, 
                     surrogate_model: object, 
                     likelihood: gpytorch.likelihoods.Likelihood, 
                     state_dict: dict=None, 
                     covariance_matrix: torch.Tensor=None,
                     num_inducing_points: int=100
                     ) -> Tuple[gpytorch.mlls.VariationalELBO, object]:
    """
    Initialise model and loss function.

    Args:
        x_train: graphs that were previously transformed into Tensors
        y_train: tensor of outputs
        surrogate_model: model to be used
        likelihood: likelihood function to be used
        kernel: kernel function to be used
        state_dict: current state dict used to speed up fitting
        num_inducing_points: number of inducing points to be used

    Returns:
        mll: marginal log likelihood
        model: model used for surrogate model fitting
    """
    # Define model for objective
    model = surrogate_model(x_train, y_train, likelihood)
    
    # Load state dict if it is passed
    if state_dict is not None:
        state_dict.pop('variational_strategy.inducing_points', None)
        model.load_state_dict(state_dict, strict=False)
    else:
        model.likelihood.noise_covar.register_constraint("raw_noise",
                                                         GreaterThan(1e-5))
        model.likelihood.noise_covar.initialize(raw_noise=torch.tensor(1e-5))
    
    # use last covariance matrix if it is passed
    # used in entropy search a cquisition functions
    if covariance_matrix is not None:
        model.covariance_matrix = covariance_matrix
    
    # Set inducing points and likelihood lower bound
    inducing_points = torch.zeros(num_inducing_points, len(x_train))
    model.variational_strategy.inducing_points = torch.nn.Parameter(inducing_points)
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, 
                                        num_data=y_train.size(0))

    return mll, model

def optimize_mll(mll: gpytorch.mlls.VariationalELBO, 
                 model: object, 
                 optimizer: torch.optim.Optimizer,
                 num_epochs: int=100, 
                 tolerance: float=1e-5, 
                 max_iter_without_improvement: int=100,
                 decomposition_size: int=30) -> object:
    """
    Optimise the model using the marginal log likelihood.
    
    Args:
        mll: marginal log likelihood
        model: model used for surrogate model fitting
        optimizer: optimizer used for fitting
        num_epochs: number of epochs to run
        tolerance: tolerance for convergence
        max_iter_without_improvement: maximum iterations without improvement
        decomposition size: how many iterations of Lanczos we want to use for SKIP
    
    Returns:
        model: model used for surrogate model fitting
    """
    best_loss = float('inf')
    best_state_dict = None
    iter_without_improvement = 0
    model.likelihood.train()
    with gpytorch.settings.skip_posterior_variances(state=True), gpytorch.settings.skip_logdet_forward(state=True), gpytorch.settings.max_root_decomposition_size(decomposition_size):
        for _ in range(num_epochs):
            model.train()
            optimizer.zero_grad()
            output = model(model.train_inputs[0])
            loss = -mll(output, model.train_targets)

            loss.backward()
            optimizer.step()

            if loss.item() < best_loss:
                best_loss = loss.item()
                best_state_dict = model.state_dict()
                iter_without_improvement = 0
            else:
                iter_without_improvement += 1

    del optimizer
    model.load_state_dict(best_state_dict)
    model.eval()
    torch.cuda.empty_cache()

    return model

class GraphGP(gpytorch.models.ExactGP):
    """
    A subclass of the SIGP class that allows us for fitting Gaussian
    Processes with graph kernels.
    """
        
    def __init__(
        self,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        likelihood: gpytorch.likelihoods.Likelihood,
        num_inducing_points: int = 20,
        ) -> None:
        """
        Initialise the GraphGP model.
        
        Args:
            x_train: graphs that were previously transformed into Tensors
            y_train: tensor of outputs
            likelihood: likelihood function to be used
            kernel: kernel function to be used
            num_inducing_points: number of inducing points to be used
        
        Returns:
            None
        """
        super().__init__(x_train, y_train, likelihood)

        base_covar_module = gpytorch.kernels.RBFKernel()

        self.kernel = gpytorch.kernels.ProductStructureKernel(
            gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.GridInterpolationKernel(base_covar_module, grid_size=128, num_dims=1)
            ), num_dims=768
        )
        
        self.mean = ConstantMean()
        self.covariance_matrix = None
        self.covariance_model = None
        
        self.covariance = self.kernel

        # set inducing points and variational distribution for optimising likelihood lower bound
        self.inducing_points = torch.zeros(num_inducing_points, len(x_train))
        
        self.variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            num_inducing_points=num_inducing_points
        )
        
        self.variational_strategy = gpytorch.variational.VariationalStrategy(
            self,
            self.inducing_points,
            self.variational_distribution,
            learn_inducing_locations=True,
        )
        
    def forward(self, x: torch.tensor) -> MultivariateNormal:
        """
        A forward pass through the model.

        Args:
            x: Tensors

        Returns:
            MultivariateNormal: The predictive distribution.
        """
        mean = self.mean(torch.zeros(len(x), 1)).float()

        covariance = self.covariance(x).to('cuda' if torch.cuda.is_available() else 'cpu')
        # Because graph kernels operate over discrete inputs it is beneficial
        # to add some jitter for numerical stability.
        jitter = max(covariance.diag().mean().detach().item() * 1e-4, 1e-4)
        covariance += torch.eye(len(x)).to('cuda' if torch.cuda.is_available() else 'cpu') * jitter


        mult_normal = MultivariateNormal(mean, covariance)

        return mult_normal
    

class GPModel():
    """Custom written Gaussian Process model."""
    def __init__(self, kernel, inversion_method='sm', num_inducing_points=100):

        self.mean = ConstantMean()
        # self.lenthscale = nn.Parameter(torch.tensor(1.0)) not needed for DGK
        self.kernel = kernel
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.inversion_method = inversion_method
        self.num_inducing_points = num_inducing_points

        # if using DGK, we need to store the graph embeddings
        if isinstance(kernel, GraphormerGraphEncoder):
            self.isDeepKernel = True
            self.covariance_model = kernel
            self.covariance = get_covariance_matrix
            self.graph_embeddings = None
            self.x_train = torch.tensor([])
        else:
            self.isDeepKernel = False
            self.covariance = kernel
            self.x_train = []
        
        self.covariance_matrix = None
        self.cov_inverse = None

        self.y_train = torch.tensor([])

    def __call__(self, x):
        return self.forward(x)

    def fit(self, x_train, y_train):
        """Method to fit the model to the data. Can this be optimised for adding new points - no 
        need to create and retrain the model each time."""

        
        self.y_train = torch.cat((self.y_train, y_train), dim=0)
        
        if self.isDeepKernel:
            self.x_train = torch.cat((self.x_train, x_train), dim=0)
            # given points, update the covariance matrix
            if self.covariance_matrix is not None:
                # get similarity scores for new graphs with all other graphs only
                self.covariance_matrix = self.x_train @ self.x_train.T

                K_inv = self.block_matrix_inversion(x_train)
                self.cov_inverse = K_inv

                # add new points to graph embeddings
                self.graph_embeddings = torch.cat((self.graph_embeddings, x_train), dim=0)   # (N, D) where N is number of graphs and D is the dimension of the embedding (768)

            else:
                # get similarity scores for all graphs
                self.covariance_matrix = x_train @ x_train.T
                L = torch.linalg.cholesky(self.covariance_matrix)
                self.cov_inverse = torch.cholesky_inverse(L)
                # store points for future use
                self.graph_embeddings = x_train

        else: # traditional graph kernel
            # add new graph
            # grakel_graphs = convert_nx_to_grakel(x_train)
            
            if self.covariance_matrix is not None:
          
                self.x_train.extend([x_train])
                # update covariance matrix and its inverse
                # avoid recalculating covariance for fitted points with themselves
                self.kernel.fit(self.x_train)   # fit kernel to all new points
                
                cov_update = self.kernel.transform([x_train])  # get n covariances rather than n^2

                self.covariance_matrix, block_inversion_update = self.update_covariance_matrix(cov_update)

                # self.covariance_matrix = self.kernel.transform(self.x_train)

                # cholesky decomposition inverse
                # jitter = max(self.covariance_matrix.diag().mean().detach().item() * 1e-4, 1e-4)
                # self.covariance_matrix.diagonal().add_(jitter)
                # L = torch.linalg.cholesky(self.covariance_matrix)
                # self.cov_inverse = torch.cholesky_inverse(L)

                # block matrix inversion
                K_inv = self.block_matrix_inversion(block_inversion_update[:, :-1]) # remove last column which is self similarity
                self.cov_inverse = K_inv
            else:
                # calculate covariance matrix and its inverse
                self.x_train = [graph for graph in x_train]
                self.covariance_matrix = self.kernel(x_train)
                L = torch.linalg.cholesky(self.covariance_matrix)
                self.cov_inverse = torch.cholesky_inverse(L)


    def forward(self, x: Union[torch.Tensor, List]) -> MultivariateNormal:
        """
        A forward pass through the model.

        Args:
            x: Tensors

        Returns:
            MultivariateNormal: The predictive distribution.
        """

        if self.isDeepKernel:
            # get embeddings for new graphs
            new_embeddings = x#get_embeddings(self.kernel, x)

            # get covariance between new graphs and fitted graphs
            k1 = new_embeddings @ self.graph_embeddings.T

            # get covariance between new graphs with themselves
            k2 = new_embeddings @ new_embeddings.T
        else:
            # get covariance between new graphs and fitted graphs
            k1 = self.kernel.transform(x)

            # get covariance between new graphs with themselves
            k2 = self.kernel.transform_self(x)

            self.kernel.clear_test_cache()

        # Because graph kernels operate over discrete inputs it is beneficial
        # to add some jitter for numerical stability.
        jitter = max(k2.diag().mean().detach().item() * 1e-4, 1e-4)
        k2.diagonal().add_(jitter)

        mu = k1 @ self.cov_inverse @ self.y_train
        var = k2 - k1 @ self.cov_inverse @ k1.T
        var = torch.clamp(var, min=1e-8)

        return mu, var.diagonal()

    def calculate_likelihood_sm(self, x, y, covariance_matrix):
        """Calculate the likelihood of the model."""
   
        K_inv = self.sm_matrix_inversion(x)

        self.cov_inverse = K_inv

        # Three terms of the log licovariance_matrixelihcovariance_matrixod
        data_fit = -0.5 * y.T @ K_inv @ y
        sign, logdet = torch.linalg.slogdet(covariance_matrix)
        complexity = -0.5 * logdet
        constant = -0.5 * len(x) * torch.log(2 * torch.pi)
        
        return data_fit + complexity + constant
    
    def calculate_likelihood_cholesky(self, x, y, covariance_matrix):
        """Calculate the likelihood of the model."""
        
        L = torch.linalg.cholesky(covariance_matrix)  # Numerically stable vs direct inverse
        alpha = torch.linalg.solve(L.T, torch.linalg.solve(L, y))
        
        # Three terms of the log likelihood
        data_fit = -0.5 * y.T @ alpha
        complexity = -torch.sum(torch.log(torch.diag(L)))  # log determinant
        constant = -0.5 * len(x) * torch.log(2 * torch.pi)
        
        return data_fit + complexity + constant
    
    def calculate_likelihood_sparse(self, x, y, covariance_matrix):
        """Calculate the likelihood of the model."""
        
        Qff_plus_sigma = self.sparse_matrix_inversion(covariance_matrix)
        alpha = torch.linalg.solve(Qff_plus_sigma, y)
        
        # Three terms of the log likelihood
        data_fit = -0.5 * y.T @ alpha
        complexity = -0.5 * torch.log(torch.linalg.det(Qff_plus_sigma))
        constant = -0.5 * len(x) * torch.log(2 * torch.pi)
        
        return data_fit + complexity + constant

    def block_matrix_inversion(self, u: torch.Tensor) -> torch.Tensor:
        """
        Sherman-Morrison matrix inversion.
        
        Args:
            u: tensor 
        """
        cov_inverse = self.cov_inverse

        n = cov_inverse.shape[0]

        if self.isDeepKernel:
            # Compute covariances between new point and fitted points
            u = self.graph_embeddings @ u.T  # (N, 1)
        else: # otherwise input will be covariances
            u = u.T

        # Ensure u is a column vector
        if u.ndim == 1:
            u = u.reshape(-1, 1)
        
        # Using block matrix inversion formula:
        # [A    u ]^(-1) = [A^(-1) + A^(-1)u(1-u^T A^(-1)u)^(-1)u^T A^(-1)    -A^(-1)u(1-u^T A^(-1)u)^(-1) ]
        # [u^T  1 ]       = [-((1-u^T A^(-1)u)^(-1))u^T A^(-1)                  (1-u^T A^(-1)u)^(-1)         ]
        
        # Calculate common terms
        Ainv_u = cov_inverse @ u
        uT_Ainv_u = u.T @ Ainv_u
        scalar = 1.0 - uT_Ainv_u
        scalar_inv = 1.0 / (scalar + 1e-8)  # Add epsilon for numerical stability
        
        # Compute the blocks of the new inverse
        top_left = cov_inverse + scalar_inv * (Ainv_u @ Ainv_u.T)
        top_right = -scalar_inv * Ainv_u
        bottom_left = -scalar_inv * Ainv_u.T
        bottom_right = scalar_inv.reshape(1, 1)
        
        # Construct the new inverse matrix
        new_inverse = torch.zeros((n + 1, n + 1), device=cov_inverse.device)
        new_inverse[:n, :n] = top_left
        new_inverse[:n, n:] = top_right
        new_inverse[n:, :n] = bottom_left
        new_inverse[n:, n:] = bottom_right
        
        # Ensure symmetry (might be broken by numerical errors)
        new_inverse = 0.5 * (new_inverse + new_inverse.T)
        
        return new_inverse
    
    def sparse_matrix_inversion(self, inv: torch.Tensor, updates: torch.Tensor) -> torch.Tensor:
        """Inducing point matrix inversion."""
        N = inv.shape[0]  # Number of training points

        # use kmeans to select inducing points
        inducing_points = get_inducing_points()#KMeans(n_clusters=self.num_inducing_points).fit(x_train).cluster_centers_
        
        # Compute kernel matrices
        Kzz = kernel_func(inducing_points, inducing_points)  # (M x M)
        Kxz = kernel_func(inv, inducing_points)  # (N x M)
        Kzx = Kxz.T  # (M x N)
        
        # Add small jitter to diagonal for numerical stability
        Kzz += torch.eye(self.num_inducing_points) * 1e-10
        
        # Compute Cholesky decomposition of Kzz for stable inversion
        L = cholesky(Kzz, lower=True)
        
        # Compute intermediate terms
        V = solve(L, Kzx)  # L\Kzx
        
        # Compute Qff = Kxz Kzz^(-1) Kzx
        Qff = Kxz @ solve(L.T, V)  # Kxz @ (Kzz\Kzx)
        
        # Add noise variance
        Qff_plus_sigma = Qff + 1e-8 * torch.eye(N)
        
        return Qff_plus_sigma

    def update_covariance_matrix(self, update: torch.Tensor) -> torch.Tensor:
        """Update the covariance matrix with new points."""
        
        n = self.covariance_matrix.shape[0]

        new_cov = torch.zeros((n + 1, n + 1), device=self.covariance_matrix.device)

        new_cov[:n, :n] = self.covariance_matrix  # Old matrix icov_dim top-left
        new_cov[:, -1] = update.squeeze()  # New values in last column
        new_cov[-1, :] = update.squeeze()  # New values in last row
        # new_cov[-1, -1] = 1.0 # self similarity

        return new_cov, update


    def eval(self):
        """Set model to evaluation mode."""
        self.mean.eval()
        #self.kernel.eval()

        # anything else we need to set to eval mode?