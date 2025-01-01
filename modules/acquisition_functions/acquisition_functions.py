# Standard library imports
from typing import Tuple

# Related third party imports
import torch
import numpy as np
import gpytorch
from gpytorch.mlls import ExactMarginalLogLikelihood
from typing import List, Union
import networkx as nx
from modules.surrogate_models.gp_model import optimize_mll
from botorch.exceptions.errors import ModelFittingError
from modules.DGK_Graphormer.Graphormer_DGK import get_embedings
from modules.DGK_Graphormer import Graphormer_DGK


class AcquisitionFunction:
    """Base class for acquisition functions using graph-based inputs."""
    
    def __init__(self, model: gpytorch.models.ExactGP, 
                 best_f: Union[float, None] = None,
                 iteration_count: int = 1):
        """ Initialises the acquisition function.

        Args:
            model: model informing the acquisition function.
            best_f: best function value observed so far. Defaults to None.
            iteration_count: number of iterations of the optimisation process;
                             defaults to 1.
                             
        Returns:
            None
        """
        self.model = model
        self.best_f = best_f
        self.iteration_count = iteration_count

    def _calculate_mean_std(self, 
                            graphs: List[nx.Graph]
                            ) -> Tuple[torch.Tensor]:
        """ Return the posterior mean and standard deviation.
        
        Evaluate the GP model on a graph and use the output to calculate the
        mean and standard deviation of the posterior distribution.
        
        Args:
            graphs: molecular structures to evaluate.
        
        Returns:
            mean: mean of the posterior distribution.
            std: standard deviation of the posterior distribution.
        """ 
        # Set model to evaluation mode
        self.model.eval() 

        # Forward pass through the model
        if self.model.isDeepKernel:
            inputs = get_embedings(Graphormer_DGK.load_new_model(), graphs)

            # use LOVE to speed up inference
            with torch.no_grad(), gpytorch.settings.fast_pred_samples(), gpytorch.settings.fast_pred_var(), gpytorch.settings.max_root_decomposition_size(100):
                mu, var = self.model(inputs.to('cuda' if torch.cuda.is_available() else 'cpu'))

        else:
            mu, var = self.model(graphs)
        
        # calculate the standard deviation
        std = torch.sqrt(var)            
     
        return mu, std

    def update_best(self, best_f: float):
        """Update the best observed function value.
        
        Args:
            best_f: best function value observed so far.
        
        Returns:
            None
        """
        self.best_f = best_f


class GraphExpectedImprovement(AcquisitionFunction):
    """ Class for the Expected Improvement acquisition function. """
    
    def __init__(self, model: gpytorch.models.ExactGP = None, 
                 best_f: float = None, iteration_count: int = 1) -> None:
        """ Initialises the Expected Improvement acquisition function.
        
        Args:
            model: model informing the acquisition function.
            best_f: best function value observed so far.
            iteration_count: number of iterations of the optimisation process;
                                defaults to 1.
        
        Returns:
            None
        """
        super().__init__(model, best_f)

    def calculate_expected_improvement(self,
                                       mean: torch.Tensor, 
                                       std: torch.Tensor, 
                                       epsilon: float = 0.01) -> torch.Tensor:
        """ Calculates the expected improvement given a posterior distribution.

        Args:
            mean: mean of the posterior distribution.
            std: standard deviation of the posterior distribution.
            epsilon: minimum improvement in f(x) to be considered.

        Returns:
            exp_improvement: expected improvement of f(x) for graph x.
        """
        # Calculate the z-score
        z = (mean - self.best_f - epsilon) / (std + 1e-6)
        normal = torch.distributions.Normal(0, 1)

        # Calculate the expected improvement
        exp_improvement = (mean - self.best_f) * normal.cdf(z)
        exp_improvement += std * torch.exp(normal.log_prob(z))

        return exp_improvement

    def __call__(self, graphs: List[nx.Graph], epsilon: float = 0.01) -> torch.Tensor:
        """Computes the Expected Improvement on a graph.

        Args:
            graph: list of graphs to evaluate.
            epsilon: minimum improvement in f(x) to be considered.

        Returns:
            exp_improvement: expected improvement of graph.
        """
        # Compute the mean and standard deviation of the posterior distribution
        mean, std = self._calculate_mean_std(graphs)
        
        # Calculate z-score
        z = (mean - self.best_f - epsilon) / (std + 1e-6)
        normal = torch.distributions.Normal(0, 1)

        # Calculate the expected improvement
        exp_improvement = (mean - self.best_f) * normal.cdf(z)
        exp_improvement += std * torch.exp(normal.log_prob(z))

        return exp_improvement


class GraphProbabilityOfImprovement(AcquisitionFunction):
    """ Class for the Probability of Improvement acquisition function. """
    
    def __init__(self, model: gpytorch.models.ExactGP = None,
                 best_f: float = None, 
                 iteration_count: int = 1) -> None:
        """ Initialises the Probability of Improvement acquisition function.
        
        Args:
            model: model informing the acquisition function.
            best_f: best function value observed so far.
            iteration_count: number of iterations of the optimisation process;
                                defaults to 1.
        
        Returns:
            None
        """
        super().__init__(model, best_f)
        self.iteration_count = iteration_count

    def calculate_prob_of_improvement(self, 
                                      mean: torch.Tensor, 
                                      std: torch.Tensor, 
                                      epsilon: float = 0.1) -> torch.Tensor:
        """Calculates the probability of improvement of a graph.

        Args:
            mean: mean of the posterior distribution.
            std: standard deviation of the posterior distribution.
            epsilon: minimum improvement in f(x) to be considered.

        Returns:
            prob_of_improvement: probability of improvement of graph.
        """
        # Calculate the z-score
        normal = torch.distributions.Normal(0, 1)

        z = (mean - self.best_f - epsilon) / (std + 1e-6) # add small value to avoid division by zero

        # Calculate the probability of improvement
        prob_of_improvement = normal.cdf(z)

        return prob_of_improvement

    def __call__(self, graphs: List[nx.Graph], epsilon: float = 0.1) -> torch.Tensor:
        """Computes the Probability Improvement on a graph.

        Args:
            graph: graph to evaluate.
            epsilon: minimum improvement in f(x) to be considered.

        Returns:
            prob_of_improvement: probability improvement of graph.
        """
        # Compute the mean and standard deviation of the posterior distribution
        mean, std = self._calculate_mean_std(graphs)

        # Calculate the probability of improvement
        prob_of_improvement = self.calculate_prob_of_improvement(mean, 
                                                                 std, epsilon)

        return prob_of_improvement


class GraphUpperConfidenceBound(AcquisitionFunction):
    """ Class for the Upper Confidence Bound acquisition function. """
    
    def __init__(self, model: gpytorch.models.ExactGP = None, 
                 best_f: float = None, iteration_count: int = 1) -> None:
        """ Initialises the Upper Confidence Bound acquisition function. 
        
        Args:
            model: model informing the acquisition function.
            best_f: best function value observed so far.
            iteration_count: number of iterations of the optimisation process;
                                defaults to 1.
        
        Returns:
            None
        """
        super().__init__(model, best_f)
        self.iteration_count = iteration_count

    def calculate_upper_confidence_bound(self, 
                                         mean: torch.Tensor, 
                                         std: torch.Tensor, 
                                         epsilon: float = 0.01) -> torch.Tensor:
        
        """ Calculates the Upper Confidence Bound (UCB).
        
        For given mean and standard deviation of predictions and incorporating
        a dynamic exploration parameter based on the iteration count, the UCB
        is calculated.

        Args:
            mean: Mean predictions from the Gaussian process model.
            std: Standard deviations of predictions from the Gaussian process.
            epsilon: Exploration  decay parameter.

        Returns:
            ucb: Calculated UCB values.
        """
        
        # Calculate the exploration parameter
        epsilon = 0.2 + (2 - 0.2) * np.exp(- epsilon * self.iteration_count) # decay exponentially to 0.2

        self.iteration_count = self.iteration_count + 1
        
        # Calculate the upper confidence bound
        ucb = mean + epsilon * std

        return ucb
    
    def __call__(self, 
                graphs: List[nx.Graph],
                epsilon: float = 0.001) -> torch.Tensor:
        """Computes the Upper Confidence Bound Improvement on a graph.

        Args:
            graph: list of graphs to evaluate.
            epsilon: exploration  decay parameter.

        Returns:
            ucb: upper confidence bound of improvement of graph.
        """
     
        # Compute the mean and standard deviation of the posterior distribution
        mean, std = self._calculate_mean_std(graphs)

        ucb = self.calculate_upper_confidence_bound(mean, std, epsilon)

        return ucb


class EntropySearch(AcquisitionFunction):
    """ Class for the Entropy Search acquisition function. """
    
    def __init__(self, model: gpytorch.models.ExactGP = None,
                 best_f: float = None,
                 iteration_count: int = 1) -> None:
        """ Initialises the Entropy Search acquisition function.
        
        Args:
            model: model informing the acquisition function.
            best_f: best function value observed so far.
            iteration_count: number of iterations of the optimisation process:
                                defaults to 1.
        
        Returns:
            None
        """
        super().__init__(model, best_f)
        self.iteration_count = iteration_count

    def __call__(self, graphs: List[nx.Graph],
                 n_samples: int = 1) -> torch.Tensor:
        """Determines the information gain of a graph.

        Args:
            graphs: list of graphs to evaluate.
            n_samples: number of samples to draw from posterior.

        Returns:
            information_gains: information gain of graph.
        """
        entropy = self.calculate_entropy(self.model.covariance_matrix)
        posterior_means, posterior_stds = self._calculate_mean_std(graphs)
        information_gains = torch.zeros(len(graphs))

        for idx, graph in enumerate(graphs):
            try:
                # Evaluate the graph
                posterior_mean = posterior_means[idx]

                # Create a copy of the model
                likelihood = gpytorch.likelihoods.GaussianLikelihood()
                model_copy = self.model.__class__(
                    # add graph current data
                    NonTensorialInputs(self.model.x + [graph]),
                    # add average of samples form posterior to y data
                    torch.cat((self.model.y, torch.tensor([posterior_mean]))),
                    likelihood, self.model.kernel)
                
                # Fit the model
                mll = ExactMarginalLogLikelihood(model_copy.likelihood, 
                                                 model_copy)
                optimizer = torch.optim.Adam(model_copy.parameters(), lr=0.003)
                model_copy = optimize_mll(mll, model_copy, optimizer)

            except ModelFittingError as e:
                continue
            
            # Calculate the information gain
            information_gain = entropy - self.calculate_entropy(model_copy.covariance_matrix) # old entropy - new entropy
            information_gains[idx] = information_gain

            # Delete model copy for memory
            del model_copy  
 
        return information_gains

    def calculate_entropy(self, covariance_matrix: torch.Tensor) -> torch.Tensor:
        """Calculates the entropy of the surrogate model.

        Args:
            covariance_matrix: covariance matrix of the surrogate model.

        Returns:
            entropy: entropy of the surrogate model.
        """

        # Calculate the model entropy
        entropy = 0.5 * torch.logdet(2 * np.pi * np.e * covariance_matrix)

        return entropy

class EntropySearchPortfolio(AcquisitionFunction):
    """ Class for the Entropy Search acquisition function. """
    
    def __init__(self, model: gpytorch.models.ExactGP = None,
                best_f: float = None,
                iteration_count: int = 1) -> None:
        """ Initialises the Entropy Search acquisition function.
        
        Args:
            model: model informing the acquisition function.
            best_f: best function value observed so far.
            iteration_count: number of iterations of the optimisation process:
                                defaults to 1.
        
        Returns:
            None
        """
        self.model = model
        self.best_f = best_f
        self.iteration_count = iteration_count

    def __call__(self, graphs: List[nx.Graph],
                 n_samples: int = 1) -> torch.Tensor:
        """Determines the information gain of a graph.

        Args:
            graphs: graph to evaluate.

        Returns:
            float: information gain of graph.
            
        """

        # calculate entropy
        entropy = self.calculate_entropy(self.model.covariance_matrix)
        
        # get samples from posterior distribution
        ei, pi, ucb = self.acquisition_functions(self.model, graphs, n_samples)
        
        # get best candidates as decided by each acquisition function
        ei_max = graphs[torch.argmax(ei)]
        pi_max = graphs[torch.argmax(pi)]
        ucb_max = graphs[torch.argmax(ucb)]

        acfs = [ei, pi, ucb]
        acf_graphs = [ei_max, pi_max, ucb_max]

        information_gains = torch.zeros(len(acf_graphs))

        posterior_means, posterior_stds = self._calculate_mean_std(acf_graphs)
        
        for idx, graph in enumerate(acf_graphs):

            try: 
                # generate samples from posterior and calculate average
                posterior_mean = posterior_means[idx]

                # update copy of GP
                likelihood = gpytorch.likelihoods.GaussianLikelihood()
                # initiate new model with updated data. Using posterior mean as y value.
                model_copy = self.model.__class__(NonTensorialInputs(self.model.x + [graph]), 
                                                  torch.cat((self.model.y, torch.tensor([posterior_mean]))), 
                                                  likelihood, self.model.kernel)
                mll = ExactMarginalLogLikelihood(model_copy.likelihood, model_copy)
                optimizer = torch.optim.Adam(model_copy.parameters(), lr=0.01)
                # fit new model
                model_copy = optimize_mll(mll, model_copy, optimizer)

                # calculate new entropy after update and get reduction in entropy (information gain)
                information_gain = entropy - self.calculate_entropy(model_copy.covariance_matrix)
        
                # add information gain to tensor
                information_gains[idx] = information_gain

                # can remove model_copy from memory
                del model_copy
                
            except ModelFittingError as e:
                continue
            
        # Return values of acquisition function which maximises information gain
        acf_max_ig = acfs[torch.argmax(information_gains)]
        
        return acf_max_ig
    
    def calculate_entropy(self, covariance_matrix: torch.Tensor) -> torch.Tensor:
        """Calculates the entropy of the surrogate model.

        Args:
            covariance_matrix (torch.Tensor): covariance matrix of the surrogate model.

        Returns:
            float: entropy of the surrogate model.
        """

        # calculate differential entropy of Gaussian
        entropy = 0.5 * torch.logdet(2 * np.pi * np.e * covariance_matrix)

        return entropy
    
    def acquisition_functions(self, model, graph, epsilon=0.01):
        """Calculates acquisition function values for a given sample graph.
         
          Args:
                model: trained surrogate model.
                graph: list of graphs to evaluate.
                epsilon: minimum improvement in f(x) to be considered.
              
          Returns:
                float: expected improvement of graph.
                float: probability improvement of graph.
                float: sampled based improvement of graph.
        """

        # Get predictive mean and standard deviation
        posterior_mean, posterior_std = self._calculate_mean_std(graph)

        normal = torch.distributions.Normal(0, 1)

        ei_acf = GraphExpectedImprovement(model, self.best_f, self.iteration_count)
        pi_acf = GraphProbabilityOfImprovement(model, self.best_f, self.iteration_count)
        ucb_acf = GraphUpperConfidenceBound(model, self.best_f, self.iteration_count)

        ei = ei_acf.calculate_expected_improvement(posterior_mean, posterior_std, epsilon)
        pi = pi_acf.calculate_prob_of_improvement(posterior_mean, posterior_std, epsilon)
        ucb = ucb_acf.calculate_upper_confidence_bound(posterior_mean, posterior_std, epsilon)

        self.iteration_count = self.iteration_count + 1

        return ei, pi, ucb


class RandomSampler(AcquisitionFunction):
    """ Class for the Random Sampler acquisition function. """
    
    def __init__(self, model: gpytorch.models.ExactGP = None, 
                 best_f: float = None,
                 iteration_count: int = 1) -> None:
        """ Initialises the Random Sampler acquisition function.
        
        Args:
            model: model informing the acquisition function.
            best_f: best function value observed so far.
            iteration_count: number of iterations of the optimisation process;
                                defaults to 1.
        
        Returns:
            None
        """
        super().__init__(model, best_f)
        self.iteration_count = iteration_count

    def calculate_random_values(self, n: int) -> torch.Tensor:
        """Computes the random aquisiton values
        
        Args:
            n: number of aquisition values to generate

        Returns: 
            torch.Tensor: random aquisition values
        """
        return torch.rand(n)

    def __call__(self, graphs: List[nx.Graph]) -> torch.Tensor:
        """Computes the random aquisiton values
        
        Args:
            graphs: graphs to evaluate

        Returns: 
            torch.Tensor: random aquisition values
        """
        output = self.calculate_random_values(len(graphs))

        return output
    



class GraphUpperConfidenceBoundWithTuning(AcquisitionFunction):
    """ Class for the Upper Confidence Bound acquisition function. """
    
    def __init__(self, model: gpytorch.models.ExactGP = None, 
                 best_f: float = None, iteration_count: int = 1, 
                 epsilon: float = 0.01, asymptote: float = 0.2,
                 inital_value: float = 2.0) -> None:
        """ Initialises the Upper Confidence Bound acquisition function. 
        
        Args:
            model: model informing the acquisition function.
            best_f: best function value observed so far.
            iteration_count: number of iterations of the optimisation process;
                                defaults to 1.
        
        Returns:
            None
        """
        super().__init__(model, best_f)
        self.iteration_count = iteration_count
        self.epsilon = epsilon
        self.asymptote = asymptote
        self.inital_value = inital_value

    def calculate_upper_confidence_bound(self, 
                                         mean: torch.Tensor, 
                                         std: torch.Tensor) -> torch.Tensor:
        
        """ Calculates the Upper Confidence Bound (UCB).
        
        For given mean and standard deviation of predictions and incorporating
        a dynamic exploration parameter based on the iteration count, the UCB
        is calculated.

        Args:
            mean: Mean predictions from the Gaussian process model.
            std: Standard deviations of predictions from the Gaussian process.
            epsilon: Exploration  decay parameter.
            asymptote: The asymptote value of the UCB.
            inital_value: The initial value of the UCB.

        Returns:
            ucb: Calculated UCB values.
        """
        
        # Calculate the exploration parameter
        epsilon = self.asymptote + (self.inital_value - self.asymptote) * np.exp(- self.epsilon * self.iteration_count)

        self.iteration_count = self.iteration_count + 1
        
        # Calculate the upper confidence bound
        ucb = mean + epsilon * std

        return ucb
    
    def __call__(self, 
                graphs: List[nx.Graph],
                epsilon: float = 0.001) -> torch.Tensor:
        """Computes the Upper Confidence Bound Improvement on a graph.

        Args:
            graph: list of graphs to evaluate.
            epsilon: exploration  decay parameter.

        Returns:
            ucb: upper confidence bound of improvement of graph.
        """
     
        # Compute the mean and standard deviation of the posterior distribution
        mean, std = self._calculate_mean_std(graphs)


        ucb = self.calculate_upper_confidence_bound(mean, std)

        return ucb