# Baysain optimsation over graphs
![Banner](figures/assets/bo_mol_banner.jpg)
## Description

This repository provides a tool for investigating the performance of a Gaussian fitting over graphs to find molecules that maximise the score received from some function. This is primarily meant to assess a graph-based approach and how this can be optimised using different kernels and hyperparameters. However, this software can also determine the effectiveness of a graph-based approach to a particular optimisation problem. To this end, the benchmarks set out by Wenhao Gao in Sample Efficiency Matters: A Benchmark for Practical Molecular Optimization [https://openreview.net/pdf?id=yCZRdI0Y7G#:~:text=A molecular optimisation method has, that navigates this chemical space.] are implemented as optional oracles. Moreover, a deep graph kernel based on the pre-trained model provided by the Graphormer team [https://github.com/Microsoft/Graphormer] is implemented using the code supplied by hugging face to imprint with the effectiveness of deep graph neural networks for molecule representation using graphs.
To effectively use the package, one can alter the experience.yaml file according to the currently implemented features. This can be found in experiments/configs

## Installation

1. Clone the repository: `git clone https://gitlab.doc.ic.ac.uk/g237007905/bo_molecules.git`
2. Navigate to the project root directory: `cd bo_molecules`
3. Create a new enviroment using python 3.10
4. Install the dependencies by running : `installation.py`

## Usage

To run the tests using `unittest`, follow these steps:

1. Open a terminal or command prompt.
2. Navigate to the project root directory: `cd bo_molecules`
3. Run a specific test use the following command: `python3 -m unittest tests.acquisition_fun_tests`
4. Run all tests using the following command: `python -m unittest discover -s tests -p '*test*.py'`

The tests will be executed and the results will be displayed in the terminal or command prompt.

Kernels:
- Vertex Histogram
- Edge Histogram
- weisfeiler Lehman
- Neighborhood Hash
- Random Walk
- Shortest Path
- Weisfeiler LehmanOptimal Assignment
- Deep Graph Kernel

Oracles:
- albuterol_similarity
- amlodipine_mpo
- celecoxib_rediscovery
- deco_hop
- drd2
- fexofenadine_mpo
- gsk3b
- isomers_c7h8n2o2
- isomers_c9h10n2o2pf2cl
- jnk3
- median1
- median2
- mestranol_similarity
- osimertinib_mpo
- perindopril_mpo
- QED
- ranolazine_mpo
- scaffold_hop
- sitagliptin_mpo
- thiothixene_rediscovery
- troglitazone_rediscovery
- valsartan_smarts
- zaleplon_mpo

acquisition function optimisers:
- GA
- Random Sampler

Question functions:
- expected improvement
- probability of improvement
- upper confidence bound
- entropy search
- entropy search portfolio

The yaml file also takes in a set of parameters which must be specified:
- n trials (The total number of oracle calls)
- n intial samples (The number of random samples for the GP to be initially fit to)

Currently not fully implemented in the production branch:
- Sampling from a custom dataset
- Store_oracle_values_and_smiles


## Expected output

Once the appropriate alterations have been made to the experiment, the yaml file and then the BO loop can be run using the main.py file. This will produce a new results subfolder, which contains the experiment output and logging for the values generated as a function of the number of Oracle calls.