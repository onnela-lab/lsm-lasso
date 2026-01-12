# lsm-lasso

This repository contains Python code for the study "Covariate Selection for Joint Latent Space Modeling of Sparse Network Data".

For any further inquiries, please email emma_crenshaw@g.harvard.edu.

# File Overview

The files are split into three groups:
- Real Data: supports the generation of the real data example
- Simulations: generates the results with simulated data
- Utilities: core functions for the estimation procedure and creation of simulated data

## Real Data
The data used in this example can be found through the Harvard Dataverse at https://doi.org/10.7910/DVN/U3BIHX

__Initial Data Processing__
- microfinance_data_processing: initial data processing, generation of augmented noise variables and data binarization, selection of the 10 villages for the emulated pilot study

__Generate Results__
- microHH_MEglasso_extranoise.py: run measurement error-aware lasso on the pilot villages
- choose_var_subset.ipynb: using the measurement-error aware lasso results, choose the subset of variables to use in the study
- microHH_nolasso_extranoise.py: run the joint estimation procedure without lasso on the non-pilot villages
- microHH_limited_extranoise.py: run the joint estimation procedure without lasso on the non-pilot villages using only the selected subset of node covariates
- microHH_performance.ipynb: summarize the results and generate figures used in the paper

## Simulations
__Generate simulated datasets__
- Utilities/create_simulated_data.py: create simulated datasets

__Generate Tesults__
- fit_nocovariates.py: run the estimation procedure without information about the node covariates
- fit_nolasso.py: run the estimation procedure using the node covariates (not sparsity aware)
- fit_lasso.py: run the estimation procedure using the node covariates with lasso
- fit_lasso_ME.py: run the estimation procedure using the node covariates with measurement error-aware lasso
- simulation_performance.ipynb: Compare the results of the four different models and generate the figures in the paper


