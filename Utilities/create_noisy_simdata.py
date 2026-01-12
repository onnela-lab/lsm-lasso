# Program: create_noisy_simdata.py
# Purpose: This program creates simulated datasets and is designed for a computing cluster batch array job.

import joint_estimation_utils_ME_20250922
from joint_estimation_utils_ME_20250922 import *
import sys

seed_num = int(sys.argv[1]) #sys.argv[1] is the task ID number
date = sys.argv[2] #date

# Get other arguments
n = 200 # network size
k = 2 # number of latent space dimensions
q = 25 # number of node covariates

# mean and variance of the node covariate regression coefficients
B_mu = 1
B_sigma = 0.1

# set the alpha range to [-1, -2]
a_high = -2
a_range = 1

a_high_text = str(a_high).replace(".","").replace("-", "neg")
a_range_text = str(a_range).replace(".","").replace("-", "neg")

noise_dim_list = [0, 10, 20]

for i in range(len(noise_dim_list)):
        
    noise_dim = noise_dim_list[i]
        
    input_dat = simulate_data_with_noise(n, k, q, noise_dim, B_mu, B_sigma, a_high, a_range)

    with open(str('simulated_data/simdata_' + 'n' + str(n) + '_k' + str(k) + "_noiseq" + str(noise_dim) +'-' +  str(q) + '_btypeN' + 
                  str(B_mu) + str(B_sigma) + '_a' + str(a_high_text) + str(a_range_text) + "_" + str(date) + '_' + str(seed_num) +'.pkl'), 'wb') as file: 
        # A new file will be created 
        pickle.dump(input_dat, file) 
