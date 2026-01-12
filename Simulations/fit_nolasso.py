# Program: fit_nolasso.py
# Purpose: runs the joint estimation procedure without using lasso

import joint_estimation_utils_ME_20250922
from joint_estimation_utils_ME_20250922 import *
import sys

seed_num = int(sys.argv[1]) #sys.argv[1] is the task ID number
date = sys.argv[2] # date
data_date = "2025-03-27" # date the simulated data was created

np.random.seed(seed_num*100)
random.seed(seed_num*100)

# Get other arguments
n = 200 # network size
k = 2 # number of latent space dimensions
q = 25 # number of node covariates

# mean and standard deviation of non-noise covariates
B_mu = 1
B_sigma = 0.1

# set alpha range to [-2, -1]
a_high = -2
a_range = 1

a_high_text = str(a_high).replace(".","").replace("-", "neg")
a_range_text = str(a_range).replace(".","").replace("-", "neg")

eta = 10
T = 100000
noise_dim_list = [0, 10, 20]

lambda1_list = [0.1]
optimizer = "adagrad"

for i in range(len(noise_dim_list)):
        
    noise_dim = noise_dim_list[i]
        
    # Read in pre-simulated data
    with open(str('simulated_data/simdata_' + 'n' + str(n) + '_k' + str(k) + "_noiseq" + str(noise_dim) +'-' +  str(q) + '_btypeN' + 
                  str(B_mu) + str(B_sigma) + '_a' + str(a_high_text) + str(a_range_text) + "_" + str(data_date) + '_' + str(seed_num) +'.pkl'), 'rb') as f:
        input_data = pickle.load(f)
    
    A_in = input_data[0]
    Y_in = input_data[1]
    B_in = input_data[2]
    Z_in = input_data[3]
    a_in = input_data[4]
    y_in = input_data[5]
    
    # get random initializations
    a_init = np.random.uniform(low = -1, high = 1, size = n)
    Z_init = np.random.normal(size = (n, k))
    y_init, B_init = optimize_gamma_B_binary(Y_in, Z_init) #don't have lasso with initialization
    
    # Run  model
    ## No lasso
    for w in range(len(lambda1_list)):
        lamb = lambda1_list[w]
         
        print("Current lambda value: ", lamb)
    
        res_nolas = run_simulations(n=n, k=k, q=q, eta=eta, lamb=lamb, T=T,
                         A_in=A_in, Y_in=Y_in,
                         Z_init=Z_init, B_init=B_init, a_init=a_init, y_init=y_init,
                         tuning_param_list=None, change_lasso_step=None,
                         adaptive=None, gamma_adapt=1.0,
                         beta1=0.9, beta2=0.999,
                         tol_loss=1e-6, patience_loss=500,
                         auto_select_lambda=False,
                         lambda_grid=None, cv_indicate = False, cv_folds=0,
                         se_rule = 0, ME_aware = False, true_avail = True, Z_true = Z_in, B_true = B_in,
                         random_state=0, optimizer = str(optimizer),
                         verbose=True)

        with open(str('sim_nolasso' + "_noiseq" + str(noise_dim) +'-' +  str(q) + "_lambda" + str(lamb) + "_" + str(optimizer) + "_" +  str(date) + '_' + str(seed_num) +'.pkl'), 'wb') as file: 
            # A new file will be created 
            pickle.dump(res_nolas, file) 

        