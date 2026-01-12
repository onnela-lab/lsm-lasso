# Program: microHH_MEglasso_extranoise.py
# Purpose: this program runs measurement-aware lasso estimation procedure for the selected pilot villages
#          This program is designed to be parallelized over a computing cluster batch array job.
#          The array job task number selects the village to run

import joint_estimation_utils_ME_20250922
from joint_estimation_utils_ME_20250922 import *
import sys

seed_num = int(sys.argv[1]) #sys.argv[1] is the task ID number
date = sys.argv[2] #date

# Get other arguments
eta = 10
T = 75000

# Use covariate information
lamb = 0.1

village_index_list = [0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 13, 14, 15, 16, 17,
       18, 19, 20, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
       36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52,
       53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69,
       70, 71, 72, 73, 74, 75, 76]

village_list = [x+1 for x in village_index_list] 

# Read in microfinance data
seed_index = seed_num - 1
villno = village_list[seed_index]
path = 'Data/'

filenameA = path + "1. Network Data/Adjacency Matrices/adj_allVillageRelationships_HH_vilno_" + str(villno) + ".csv"
filenameY = path + "HH_binary_augment.csv"
A_in = np.loadtxt(filenameA, delimiter=",")
Y = pd.read_csv(filenameY)

Y_in = np.array(Y[Y.iloc[:, 0] == villno-1])[:,1:] #don't include village number in regression

n = len(Y_in[:,0])
q = len(Y_in[0,:])
k = 2
optimizer = "adagrad" 

print("Shape Y: :",np.shape(Y_in))

# get random initializations
a_init = np.random.uniform(low = -1, high = 1, size = n)
Z_init = np.random.normal(size = (n, k))
y_init, B_init = optimize_gamma_B_binary(Y_in, Z_init) #don't have lasso with initialization

# Run  model
lambda_grid = np.geomspace(1, 1e-4, 25)
        
print("Searched lasso penalties: ", lambda_grid)

## Group lasso
res_glas = run_simulations(n=n, k=k, q=q, eta=eta, lamb=lamb, T=T, A_in=A_in, Y_in=Y_in,
                        Z_init=Z_init, B_init=B_init, a_init=a_init, y_init=y_init,
                        tuning_param_list=None, change_lasso_step=int(T*0.9),
                         adaptive=None, gamma_adapt=1.0,
                         beta1=0.9, beta2=0.999,
                         tol_loss=1e-6, patience_loss=500,
                         auto_select_lambda=True,
                         lambda_grid=lambda_grid, cv_indicate = False, cv_folds=0, se_rule=0, ME_aware = False, 
                         true_avail = False, Z_true = None, B_true = None,
                         random_state=0, optimizer = str(optimizer),
                         verbose=True)
    
with open(str('microHHbinary_' + str(villno) + '_MEglasso_'+ '_' + str(optimizer) + '_' + str(date) +'.pkl'), 'wb') as file: 
    # A new file will be created 
    pickle.dump(res_glas, file) 
