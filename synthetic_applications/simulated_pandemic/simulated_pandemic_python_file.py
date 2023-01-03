

'''

NOTE - This script generates the simulated pandemic distribution pickles for various numbers of ensemble members.

It does not need to be rerun, as the produced pickles are included with this code for convenience, however is included for completeness incase one wishes to run the code themselves.
 
 '''


#.py script for running code using multiprocessing -  .ipynb files

#import the dependencies
from DMDEnKF.classes.DMDEnKF import DMDEnKF
import numpy as np
import matplotlib.pyplot as plt
import cmath
import DMDEnKF.helper_functions.simulated_pandemic_functions as spf
import pickle
import multiprocess as mp
from os.path import exists

#SET ALL PARAMETERS HERE

#Data Params
num_data = 1000
num_for_spin_up = 100
max_growth = 1.01
max_decay = 0.99
low_obs_cov_const = 0.05**2
high_obs_cov_const = 0.5**2

#SELECT WHETHER USING LOW OR HIGH NOISE
obs_cov_const = high_obs_cov_const


#Model Params
#WTDMD params
window_size = 10

#ODMD params
rho = 0.9

#DMDEnKF/HDMDEnKF params
system_cov_const = (obs_cov_const/10)**2
eig_cov_const = 0.05**2/1000
ensemble_size = 100

#HDMDEnKF params
hankel_dim = 50


#Forecasting Params
n_step_ahead = 50

#Distributions from multiple runs params
num_runs = 1000

traj_args = [max_growth,max_decay,num_data,n_step_ahead,window_size,rho,num_for_spin_up,
             hankel_dim,system_cov_const,obs_cov_const,eig_cov_const,ensemble_size]

'''#Run synchronously
relative_error_distributions = np.hstack([trajectory_wrapper(i) for i in range(num_runs)])'''

#Run ascynchronously
if __name__ == '__main__':
    #variable used to calculate how many runs completed based on pickle length
    num_to_keep = (num_data - num_for_spin_up - n_step_ahead) + 1
    #variables for how many runs to perform before saving, and how many processes to use
    block_size = 4
    processors = 4
    #check if previous checkpoint exists and if so open it
    if exists(f'data/{np.sqrt(obs_cov_const)}simulated_pandemic_distributions.pkl'):
        with open(f'data/{np.sqrt(obs_cov_const)}simulated_pandemic_distributions.pkl', 'rb') as f:
            previous_checkpoint = pickle.load(f)
            #set how many runs have been completed so one can carry on where they left off
            num_completed = int(previous_checkpoint.shape[-1]/num_to_keep)
    else:
        previous_checkpoint = None
        num_completed = 0
    #loop through runs in blocks of specified size
    while num_completed + block_size <= num_runs:
        #asynchronously calculate the distributions for this block
        with mp.Pool(processors) as pool:
            relative_error_distributions = pool.starmap(spf.trajectory_wrapper,[[i,*traj_args] for i in range(num_completed,num_completed+block_size)])
            print('got to end of pool')
        final_distributions = np.hstack(relative_error_distributions)
        #add them onto the previous checkpoint (if it exists) and save
        if previous_checkpoint is not None:
            final_distributions = np.hstack([previous_checkpoint,final_distributions])
            print('prev checkpoint added')
        with open(f'data/{np.sqrt(obs_cov_const)}simulated_pandemic_distributions.pkl', 'wb') as f:
            pickle.dump(final_distributions, f)
        #Update the loop variables to show previous block has been completed
        print(f'{num_completed} to {num_completed+block_size} Complete')
        num_completed+=block_size
        previous_checkpoint = final_distributions
    print('High noise finished')



    #Low noise case
    obs_cov_const = low_obs_cov_const
    system_cov_const = (obs_cov_const/10)**2
    traj_args = [max_growth,max_decay,num_data,n_step_ahead,window_size,rho,num_for_spin_up,
             hankel_dim,system_cov_const,obs_cov_const,eig_cov_const,ensemble_size]
    
    
        #check if previous checkpoint exists and if so open it
    if exists(f'data/{np.sqrt(obs_cov_const)}simulated_pandemic_distributions.pkl'):
        with open(f'data/{np.sqrt(obs_cov_const)}simulated_pandemic_distributions.pkl', 'rb') as f:
            previous_checkpoint = pickle.load(f)
            #set how many runs have been completed so one can carry on where they left off
            num_completed = int(previous_checkpoint.shape[-1]/num_to_keep)
    else:
        previous_checkpoint = None
        num_completed = 0
    while num_completed + block_size <= num_runs:
        #asynchronously calculate the distributions for this block
        with mp.Pool(processors) as pool:
            relative_error_distributions = pool.starmap(spf.trajectory_wrapper,[[i,*traj_args] for i in range(num_completed,num_completed+block_size)])
            print('got to end of pool')
        final_distributions = np.hstack(relative_error_distributions)
        #add them onto the previous checkpoint (if it exists) and save
        if previous_checkpoint is not None:
            final_distributions = np.hstack([previous_checkpoint,final_distributions])
            print('prev checkpoint added')
        with open(f'data/{np.sqrt(obs_cov_const)}simulated_pandemic_distributions.pkl', 'wb') as f:
            pickle.dump(final_distributions, f)
        #Update the loop variables to show previous block has been completed
        print(f'{num_completed} to {num_completed+block_size} Complete')
        num_completed+=block_size
        previous_checkpoint = final_distributions
    print('Low noise finished')