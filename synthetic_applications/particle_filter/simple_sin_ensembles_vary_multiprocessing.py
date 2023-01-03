

'''

NOTE - This script generates the simple sin distribution pickles for various numbers of ensemble members.

It does not need to be rerun, as the produced pickles are included with this code for convenience, however is included for completeness incase one wishes to run the code themselves.
 
 '''


#.py script for running code using multiprocessing - due to errors encountered when running in .ipynb files

#Import the relevant dependencies
from DMDEnKF.classes.DMDEnKF import DMDEnKF
import DMDEnKF.helper_functions.simple_sin_functions as ssf
import numpy as np
import matplotlib.pyplot as plt
import cmath
import seaborn as sb
import multiprocessing as mp
import pickle

#Set matplotlib settings as required
SMALL_SIZE = 14
MEDIUM_SIZE = 16
BIGGER_SIZE = 18
plt.rcParams["figure.figsize"] = (16,8) # (w, h)
plt.rc('font', size=SMALL_SIZE)           # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

#Initialise a random seed for reproducibility
np.random.seed(0)

#SET ALL PARAMETERS HERE

#Data Params
num_data = 500 -1
num_for_spin_up = 100
theta_start = np.pi/64
theta_end = np.pi/8
thetas = np.linspace(theta_start,theta_end,num_data)
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
eig_cov_const = 0.01**2

#HDMDEnKF params
hankel_dim = 50


#Distributions from multiple runs params
num_runs = 1000
num_to_keep = 400

#[5,10,20,30,40,50]
for ensemble_size in [40,50]:
    #trajectory wrapper so that this code can be run more easily in a asnyncronous manner
    def trajectory_wrapper(random_state):
        np.random.seed(random_state)
        #generate data
        data = ssf.generate_data(thetas)
        noise = np.expand_dims(np.random.multivariate_normal([0]*2,obs_cov_const*np.identity(2),500),2)
        data = data + noise

        #run trajectory and store outputs in array ordered (STDMD, WTDMD, ODMD, DMDEnKF, HDMDEnKF)
        traj = ssf.run_trajectory(data,num_to_keep,num_for_spin_up,thetas,window_size,rho,obs_cov_const,system_cov_const,eig_cov_const,ensemble_size,hankel_dim)
        print(random_state)
        return traj

    #Run asynchronously using multiprocessing
    pool = mp.Pool(processes=4)
    distributions = pool.map(trajectory_wrapper,[i for i in range(num_runs)])
    pool.close()
    pool.join()
    distributions = np.array(distributions)
    distributions = np.swapaxes(distributions,0,1)
    final_distributions = {'args':np.hstack(distributions[0]), 'mods':np.hstack(distributions[1])}

    with open(f'data/{ensemble_size}simple_sin_distributions.pkl', 'wb') as f:
        pickle.dump(final_distributions, f)
    print(f'Ensemble size {ensemble_size} finished')