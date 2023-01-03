import numpy as np
from DMDEnKF.classes.DMDEnKF import TDMD
from DMDEnKF.classes.particle_filter import ParticleFilter
import DMDEnKF.helper_functions.simple_sin_functions as ssf
import cmath
from scipy.stats import multivariate_normal


'''

Helper functions for the particle filter example, the second case in the synthetic applications section of the DMDEnKF paper

'''


def run_pf_trajectory(data,true_eigs,num_for_spin_up,num_to_keep,obs_cov_const,system_cov_const,eig_cov_const,particle_num,min_effective_sample_ratio):
    """
    Runs DMD over the spin up data, then applys a particle filter over the rest of the data using DMD as system dynamics model

    Parameters
    ----------
    data : numpy.array
        Ordered measurement data of a dynamical system, taken at evenly spaced time intervals to apply DMD/particle filter to
    true_eigs : list
        The systems true eigenvalues
    num_for_spin_up : int
        The number of data points to use training the DMD model in the spin-up phase
    num_to_keep : int
        The number of unique eigenvalue modulus/arguments to retain
    obs_cov_const : float
        Float that governs the size of the measurement uncertainty covariance matrix (used in particle filter)
    system_cov_const : float
        Float that governs the size of the system state uncertainty covariance matrix (used in particle filter)
    eig_cov_const : float
        Float that governs the size of the system eigenvalue uncertainty covariance matrix (used in particle filter)
    particle_num : int
        Number of particles to use in the filter
    min_effective_sample_ratio : float
        Float between 0-1 that when effective sample size reduces below this ratio of total particles resampling occurs
    
    Returns
    -------
    particle_filter_dist : numpy.array
        Array of the distance of the particle filter's eigenvalue estimate from the true systems eigenvalue at each timestep
    """
    
    #takes in data, runs the particle filter over it, then outputs the error distribution
    #Fit spinup TDMD model
    data = np.squeeze(data).T
    f = TDMD()
    f.fit(data[:,:num_for_spin_up],r=2)


    #Generate particle filter prior via initial state/eigs and covariance
    x_len = f.data.shape[0]
    e_len = f.E.shape[0]
    Y = data[:,num_for_spin_up:]
    #check if the initial DMD found a complex conjugate pair or not
    found_conj_pair = check_conj_pairs(f)
    #initial state uses the eig mod and arg from the first eig if conjugate, simply the real eigs if not
    if found_conj_pair:
        x0 = np.hstack([Y[:,0],[cmath.polar(e) for e in f.E][0]])
    else:
        x0 = np.hstack([Y[:,0],f.E])
    P0 = np.real(np.cov(f.Y-(f.DMD_modes@np.diag(f.E)@f.inv_DMD_modes@f.X)))
    P0 = np.block([[P0,np.zeros([x_len,e_len])],[np.zeros([e_len,x_len]),np.diag([eig_cov_const]*e_len)]])
    prior = np.random.multivariate_normal(x0,P0,particle_num)
    #set relevant matrices to be used in particle filter
    observation_operator = np.hstack((np.identity(x_len),np.zeros((x_len,e_len))))
    system_cov = np.diag([system_cov_const]*x_len + [eig_cov_const]*e_len)
    observation_cov = obs_cov_const * np.identity(x_len)
    
    
    #Define model, likelihood and apply particle filter
    #Use vectorised versions of model and likelihood function for faster execution
    def fast_model(particles):
        #seperate states and eigs
        particles = np.array(particles)
        pstates = particles[:,:x_len]
        peigs = particles[:,-e_len:]
        #apply matrix-able DMD calc
        pstates = f.inv_DMD_modes@pstates.T
        #apply eigs, with neccassary preprocessing dependant on if they are a complex conj pair or not
        if found_conj_pair:
            Es = [np.diag([peig[0]*np.exp(peig[1]*1j), peig[0]*np.exp(-peig[1]*1j)]) for peig in peigs]
        else:
            Es = [np.diag(peig) for peig in peigs]
        pstates = np.array([E@pstate for E, pstate in zip(Es,pstates.T)])
        #apply matrix-able DMD calc
        pstates = (f.DMD_modes@pstates.T).T
        #reattach eigs
        particles = np.hstack([pstates,peigs])
        #add noise
        particles = particles + np.random.multivariate_normal(np.zeros(particles.shape[-1]),system_cov,particle_num)
        return particles

    def fast_likelihood(particles,measurement):
        like = multivariate_normal.pdf((observation_operator@particles.T).T,mean=measurement,cov=observation_cov)
        return like

    pf = ParticleFilter(prior,fast_model,fast_likelihood,particle_num,min_effective_sample_ratio,mode='vector')
    pf.fit(Y.T)
    
    
    #find the particle filter's eigenvalue argument estimate
    if found_conj_pair:
        particle_filter_args = [abs(p[-1]) for p in pf.X][-num_to_keep:]
    else:
        particle_filter_args = [0]* num_to_keep
    
    
    true_eigs = true_eigs[-num_to_keep:]
    particle_filter_dist = ssf.dist_from_true_data(true_eigs,particle_filter_args)
    
    return particle_filter_dist


def check_conj_pairs(DMD):
    """
    Used in the particle filter trajectory function, to check whether an eigenvalue pair is conjugate or not

    Parameters
    ----------
    DMD : DMDEnKF.DMD
        DMD object to check the eigenvlaues of
    
    Returns
    -------
    eigenvalues conjugate : bool
        Bool which indicates if the eigenvalues are conjugate (True) or not (False)
    """
    
    #quick function that checks if eigs are conjugate pair, and if not returns false
    #(this is used as a utility in the trajectory code)
    tol = 10**-8
    for eig in DMD.E:
        if abs(np.imag(eig)) < tol:
            return False
    return True