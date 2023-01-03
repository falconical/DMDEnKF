import pydmd
from DMDEnKF.classes.DMDEnKF import DMDEnKF, TDMD
import numpy as np
import cmath
from DMDEnKF.classes.online_dmd import OnlineDMD
import DMDEnKF.helper_functions.simple_sin_functions as ssf


'''

Helper functions for simulated pandemic example, the third case in the synthetic applications section of the DMDEnKF paper

'''


def generate_growth_decay_operators(size,max_growth,max_decay,num_operators):
    """
    Generate a random square matrix,
    then create a list of scaled copies where the dominant eigenvalue's modulus changes linearly from max growth to max decay

    Parameters
    ----------
    size : int
        The size of the random square matrix
    max_growth : float
        The maximum value for the dominant eigenvalue's modulus to take during scaling (generally >1)
    max_decay : float
        The minimum value for the dominant eigenvalue's modulus to take during scaling (generally <1)
    num_operators : int
        The number of steps to go from the max_growth scaled copy to the max_decay scaled copy
    
    Returns
    -------
    As : list
        List of scaled copies of A where the dominant eigenvalue's modulus changes linearly from growth to decay
    """
    
    #generate random operator A, then create a list of A's
    #with dominant eigenvalue that linearly changes from max growth to max decay
    A = np.random.rand(size,size)
    eigs = np.linalg.eigvals(A)
    largest_eig = np.max([np.linalg.norm(e) for e in eigs])
    scaled_A = A/largest_eig
    As = np.linspace(scaled_A*max_growth,scaled_A*max_decay,num_operators)
    return As


def generate_data(As):
    """
    Generate measurement data, using a list of operators to represent the dynamics at the respective timestep

    Parameters
    ----------
    As : list
        List of operators for each timestep
    
    Returns
    -------
    states : numpy.array
        Array of states starting from [1,1,1], and then having operators from A applied to generate each new state
    """
    
    #use A's and apply them as the forward operator at each timestep in a discrete dynamical system,start at a vector of 1's
    state = np.array([[1],[1],[1]])
    states = [state]
    for A in As:
        state = A@state
        states.append(state)
    return np.hstack(states)


def n_step_ahead_pred(current_state,n,dmd_modes,dmd_eigs):
    """
    Predict n steps ahead using DMD modes and eigs
    Note - this does not add noise at each prediction step, so is a legacy function only

    Parameters
    ----------
    current_state : nump.array
        Current state of the system
    n : int
        Number of steps to predict ahead
    dmd_modes : numpy.array
        The DMD modes from a fitted DMD model
    dmd_eigs : numpy.array
        The eigenvalues from a fitted DMD model
    
    Returns
    -------
    pred : numpy.array
        Prediction for the new state n timesteps ahead of the input current state
    """
    
    #Basic forward prediction for DMD methods
    #Note this is not used in DMDEnKF as a prediction with stochastic noise at each step is instead applied
    diag_eigs = np.linalg.matrix_power(np.diag(dmd_eigs),n)
    pred = dmd_modes@diag_eigs@np.linalg.pinv(dmd_modes)@current_state
    return pred


'''

Apply Streaming, Windowed and Online DMD functions are very similar to those used in the simple sin helper functions

These versions also return predictions, and receive data in a slightly different format hence were written as their own functions to save uneccassary complications in the simple sin functions

'''


def iterate_streaming_tdmd(data,n_step_ahead):
    """
    Apply streaming Total DMD over each data point received

    Parameters
    ----------
    data : list
        List of measurements to perform Streaming TDMD over
    n_step_ahead : int
        Number of timesteps to predict ahead
        
    Returns
    -------
    preds : list
        List of n step ahead predicted states at each timestep
    eigs : list
        A list of the eigenvalues at each timestep
    """
    
    #Applys TDMD over the first data points, adding a new data point each step until all data has had TDMD applied to it
    #returns num_data -1 eigs that should be aligned with the tail end of the data
    eigs = []
    preds = []
    for i in range(4,len(data.T)+1):
        relevant_data = data[:,:i]
        pdmd = pydmd.DMD(2,2)
        pdmd.fit(relevant_data)
        pred = n_step_ahead_pred(np.expand_dims(relevant_data[:,-1],1),n_step_ahead,pdmd.modes,pdmd._eigs)
        eigs.append(pdmd._eigs)
        preds.append(pred)
    return preds, eigs


def windowed_tdmd(data,window_size,n_step_ahead):
    """
    Applies Windowed TDMD with a sliding window over all the data received

    Parameters
    ----------
    data : list
        List of measurements to perform Windowed TDMD over
    window_size : int
        The size of the sliding windowed to use for Windowed TDMD
    n_step_ahead : int
        Number of timesteps to predict ahead
        
    Returns
    -------
    preds : list
        List of n step ahead predicted states at each timestep
    eigs : list
        A list of the eigenvalues at each timestep
    """
    
    #also should be backwards aligned
    eigs = []
    preds = []
    for i in range(window_size,len(data.T)):
        relevant_data = data[:,i-window_size:i]
        pdmd = pydmd.DMD(2,2)
        pdmd.fit(relevant_data)
        pred = n_step_ahead_pred(np.expand_dims(relevant_data[:,-1],1),n_step_ahead,pdmd.modes,pdmd._eigs)
        eigs.append(pdmd._eigs)
        preds.append(pred)
    return preds, eigs


def iterate_odmd(data,rho,n_step_ahead):
    """
    Applies Online DMD with exponentially decaying importance over all the data received 

    Parameters
    ----------
    data : list
        List of measurements to perform Online DMD over
    rho : float
        Float between 0-1 that governs the rate of decay in the importance weighting of data from previous timesteps
    n_step_ahead : int
        Number of timesteps to predict ahead
        
    Returns
    -------
    preds : list
        List of n step ahead predicted states at each timestep
    eigs : list
        A list of the eigenvalues at each timestep
    """
    
    #applys Online DMD over the data with exponentially decayed weighting rho
    eigs_modes = []
    odmd = OnlineDMD(3,rho)
    odmd.initialize(data[:,:3],data[:,1:4])
    odmd.initializeghost()
    eigmode = odmd.computemodes()
    eigs_modes.append(eigmode)
    for i in range(3,len(data.T)-1):
        odmd.update(data[:,i:i+1],data[:,i+1:i+2])
        eigmode = odmd.computemodes()
        eigs_modes.append(eigmode)
    preds = [n_step_ahead_pred(np.expand_dims(d,1),n_step_ahead,em[1],em[0]) for d,em in zip(data[:,-len(eigs_modes):].T,eigs_modes)]
    eigs = [em[0] for em in eigs_modes]
    return preds, eigs


def apply_dmdenkf(data,num_for_spin_up,system_cov_const,obs_cov_const,eig_cov_const,ensemble_size=100,n_step_ahead=50):
    """
    Initialises the DMDEnKF, and fits it to the provided data

    Parameters
    ----------
    data : numpy.array
        Measurements to fit the DMDEnKF to
    num_for_spin_up : int
        Number of measurements to use training the DMD model in the spin-up phase
    system_cov_const : float
        Float that governs the size of the system state uncertainty covariance matrix (used in filtering step)
    obs_cov_const : float
        Float that governs the size of the measurement uncertainty covariance matrix (used in filtering step)
    eig_cov_const : float
        Float that governs the size of the system eigenvalue uncertainty covariance matrix (used in filtering step)
    ensemble_size : int
        Number of ensemble members to use in the EnKF (default 100)
    n_step_ahead : int
        Number of timesteps to predict ahead
    
    Returns
    -------
    dmdenkf_preds : numpy.array
        Array of n step ahead predicted states at each timestep
    dmdenkf_eigs : list
        A list of the eigenvalues at each timestep
    dmdenkf : DMDEnKF.DMDEnKF
        A fitted DMDEnKF object, with all relevant info to make reconstructions/predictions stored attributes
    """
    
    #Use simple sin helper function to set up the DMDEnKF wiht relevant matrices, fit and return full filter
    #Then uses this filter to produce predictions and eigenvalue estimates
    #apply dmdenkf as in the simple_sin example, then use dmdenkf object to predict and extract eigs
    dmdenkf = ssf.apply_dmdenkf(data,num_for_spin_up,system_cov_const,obs_cov_const,eig_cov_const,ensemble_size)
    #Predict and get eigs from DMDEnKF
    dmdenkf_preds = dmdenkf.fast_predict_from_ensemble(n_step_ahead).T
    dmdenkf_eigs = [dmdenkf.kf_param_state_to_eigs(x[-dmdenkf.param_state_size:]) for x in dmdenkf.X]
    #return preds, eigs and dmdenkf object
    return dmdenkf_preds, dmdenkf_eigs, dmdenkf


def apply_hankel_dmdenkf(data, num_for_spin_up,hankel_dim,system_cov_const,obs_cov_const,eig_cov_const,ensemble_size = 100, n_step_ahead=50):
    """
    Initialises the Hankel-DMDEnKF, and fits it to the provided data

    Parameters
    ----------
    data : numpy.array
        Measurements to fit the Hankel-DMDEnKF to
    num_for_spin_up : int
        Number of measurements to use training the DMD model in the spin-up phase
    hankel_dim : int
        Number of timesteps to stack in the time-delay embedded data
    system_cov_const : float
        Float that governs the size of the system state uncertainty covariance matrix (used in filtering step)
    obs_cov_const : float
        Float that governs the size of the measurement uncertainty covariance matrix (used in filtering step)
    eig_cov_const : float
        Float that governs the size of the system eigenvalue uncertainty covariance matrix (used in filtering step)
    ensemble_size : int
        Number of ensemble members to use in the EnKF (default 100)
    n_step_ahead : int
        Number of timesteps to predict ahead
    
    Returns
    -------
    hdmdenkf_preds : numpy.array
        Array of n step ahead predicted states at each timestep
    hdmdenkf_eigs : list
        A list of the eigenvalues at each timestep
    hdmdenkf : DMDEnKF.DMDEnKF
        A Hankel-DMDEnKF object fitted to the delay-embedded data,
        with all relevant info to make reconstructions/predictions stored attributes
    """
    
    #Use simple sin helper function to set up the Hankel DMDEnKF wiht relevant matrices, fit and return full filter
    #Then uses this filter to produce predictions and eigenvalue estimates
    hdmdenkf = ssf.apply_hankel_dmdenkf(data,num_for_spin_up,hankel_dim,system_cov_const,obs_cov_const,eig_cov_const,ensemble_size)
    #Predict and get eigs from DMDEnKF
    hdmdenkf_preds = hdmdenkf.fast_predict_from_ensemble(n_step_ahead).T
    hdmdenkf_eigs = [hdmdenkf.kf_param_state_to_eigs(x[-hdmdenkf.param_state_size:]) for x in hdmdenkf.X]
    #return preds, eigs and dmdenkf object
    return hdmdenkf_preds, hdmdenkf_eigs, hdmdenkf


def compute_all_methods_predictions(data,n_step_ahead,window_size,rho,
                        num_for_spin_up,hankel_dim,system_cov_const,obs_cov_const,eig_cov_const,ensemble_size):
    """
    Fits each iterative DMD variant to the provided data, then returns their n step ahead predictions

    Parameters
    ----------
    data : numpy.array
        Measurements to fit the DMD variants to
    n_step_ahead : int
        Number of timesteps to predict ahead
    window_size : int
        Number of timesteps to use in Windowed TDMD sliding window
    rho : float
        Between 0-1, exponential decay factor for the importance weighting applied to previous timesteps
    num_for_spin_up : int
        Number of measurements to use training the DMD model in the spin-up phase
    hankel_dim : int
        Number of timesteps to stack in the time-delay embedded data
    system_cov_const : float
        Float that governs the size of the system state uncertainty covariance matrix (used in filtering step)
    obs_cov_const : float
        Float that governs the size of the measurement uncertainty covariance matrix (used in filtering step)
    eig_cov_const : float
        Float that governs the size of the system eigenvalue uncertainty covariance matrix (used in filtering step)
    ensemble_size : int
        Number of ensemble members to use in the EnKF (default 100)
    
    Returns
    -------
    DMD variant predictions : numpy.array
        Array containing the n step ahead predictions for STDMD, WTDMD, ODMD, DMDEnKF and Hankel-DMDEnKF
    """
    
    #compute predictions using each of the 5 methods on data that is input
    #Appply STDMD to the data
    streaming_tdmd_preds, streaming_tdmd_eigs = iterate_streaming_tdmd(data,n_step_ahead)
    streaming_tdmd_preds = np.hstack(streaming_tdmd_preds)

    #Appply WTDMD to the data
    windowed_tdmd_preds, windowed_tdmd_eigs = windowed_tdmd(data,window_size,n_step_ahead)
    windowed_tdmd_preds = np.hstack(windowed_tdmd_preds)

    #Appply ODMD to the data
    odmd_preds, odmd_eigs = iterate_odmd(data,rho,n_step_ahead)
    odmd_preds = np.hstack(odmd_preds)
    print('Doing DMDEnKF')
    #Appply DMDEnKF to the data
    dmdenkf_preds, dmdenkf_eigs, dmdenkf = apply_dmdenkf(data,num_for_spin_up,
                                                         system_cov_const,obs_cov_const,eig_cov_const,
                                                        ensemble_size,n_step_ahead)
    print('Doing Hankel DMDEnKF')
    #Appply HDMDEnKF to the data
    hdmdenkf_preds, hdmdenkf_eigs, hdmdenkf = apply_hankel_dmdenkf(data,num_for_spin_up,hankel_dim,
                                                                  system_cov_const,obs_cov_const,eig_cov_const,
                                                                  ensemble_size,n_step_ahead)
    return np.array([streaming_tdmd_preds,windowed_tdmd_preds,odmd_preds,dmdenkf_preds,hdmdenkf_preds])


def prediction_post_processing(true_data, all_method_predictions,n_step_ahead):
    """
    Aligns the n step ahead predictions with the true data, then calculates relative errors

    Parameters
    ----------
    true_data : numpy.array
        The actual system states at each timestep (without measurement noise)
    all_method_predictions : numpy.array
        Array containing the n step ahead predictions for STDMD, WTDMD, ODMD, DMDEnKF and Hankel-DMDEnKF
    n_step_ahead : int
        Number of timesteps ahead the predictions are made for
    
    Returns
    -------
    relative_errors : list
        List of relative errors in n step ahead state predictions for STDMD, WTDMD, ODMD, DMDEnKF and Hankel-DMDEnKF
    """
    
    #After all methods have created predictions, we post process to get relative errors for each method
    #truncate and align predictions with data
    min_index = min([pred.shape[1] for pred in all_method_predictions])
    aligned_preds = [pred[:3,-min_index:-n_step_ahead] for pred in all_method_predictions]
    aligned_truth = true_data[:,-min_index+n_step_ahead:]
    relative_errors = [[relative_error(e,t) for e,t in zip(pred.T,aligned_truth.T)] for pred in aligned_preds]
    return relative_errors


def relative_error(estimate,truth):
    """
    Helper function to calculate relative errors

    Parameters
    ----------
    estimate : numpy.array
        The estimated system state
    truth : numpy.array
        The actual system state (without measurement noise)

    Returns
    -------
    relative error : numpy.array
        Relative error of the estimate from the truth
    """
        
    #Helper function for prediction post processing to calc relative errors
    return np.linalg.norm(estimate - truth)/np.linalg.norm(truth)


def trajectory_wrapper(random_state,
                      max_growth,max_decay,num_data,n_step_ahead,window_size,rho,num_for_spin_up,
             hankel_dim,system_cov_const,obs_cov_const,eig_cov_const,ensemble_size):
    """
    Generates data, fits each iterative DMD variant, then calculates the relative n step ahead state prediciton error
    This is written in a wrapper to make pythons multiprocessing library easier to utilise

    Parameters
    ----------
    random_state : int
        Numpy random seed initialiser, has to be passed to the wrapper to avoid duplicate work by multiprocesses
    max_growth : float
        maximum growth rate used at the start of the data generation (generally >1)
    max_decay : float
        maximum decay rate reached at the end of the data generation (generally <1)
    num_data : int
        Number of timesteps of data to generate
    n_step_ahead : int
        Number of timesteps to predict ahead
    window_size : int
        Number of timesteps to use in Windowed TDMD sliding window
    rho : float
        Between 0-1, exponential decay factor for the importance weighting applied to previous timesteps
    num_for_spin_up : int
        Number of measurements to use training the DMD model in the spin-up phase (less than num_data)
    hankel_dim : int
        Number of timesteps to stack in the time-delay embedded data
    system_cov_const : float
        Float that governs the size of the system state uncertainty covariance matrix (used in filtering step)
    obs_cov_const : float
        Float that governs the size of the measurement uncertainty covariance matrix (used in filtering step)
    eig_cov_const : float
        Float that governs the size of the system eigenvalue uncertainty covariance matrix (used in filtering step)
    ensemble_size : int
        Number of ensemble members to use in the EnKF (default 100)
    
    Returns
    -------
    run_relative_errors : numpy.array
        Array containing the relative errors in n step ahead predictions for STDMD, WTDMD, ODMD, DMDEnKF and Hankel-DMDEnKF
    """
    
    #wrap trajectory code to make easier to apply asynchronously
    #Generate data
    np.random.seed(random_state)
    As = generate_growth_decay_operators(3,max_growth,max_decay,num_data-1)
    true_data = generate_data(As)
    #add noise
    data = true_data + np.random.multivariate_normal([0]*3,obs_cov_const*np.identity(3),num_data).T
    #compute predictions using all methods
    all_method_predictions = compute_all_methods_predictions(data,n_step_ahead,window_size,rho,num_for_spin_up,hankel_dim,system_cov_const,obs_cov_const,eig_cov_const,ensemble_size)
    #post process to get relative errors for each method at each timestep
    run_relative_errors = prediction_post_processing(true_data,all_method_predictions,n_step_ahead)
    return np.array(run_relative_errors)


def iqr_outlier_detection(data):
    """
    Helper function to use the interquartile range method to identify outliers

    Parameters
    ----------
    data : numpy.array
        data to identify outliers in

    Returns
    -------
    outlier_prop : float
        The proportion of measurements in the data that would be considered outliers by the iqr method
    """
        
    #code that counts the percentage of outliers according to the interquartile range method
    q1 = np.quantile(data,0.25)
    q3 = np.quantile(data,0.75)
    iqr = q3 - q1
    iqr_ratio = 1.5
    possible_outlier_indices = np.where((data<(q1-iqr_ratio*iqr)) | (data>(q3+iqr_ratio*iqr)))[0]
    outlier_prop = (possible_outlier_indices.shape[0]/data.shape[0])
    print(f'Outlier Percentage: {outlier_prop*100}%')
    return outlier_prop