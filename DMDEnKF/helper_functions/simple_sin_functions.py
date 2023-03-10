import numpy as np
import pydmd
from DMDEnKF.classes.online_dmd import OnlineDMD
from DMDEnKF.classes.DMDEnKF import  TDMD, DMDEnKF
import cmath


'''

Helper functions for the simple sin example, the first case in the synthetic applications section of the DMDEnKF paper

'''


def generate_data(thetas):
    """
    Generate data starting from (1,0) using a rotation matrix of angle theta, with theta specified at each timestep in the list

    Parameters
    ----------
    thetas : list
        List of angles to rotate by at each timestep
        
    Returns
    -------
    data : numpy.array
        The data generated by the rotation matrices
    """
    
    #generate data starting from (1,0) and applying 2D rotation matrix of angle theta for each theta in the list of thetas
    state = np.array([[1],[0]])
    states = [state]
    for theta in thetas:
        A = np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
        state = A@state
        states.append(state)
    return np.array(states)


def iterate_streaming_tdmd(data):
    """
    Apply streaming total DMD over each data point received

    Parameters
    ----------
    data : list
        List of measurements to perform Streaming TDMD over
        
    Returns
    -------
    eigs : list
        A list of the most dominant eigenvalue (largest absolute value) at each timestep
    """
    
    #Applys TDMD over the first data points, adding a new data point each step until all data has had TDMD applied to it
    #returns num_data -1 eigs that should be aligned with the tail end of the data
    eigs = []
    for i in range(2,len(data)+1):
        pdmd = pydmd.DMD(2,2)
        pdmd.fit(data[:i])
        eig = pdmd.eigs
        abs_eigs = [abs(e) for e in eig]
        dom_eig_index = abs_eigs.index(np.max(abs_eigs))
        eigs.append(eig[dom_eig_index])
    return eigs


def windowed_tdmd(data,window_size):
    """
    Applies Windowed TDMD with a sliding window over all the data received

    Parameters
    ----------
    data : list
        List of measurements to perform Windowed TDMD over
    window_size : int
        The size of the sliding windowed to use for Windowed TDMD
        
    Returns
    -------
    eigs : list
        A list of the most dominant eigenvalue (largest absolute value) at each timestep
    """
    
    #Applies windowed TDMD with specified window size
    #also should be backwards aligned
    eigs = []
    for i in range(window_size,len(data)+1):
        pdmd = pydmd.DMD(2,2)
        pdmd.fit(data[i-window_size:i])
        eig = pdmd.eigs
        abs_eigs = [abs(e) for e in eig]
        dom_eig_index = abs_eigs.index(np.max(abs_eigs))
        eigs.append(eig[dom_eig_index])
    return eigs


def iterate_odmd(data,rho):
    """
    Applies Online DMD with exponentially decaying importance over all the data received 

    Parameters
    ----------
    data : list
        List of measurements to perform Online DMD over
    rho : float
        Float between 0-1 that governs the rate of decay in the importance weighting of data from previous timesteps
        
    Returns
    -------
    eigs : list
        A list of the most dominant eigenvalue (largest absolute value) at each timestep
    """
    
    #applys Online DMD over the data with exponentially decayed weighting rho
    eigs = []
    data= np.squeeze(data)
    odmd = OnlineDMD(2,rho)
    odmd.initialize(data[:2],data[1:3])
    odmd.initializeghost()
    eig,_ = odmd.computemodes()
    eigs.append(eig[0])
    for i in range(2,len(data)-1):
        odmd.update(data[i:i+1].T,data[i+1:i+2].T)
        eig,_ = odmd.computemodes()
        abs_eigs = [abs(e) for e in eig]
        dom_eig_index = abs_eigs.index(np.max(abs_eigs))
        eigs.append(eig[dom_eig_index])
    return eigs


def apply_dmdenkf(data,num_for_spin_up,system_cov_const,obs_cov_const,eig_cov_const,ensemble_size = 100):
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
    
    Returns
    -------
    dmdenkf : DMDEnKF.DMDEnKF
        A fitted DMDEnKF object, with all relevant info to make reconstructions/predictions stored attributes
    """
    
    #Sets up the DMDEnKF wiht relevant matrices, fits and returns full filter
    #DMD Joint EnKF
    f = TDMD()
    f.fit(data[:,:num_for_spin_up],r=2)

    #Usual dmdenkf setup of inputs
    x_len = f.data.shape[0]
    e_len = f.E.shape[0]
    observation_operator = np.hstack((np.identity(x_len),np.zeros((x_len,e_len))))
    system_cov = np.diag([system_cov_const]*x_len + [eig_cov_const]*e_len)
    observation_cov = obs_cov_const * np.identity(x_len)
    #P0 = np.diag([system_cov_const]*x_len + [eig_cov_const]*e_len)
    P0 = np.real(np.cov(f.Y-(f.DMD_modes@np.diag(f.E)@np.linalg.pinv(f.DMD_modes)@f.X)))
    P0 = np.block([[P0,np.zeros([x_len,e_len])],[np.zeros([e_len,x_len]),np.diag([eig_cov_const]*e_len)]])
    Y = data[:,num_for_spin_up:]
    #Fit DMDEnKF
    dmdenkf = DMDEnKF(observation_operator=observation_operator, system_cov=system_cov,
                          observation_cov=observation_cov,P0=P0,DMD=f,ensemble_size=ensemble_size,Y=None)
    dmdenkf.fit(Y=Y)
    #return dmdenkf
    return dmdenkf


def hankelify(data, hankel_dim):
    """
    Stacks the provided data to produce a time-delay embedding

    Parameters
    ----------
    data : numpy.array
        Measurements to time-delay embed
    hankel_dim : int
        How many timesteps to use in the delay embedding
    
    Returns
    -------
    hankel_data : numpy.array
        time-delay embedded data
    """
       
    #stacks data so that matrix structure is col1: [x1,...,xhankel], col2: [x2,...,xhankel+1], etc
    hankel_list = [data[:,i:-hankel_dim + i + 1] if i+1 != hankel_dim else data[:,i:] for i in range(hankel_dim)]
    hankel_data = np.vstack(list(reversed(hankel_list)))
    return hankel_data


def apply_hankel_dmdenkf(data, num_for_spin_up,hankel_dim,system_cov_const,obs_cov_const,eig_cov_const,ensemble_size = 100):
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
    
    Returns
    -------
    test_dmdenkf : DMDEnKF.DMDEnKF
        A Hankel-DMDEnKF object fitted to the delay-embedded data,
        with all relevant info to make reconstructions/predictions stored attributes
    """
    
    #Sets up the DMDEnKF wiht relevant matrices, fits and returns full filter
    hankel_data = hankelify(data,hankel_dim)
    #DMD Joint EnKF
    f = TDMD()
    f.fit(hankel_data[:,:num_for_spin_up-(hankel_dim-1)],r=2)

    #Usual hankel dmdenkf setup of inputs
    x_len = hankel_data.shape[0]
    data_x_len = data.shape[0]
    e_len = f.E.shape[0]
    observation_operator = np.hstack((np.identity(data_x_len),np.zeros((data_x_len,x_len - data_x_len + e_len))))
    system_cov = np.diag([system_cov_const]*x_len + [eig_cov_const]*e_len)
    observation_cov = obs_cov_const * np.identity(data_x_len)
    P0 = np.real(np.cov(f.Y-(f.DMD_modes@np.diag(f.E)@f.inv_DMD_modes@f.X)))
    P0 = np.block([[P0,np.zeros([x_len,e_len])],[np.zeros([e_len,x_len]),np.diag([eig_cov_const]*e_len)]])
    #alternative initial covariance that is simply diagonal with state cov const and eig cov const used appropriately
    #P0 = np.diag([system_cov_const]*x_len + [eig_cov_const]*e_len)
    Y = data[:,num_for_spin_up:]
    #fit Hankel DMDEnKF
    test_dmdenkf = DMDEnKF(observation_operator=observation_operator, system_cov=system_cov,
                          observation_cov=observation_cov,P0=P0,DMD=f,ensemble_size=ensemble_size,Y=None)
    test_dmdenkf.fit(Y=Y)
    #return dmdenkf
    return test_dmdenkf


#Generate Error Distributions over multiple runs
def dist_from_true_data(true_eigs,eigs):
    """
    Simple helper function that takes true eigenvalues from the estimated ones (to avoid sign switiching when code repeated)

    Parameters
    ----------
    true_eigs : float
        Real eigenvalues (modulus or argument)
    eigs : float
        Estimated eigenvalues (modulus or argument)
    Returns
    -------
    eig_difference : float
        Esitmated eigs - True eigs
    """
        
    #Generate Error Distributions over multiple runs
    return eigs - true_eigs


def run_trajectory(data,num_to_keep,num_for_spin_up,true_eigs,window_size,rho,obs_cov_const,system_cov_const,eig_cov_const, ensemble_size,hankel_dim):
    """
    Fits each iterative DMD variant to the provided data, then returns their errors in the eigenvalue argument and modulus

    Parameters
    ----------
    data : numpy.array
        Measurements to fit the DMD variants to
    num_to_keep : int
        Number of dominant (largest modulus) eigenvalues to retain
    num_for_spin_up : int
        Number of measurements to use training the DMD model in the spin-up phase
    true_eigs : list
        Systems real eigenvalues
    window_size : int
        Number of timesteps to use in Windowed TDMD sliding window
    rho : float
        Between 0-1, exponential decay factor for the importance weighting applied to previous timesteps
    obs_cov_const : float
        Float that governs the size of the measurement uncertainty covariance matrix (used in filtering step)
    system_cov_const : float
        Float that governs the size of the system state uncertainty covariance matrix (used in filtering step)
    eig_cov_const : float
        Float that governs the size of the system eigenvalue uncertainty covariance matrix (used in filtering step)
    ensemble_size : int
        Number of ensemble members to use in the EnKF (default 100)
    hankel_dim : int
        Number of timesteps to stack in the time-delay embedded data
    
    Returns
    -------
    DMD argument distributions : numpy.array
        Array containing the errors in eigenvalue argument for STDMD, WTDMD, ODMD, DMDEnKF and Hankel-DMDEnKF
    DMD modulus distributions : numpy.array
        Array containing the errors in eigenvalue modulus for STDMD, WTDMD, ODMD, DMDEnKF and Hankel-DMDEnKF
    """
    
    #fit each model, then return the errors in their args and modulus
    #estimate eigenvalues using a variety of iterative methods
    streaming_tdmd_eigs = iterate_streaming_tdmd(data)
    streaming_tdmd_mods = [abs(x)-1 for x in streaming_tdmd_eigs][-num_to_keep:]
    streaming_tdmd_periods = [cmath.polar(x)[1] for x in streaming_tdmd_eigs][-num_to_keep:]
    windowed_tdmd_eigs = windowed_tdmd(data,window_size)
    windowed_tdmd_mods = [abs(x)-1 for x in windowed_tdmd_eigs][-num_to_keep:]
    windowed_tdmd_periods = [cmath.polar(x)[1] for x in windowed_tdmd_eigs][-num_to_keep:]
    odmd_eigs = iterate_odmd(data, rho)
    odmd_mods = [abs(x)-1 for x in odmd_eigs][-num_to_keep:]
    odmd_periods = [cmath.polar(x)[1] for x in odmd_eigs][-num_to_keep:]
    #squeeze data to make into standard format (legacy code from the ghost of bad codemas past)
    data = np.squeeze(data).T
    dmdenkf = apply_dmdenkf(data,num_for_spin_up,system_cov_const,obs_cov_const,eig_cov_const,ensemble_size=ensemble_size)
    #if model does not find a complex conjugate pair, use the dominant eig mod and arg of 0
    if not dmdenkf.conj_pair_list:
        dmdenkf_mods = [np.max(abs(x[-2:]))-1 for x in dmdenkf.X][-num_to_keep:]
        dmdenkf_periods = [0]* num_to_keep
    #otherwise record the mod and arg as standard
    else:
        dmdenkf_periods = [abs(x[-1]) for x in dmdenkf.X][-num_to_keep:]
        dmdenkf_mods = [abs(x[-2])-1 for x in dmdenkf.X][-num_to_keep:]
    hdmdenkf = apply_hankel_dmdenkf(data,num_for_spin_up,hankel_dim,system_cov_const,obs_cov_const,eig_cov_const,ensemble_size=ensemble_size)
    #if model does not find a complex conjugate pair, use the dominant eig mod and arg of 0
    if not hdmdenkf.conj_pair_list:
        hdmdenkf_mods = [np.max(abs(x[-2:]))-1 for x in hdmdenkf.X][-num_to_keep:]
        hdmdenkf_periods = [0]* num_to_keep
    #otherwise record the mod and arg as standard
    else:
        hdmdenkf_periods = [abs(x[-1]) for x in hdmdenkf.X][-num_to_keep:]
        hdmdenkf_mods = [abs(x[-2])-1 for x in hdmdenkf.X][-num_to_keep:]    
    true_eigs = true_eigs[-num_to_keep:]
    streaming_tdmd_dist = dist_from_true_data(true_eigs,streaming_tdmd_periods)
    windowed_tdmd_dist = dist_from_true_data(true_eigs,windowed_tdmd_periods)
    odmd_dist = dist_from_true_data(true_eigs,odmd_periods)
    dmdenkf_dist = dist_from_true_data(true_eigs,dmdenkf_periods)
    hdmdenkf_dist = dist_from_true_data(true_eigs,hdmdenkf_periods)
    return np.array([streaming_tdmd_dist, windowed_tdmd_dist, odmd_dist, dmdenkf_dist, hdmdenkf_dist]),np.array([streaming_tdmd_mods,windowed_tdmd_mods,odmd_mods,dmdenkf_mods,hdmdenkf_mods])