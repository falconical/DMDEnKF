import numpy as np
from scipy.linalg import fractional_matrix_power, eig
import cmath
import pandas as pd


'''

Within this file lies the classes used to create the DMDEnKF:

1) DMD - basic DMD implementation - used as the base class for other DMD variants

2) TDMD - Total DMD (noise aware variant of DMD) - Overwrites the DMD function within DMD class only

3) EnKF - basic EnKF implementation - base class for filtering

4) DMDEnKF - Main method as described in the paper - takes in DMD object and wraps EnKF to provide it with relevant information for DMD filtering

'''


class DMD:
    """
    A class used to represent the Dynamic Mode Decomposition modelling method

    ...

    Attributes
    ----------
    data : numpy.array
        Data measured at evenly spaced time intervals of shape [state dimension, number of states]
    r : int
        Number of modes to truncate the data and subsequently model the system with
    X : numpy.array
        Data array with the final column removed
    Y : numpy.array
        Data array with the first column removed
    DMD_modes : numpy.array
        Fitted DMD modes arranged as array columns, shape [state dimension, r]
    E : numpy.array
        Eigenvalues associated with each DMD mode, length r
    b0 : numpy.array:
        First state in the data projected onto the DMD modes shape [r, 1]
    reconstruction : numpy.array
        Data reconstructed by running system forward from b0 using DMD modes and eigenvalues shape
        [state dimension, number of states]
    prediction : numpy.array
        Predicted future states by running the system forward from the most recent reconstruction onwards shape
        [state dimension, timesteps to predict]
    datatype : str
        Optional either 'ordered' (default) or 'discontinuous' to allow input of iterable with each entry containing data array
        that need not be continuous from one array to the next

    Methods
    -------
    fit(self,data=None,r=None,datatype='ordered')
        Fits DMD model to provided data
    DMD(self,r)
        Used within fit method to calculate DMD modes, eigenvalues and b0
    reconstruct(self)
        Reconstructs data using DMD model
    predict(self,timesteps)
        Predicts future state values for the future number of timesteps
    gavish_optimal_threshold(data)
        Applies Gavish method to calculate optimal number of modes to truncate the data at
    """
    
    
    def __init__(self):
        pass
    
    
    def fit(self,data=None,r=None,datatype='ordered'):
        """
        Fits DMD model to the data provided using r modes.

        Parameters
        ----------
        data : numpy.array
            Data measured at evenly spaced time intervals to fit DMD model to of shape [state dimension, number of states]
            (default None)
        r : int
            Number of modes to truncate the data to, and hence number of DMD modes/eigenvalues to produce
        datatype : str
            Optional either 'ordered' (default) or 'discontinuous' to allow input of iterable with each entry containing data
            array that need not be continuous from one array to the next
            
        Returns
        -------
        DMD_modes : numpy.array
            Fitted DMD modes arranged as array columns, shape [state dimension, r]
        E : numpy.array
            Eigenvalues associated with each DMD mode, length r
        b0 : numpy.array:
            First state in the data projected onto the DMD modes shape [r, 1]
        """
        
        #split data into inputs (X) and outputs (Y)
        #if data provided use that, otherwise check if data attribute already set
        if data is not None:
            self.data = data
        else:
            raise TypeError('You forgot to add any data boss')
            
        #if r is provided use that, otherwise calculate using Gavish method        
        if r is not None:
            self.r = r
        else:
            if datatype == 'ordered':
                data_for_rank = self.data
            elif datatype == 'discontinuous':
                data_for_rank = np.hstack(self.data)
            else:
                raise TypeError('Incorrect datatype selected: please choose from either "ordered" or "discontinuous"')
            self.r = self.gavish_optimal_threshold(data_for_rank)
            
        #determine data from the datatype attribute provided by the user
        if datatype == 'ordered':
            self.X = self.data[:,:-1]
            self.Y = self.data[:,1:]
        elif datatype == 'discontinuous':
            self.X = np.hstack([section[:,:-1] for section in self.data])
            self.Y = np.hstack([section[:,1:] for section in self.data])
        else:
            raise TypeError('Incorrect datatype selected: please choose from either "ordered" or "discontinuous"')
        #apply DMD
        DMD_modes, E, b0 = self.DMD(self.r)
        return DMD_modes, E, b0

    
    def DMD(self,r):
        """
        Used within fit method to calculate DMD modes, eigenvalues and b0 for instance attributes X and Y.

        Parameters
        ----------
        r : int
            Number of modes to truncate the data to, and hence number of DMD modes/eigenvalues to produce
            
        Returns
        -------
        DMD_modes : numpy.array
            Fitted DMD modes arranged as array columns, shape [state dimension, r]
        E : numpy.array
            Eigenvalues associated with each DMD mode, length r
        b0 : numpy.array:
            First state in the data projected onto the DMD modes shape [r, 1]
        """
        
        #Function that takes input/output states matrices and rank of svd truncation
        #Returns DMD modes, relevant eigenvectors and initial conditions
        #Set X,Y from instance attributes
        X = self.X
        Y = self.Y

        #Take SVD of X
        U,S,V = np.linalg.svd(X)

        #Extract high singular value components
        u = U[:,0:r]
        s = S[0:r]
        v = V[0:r].T

        #Use Pseudo-inverse from this approximation of X to create A_tilda (A written in U basis)
        s_inv = np.diag(1/s)
        A_tilda = u.conj().T@Y@v@s_inv

        #Find eigs of A_tilda, return eigvals as a list and eigvecs as columns of an array
        E, eigvecs = np.linalg.eig(A_tilda)
        
        #Convert eigvecs back to standard basis and format eigvals as a diagonal matrix
        #PROJECTED DMD MODES
        #DMD_modes = u@eigvecs
        #EXACT DMD MODES
        DMD_modes = Y@v@s_inv@eigvecs@np.diag(1/E)

        #calculate initial coniditons in u basis
        x0 = X[:,0].reshape(-1,1)
        b0 = np.linalg.pinv(DMD_modes)@x0

        #Finally return the 2 matrices and 1 vector needed to project states forward in time
        #This projection is DMD_modes@exp(E*t)@b0 in continuous time and DMD_modes@(E**t)@b0 in discrete time
        self.DMD_modes = DMD_modes
        self.E = E
        self.b0 = b0
        return DMD_modes, E, b0

    
    def reconstruct(self):
        """
        Reconstructs data using the already fitted DMD model.
        
        Returns
        -------
        reconstruction : numpy.array
            Data reconstructed by running system forward from b0 using DMD modes and eigenvalues shape
            [state dimension, number of states]
        """
        
        if not hasattr(self,'E'):
            raise AttributeError('you forgot to fit the data chief')
        Ed = np.diag(self.E,)
        reconstruction = np.hstack([self.DMD_modes@np.linalg.matrix_power(Ed,t)@self.b0 for t in range(self.data.shape[-1])])
        self.reconstruction = reconstruction
        return reconstruction
    
    
    def predict(self,timesteps):
        """
        Predicts future state values from the end of the reconstructed data for the future number of timesteps

        Parameters
        ----------
        timesteps : int
            The number of timesteps to predict into the future
            
        Returns
        -------
        prediction : numpy.array
            Predicted future states by running the system forward from the most recent reconstruction onwards shape
            [state dimension, timesteps to predict]
        """
        
        if not hasattr(self,'E'):
            raise AttributeError('you forgot to fit the data chief')
        Ed = np.diag(self.E)
        last_reconstruct_Ed = np.linalg.matrix_power(Ed,self.data.shape[-1])
        prediction = np.hstack([self.DMD_modes@last_reconstruct_Ed@np.linalg.matrix_power(Ed,t)@self.b0
                                for t in range(timesteps)])
        self.prediction = prediction
        return prediction
    
    
    @staticmethod 
    def gavish_optimal_threshold(data):
        """
        Applies Gavish method to calculate optimal number of modes to truncate the data at

        Parameters
        ----------
        data : numpy.array
            The data for which we want to know the optimal rank we can truncate at
            
        Returns
        -------
        r : int
            Number of modes to truncate the data to
        """
        
        singular_values = np.linalg.svd(data)[1]
        beta = data.shape[0]/data.shape[1]
        hard_thresh = (0.56*beta**3 - 0.95*beta**2 + 1.82*beta + 1.43) * np.median(singular_values)
        r = len([s for s in singular_values if s>hard_thresh])
        return r
    
    
    
    
class TDMD(DMD):
    """
    Inherits from the DMD class, and changes the DMD method to apply a total least squares fit (Total DMD) instead

    ...

    Attributes
    ----------
    All attributes from DMD class : inherited
    inv_DMD_modes : numpy.array
        Pseudoinverse of the fitted DMD modes stored for convenient reuse, shape [r, state dimension]

    Methods
    -------
    DMD(self,r)
        Used within fit method to apply Total DMD, and calculate DMD modes, eigenvalues and b0
    """
    
    
    def DMD(self,r):
        """
        Used within fit method to calculate DMD modes, eigenvalues and b0 for instance attributes X and Y using Total DMD.

        Parameters
        ----------
        r : int
            Number of modes to truncate the data to, and hence number of DMD modes/eigenvalues to produce
            
        Returns
        -------
        DMD_modes : numpy.array
            Fitted DMD modes arranged as array columns, shape [state dimension, r]
        E : numpy.array
            Eigenvalues associated with each DMD mode, length r
        b0 : numpy.array:
            First state in the data projected onto the DMD modes shape [r, 1]
        """
        
        #Function that takes input/output states matrices and rank of svd truncation
        #Returns DMD modes, relevant eigenvectors and initial conditions
        
        #Set X,Y from instance attributes
        X = self.X
        Y = self.Y
        
        #Project X and Y onto joint subspace
        Z = np.vstack((X,Y))
        zU,zS,zV = np.linalg.svd(Z)
        #zv = zV[0:X.shape[0]].T
        #Assume evolution on an r dim subspace
        zv = zV[0:r].T
        X = X@zv@zv.T
        Y = Y@zv@zv.T

        #Take SVD of X
        U,S,V = np.linalg.svd(X)

        #Extract high singular value components
        u = U[:,0:r]
        s = S[0:r]
        v = V[0:r].T

        #Use Pseudo-inverse from this approximation of X to create A_tilda (A written in U basis)
        s_inv = np.diag(1/s)
        A_tilda = u.conj().T@Y@v@s_inv

        #Find eigs of A_tilda, return eigvals as a list and eigvecs as columns of an array
        E, eigvecs = np.linalg.eig(A_tilda)
        
        #Convert eigvecs back to standard basis and format eigvals as a diagonal matrix
        #PROJECTED DMD MODES
        #DMD_modes = u@eigvecs
        #EXACT DMD MODES
        DMD_modes = Y@v@s_inv@eigvecs
        DMD_modes = DMD_modes/np.linalg.norm(DMD_modes,axis=0)
        inv_DMD_modes = np.linalg.pinv(DMD_modes)

        #calculate initial coniditons in u basis
        x0 = X[:,0].reshape(-1,1)
        b0 = np.linalg.pinv(DMD_modes)@x0

        #Finally return the 2 matrices and 1 vector needed to project states forward in time
        #This projection is DMD_modes@exp(E*t)@b0 in continuous time and DMD_modes@(E**t)@b0 in discrete time
        self.DMD_modes = DMD_modes
        self.inv_DMD_modes = inv_DMD_modes
        E = np.array([e if abs(np.imag(e))>10**-8 else np.real(e) for e in E])
        self.E = E
        self.b0 = b0
        return DMD_modes, E, b0
    
    
    
    
class EnKF():
    """
    Class built to apply the Ensemble Kalman Filter to data and store relevant results

    ...

    Attributes
    ----------
    F : Callable
        function that advances a state to it's values at the next timestep according to the system dynamics model
    H : numpy.array
        Matrix that sends a state to observation space (linearity assumed), shape [state dimension, measurement dimension]
    Q : numpy.array
        Covariance matrix for the system dynamics models uncertainty, shape [state dimension, state dimension]
    R : numpy.array
        Covariance matrix for the measurement uncertainty, shape [measurement dimension, measurement dimension]
    x : numpy.array
        Current state point estimate of the filter, length state dimension
    P : numpy.array
        Covariance matrix for the current uncertainty in the filters state estimate, shape [state dimension, state dimension]
    n : int
        Number of ensemble members in the filter
    state_size : int
        the dimension of the state, aka state dimension
    Y : numpy.array
        The new data to run the filter over, shape [state dimension, number of new measurements]
    X : list
        Record of the filters state estimate x at each pass of the filter
    vector_dynamics : bool
        Dictates whether the system dynamics function F should be applied to each ensemble member individually (False - Default),
        or should be called on array containing all ensemble members (True)        
    ensemble : numpy.array
        Current ensemble of states within the filter, shape [n, state dimension] 
    ensembles : list
        Record of the filters ensemble of states at each pass of the filter
    filter_step_counter : int
        Counter used to track how many measurements have been filtered

    Methods
    -------
    fit(self,Y=None)
        Applies the EnKF over each measurement in Y
    kalman_filter(self,y)
        Called within fit method, applies the EnKF to an individual measurement y
    propagate(self,F=None,Q=None,ensemble=None)
        Called within kalman_filter method, applies the system dynamics to ensemble
    update(self,H=None,R=None,y=None)
        Called within kalman_filter method, updates the propagated ensemble using measurement y
    """
    
    
    #NOTE: ENSEMBLE ORGANISED IN ROWS, INCOMING DATA Y ASSUMED TO BE IN COLUMNS
    def __init__(self,system_dynamics=None, observation_operator=None,
                 system_cov=None,observation_cov=None,x0=None,P0=None,ensemble_size = None,Y=None,vector_dynamics=False):
        """
        Sets relevant attributes upon creation of an EnKF instance

        Parameters
        ----------
        system_dynamics : Callable
            function that advances a state to it's values at the next timestep according to the system dynamics model, set as F
        observation_operatior : numpy.array
            Matrix that sends a state to observation space (linearity assumed) set as H,
            shape [state dimension, measurement dimension]
        system_cov : numpy.array
            Covariance matrix for the system dynamics models uncertainty set as Q, shape [state dimension, state dimension]
        observation_cov : numpy.array
            Covariance matrix for the measurement uncertainty set as R, shape [measurement dimension, measurement dimension]
        x0 : numpy.array
            Initial state point estimate of the filter, length state dimension
        P0 : numpy.array
            Covariance matrix for the initial uncertainty in the filters state estimate, shape [state dimension, state dimension]
        ensemble_size : int
            Number of ensemble members in the filter, set as n
        state_size : int
            the dimension of the state, aka state dimension
        Y : numpy.array
            The new data to run the filter over, shape [state dimension, number of new measurements]
        vector_dynamics : bool
            Dictates whether the system dynamics function F should be applied to each ensemble member individually,
            (False - Default), or should be called on array containing all ensemble members (True)
        """
        
    #set up the usual parameters of KF, also initialise the ensemble from current cov and mean if not given
        #set key variables as attributes
        self.F = system_dynamics
        self.H = observation_operator
        self.Q = system_cov
        self.R = observation_cov
        self.x = x0
        self.P = P0
        #dont record to save memory
        #self.Ps = [P0]
        self.n = ensemble_size
        self.state_size = x0.shape[0]
        self.Y = Y
        self.X = [x0]
        self.vector_dynamics = vector_dynamics
        
        #initialise ensemble with draws from N(x0,P0)
        self.ensemble = np.random.multivariate_normal(self.x,self.P,self.n)
        self.ensembles =[self.ensemble]
        
    
    def fit(self,Y=None):
        """
        Fits the EnKF to each state measurement Y.

        Parameters
        ----------
        Y : numpy.array
            Data array of newly received measurements at times t+1, t+2, ..., shape [state dimension, number of new measurements]
        """
        
        #takes in new data, deals with formatting and applies KF to it
        if Y is not None:
            self.Y = Y
        for n,y in enumerate(Y.T):
            self.filter_step_counter = n
            self.kalman_filter(y)
        
        
    def kalman_filter(self,y):
        """
        Used withn the fit method, applies the EnKF to an individual state measurement y

        Parameters
        ----------
        y : numpy.array
            New measurement to be assimilated, length measurement dimension
        """
        
        #applies both steps of the EnKF algorithm over new measurement
        print(self.filter_step_counter)
        self.propagate()
        self.update(y=y)
        self.ensembles.append(np.real(self.ensemble))
        self.X.append(self.x)
        #dont record to save memory
        #self.Ps.append(self.P)
    
    
    def propagate(self,F=None,Q=None,ensemble=None):
        """
        Used within kalman filter method, applies the system dynamics to the current ensemble 

        Parameters
        ----------
        F : Callable
            function that advances a state to it's values at the next timestep according to the system dynamics model
        Q : numpy.array
            Covariance matrix for the system dynamics models uncertainty, shape [state dimension, state dimension]     
        ensemble : numpy.array
            Current ensemble of states within the filter, shape [n, state dimension] 
            
        Returns
        -------
        P : numpy.array
            Current ensemble covariance matrix, updated to reflect the change in uncertainty post system dynamics application
        ensemble : numpy.array
            Current ensemble, updated by system dynamics, shape [n, state dimension] 
        """
        
        #uses model to propagate ensemble forward in time
        if F is not None:
            self.F = F
        if Q is not None:
            self.Q = Q
        if ensemble is not None:
            self.ensemble = ensemble
        if self.vector_dynamics:
            propagated_ensemble = self.F(self.ensemble)
        else:
            #apply model to each ensemble member
            propagated_ensemble = np.apply_along_axis(self.F,1,self.ensemble)
        #add system noise
        noise = np.random.multivariate_normal(np.zeros(self.state_size),self.Q,self.n)
        #FOUL HACK TO GET THINGS DONE QUICKLY WHEN NOISE COVARIANCE IS DIAGONAL MATRIX
        '''
        state_sd = np.sqrt(np.diag(self.Q)[0])
        param_sd = np.sqrt(np.diag(self.Q)[-1])
        state_noise = np.random.normal(0,state_sd,self.state_state_size*self.n)
        state_noise = np.reshape(state_noise,[self.n,-1])
        param_noise = np.random.normal(0,param_sd,self.param_state_size*self.n)
        param_noise = np.reshape(param_noise,[self.n,-1])
        noise = np.hstack([state_noise,param_noise])'''
        
        noised_propagated_ensemble = propagated_ensemble + noise
        #recalculate covariance matrix P
        P = np.cov(noised_propagated_ensemble.T)
        #save as attributes and return ensemble and P
        self.ensemble = noised_propagated_ensemble
        self.P = P
        return self.P, self.ensemble
        
    
    def update(self,H=None,R=None,y=None):
        """
        Used within kalman filter method, applies the measurement update from y to the current ensemble 

        Parameters
        ----------
        H : numpy.array
            Matrix that sends a state to observation space (linearity assumed), shape [state dimension, measurement dimension]
        R : numpy.array
            Covariance matrix for the measurement uncertainty, shape [measurement dimension, measurement dimension]
        y: numpy.array
            individual new measurement to assimilate, length measurement dimension
            
        Returns
        -------
        ensemble : numpy.array
            Current ensemble, updated by measurement assimilation, shape [n, state dimension]
        x : numpy.array
            Current ensemble point estimate of the state (mean), length state dimension
        P : numpy.array
            Current ensemble covariance matrix, updated to reflect the change in uncertainty post measurement update
        """
        
        #uses new measurement data to update ensemble via Kalman Gain operator
        if H is not None:
            self.H = H
        if R is not None:
            self.R = R
        if y is not None:
            self.y = y
        #peturb observation
        peturbed_observations = np.random.multivariate_normal(self.y,self.R,self.n)
        #calculate Kalman Gain
        innovation = peturbed_observations - (self.H@(self.ensemble.T)).T
        peturbed_observation_cov = np.cov(peturbed_observations.T)
        innovation_cov = self.H@self.P@(self.H.T) + peturbed_observation_cov
        K = self.P@(self.H.T)@(np.linalg.inv(innovation_cov))
        self.K = K
        updated_ensemble = self.ensemble + (K@innovation.T).T
        #save and return ensemble and it's mean and covariance in list       
        self.ensemble = updated_ensemble
        self.x = np.mean(updated_ensemble,0)
        self.P = np.cov(self.ensemble.T)
        return self.ensemble, self.x, self.P
    
    
    
    
class DMDEnKF(EnKF):
    """
    A class that implements the combination of Dynamic Mode Decomposition with the Ensemble Kalman Filter

    ...

    Attributes
    ----------
    All attributes from EnKF class: inherited
    DMD : DMDEnKF.DMD
        Fitted DMD object that will be used as the system dynamics model in the data assimilation phase
    state_state_size : 
        The dimension of the state from the DMD system
    param_state_size : 
        The number of eigenvalues in the DMD model
    conj_pair_list : 
        A list of the indices that any conjugate pair eigenvalues exist at
    polar_eigs : 
        DMD eigenvalues stored in polar cooridnates
    
    Methods
    -------
    gen_conj_pair_list(self)
        Creates a list of indices for the eigenvalues that are complex conjugate pairs
    eigs_to_kf_param_state(self,eigs)
        Converts complex eigenvalues into polar coordinates for filtering
    kf_param_state_to_eigs(self,kf_param_state)
        Returns polar eigenvalues into their complex states
    apply_ensemble_eig(self,pred_member,eig_member)
        Helper function used within fast predict method, that applies an individual eigenvalue to an ensemble state
    fast_predict_from_ensemble(self, n_steps)
        Propagate the ensemble forward n_steps, then take a state point estimate in an efficient manner
    """
    
    
    def __init__(self, observation_operator=None, system_cov=None,observation_cov=None,x0=None,P0=None,eig0=None,
                 DMD =None,ensemble_size = None,Y=None,vector_dynamics=False):
        """
        Sets relevant attributes upon creation of DMDEnKF

        Parameters
        ----------
        observation_operatior : numpy.array
            Matrix that sends a state to observation space (linearity assumed) set as H in EnKF,
            shape [state dimension, measurement dimension]
        system_cov : numpy.array
            Covariance matrix for the system dynamics models uncertainty set as Q in EnKF,
            shape [state dimension, state dimension]
        observation_cov : numpy.array
            Covariance matrix for the measurement uncertainty set as R in EnKF,
            shape [measurement dimension, measurement dimension]
        x0 : numpy.array
            Initial state point estimate of the filter, length state dimension
        P0 : numpy.array
            Covariance matrix for the initial uncertainty in the filters state estimate, shape [state dimension, state dimension]
        eig0: numpy.array
            Initial eigenvalues, only passed if filters have been run previously, otherwise default use E from the DMD instance
        DMD: DMDEnKF.DMD
            Fitted DMD object, to be used as the system dynamics model and have eigenvalues filtered
        ensemble_size : int
            Number of ensemble members in the filter, set as n in EnKF
        Y : numpy.array
            The new data to run the filter over, shape [state dimension, number of new measurements]
        vector_dynamics : bool
            Dictates whether the system dynamics function F should be applied to each ensemble member individually,
            (False - Default), or should be called on array containing all ensemble members (True)
        """
        
    #takes in the some stuff for EnKF but also a DMD object to initialise the rest automatically
        
        #store DMD object and size of state and params to be accessed later
        self.DMD = DMD
        self.DMD.DMD_inv_modes = np.linalg.pinv(self.DMD.DMD_modes)
        self.state_state_size = self.DMD.X[:,-1].shape[0]
        self.param_state_size = self.DMD.E.shape[0]
        
        #create list of conjugate pairs and polar eigs to use in eig to kf_param_state transformations
        self.conj_pair_list = self.gen_conj_pair_list()
        if eig0 is None:
            eig0 = self.DMD.E
        self.polar_eigs = np.array([cmath.polar(c) for c in eig0])
        kf_param_state = self.eigs_to_kf_param_state(eig0)
        #create initial state x0 from last state data point in DMD and current eigenvalue thetas
        if x0 is None:
            #legacy code when x0 could be assumed
            x0_state = self.DMD.data[:,-1]
        else:
            x0_state = x0
        x0 = np.concatenate((x0_state,kf_param_state))
        x0 = np.real(x0)
        
        #required for "hacking" the pickle to make it save
        global system_dynamics
        if vector_dynamics:
            def system_dynamics(ensembles):
                predos = np.array(ensembles)
                original_shape = predos.shape
                predos = np.reshape(predos,[-1,self.x.shape[0]])
                eigs = predos[:,-self.param_state_size:]
                one_step_forward_ens = (self.DMD.DMD_inv_modes@(predos[:,:self.state_state_size].T)).T
                one_step_forward_ens = np.array([self.apply_ensemble_eig(pred_member,eig_member) for pred_member,eig_member in zip(one_step_forward_ens,eigs)])
                one_step_forward_ens = (self.DMD.DMD_modes@(one_step_forward_ens.T)).T
                predos = np.hstack([one_step_forward_ens,eigs])
                predos = np.reshape(predos,original_shape)
                return predos
        
        else:        
            #create system dynamics function that takes in augmented state and outputs next propagated state
            def system_dynamics(state):
                #first split state into state_state and param_state
                state_state = state[:self.state_state_size]
                param_state = state[-self.param_state_size:]
                #then return the param_state to it's original eigenvalue form and advance forward in time
                eigs = self.kf_param_state_to_eigs(param_state)
                A = self.DMD.DMD_modes@np.diag(eigs)@self.DMD.DMD_inv_modes
                new_state_state = A@state_state
                #stick param_state back on the end as no systematic parameter drift is assumed
                new_state = np.concatenate((new_state_state,param_state))
                return new_state

        super().__init__(system_dynamics=system_dynamics, observation_operator=observation_operator,
                 system_cov=system_cov,observation_cov=observation_cov,x0=x0,P0=P0,
                       ensemble_size = ensemble_size,Y=Y,vector_dynamics=vector_dynamics)
        
            
    def gen_conj_pair_list(self):
        """
        Generates a list of indices for which eigenvalues of the DMD model are complex conjugate pairs
            
        Returns
        -------
        conj_pairs : list
            List of paired indices, that reflect which eigenvalues if any are complex conjugate pairs
        """
        
        tol = 10**-8
        conj_pairs = []
        ignore = []
        for i,eig in enumerate(self.DMD.E):
            if abs(np.imag(eig)) < tol:
                ignore.append(i)
            if i not in ignore:
                match = np.where(abs(self.DMD.E - np.conj(eig)) < tol)
                if match[0].size != 0:
                    conj_pairs.append([i,match[0][0]])
                    ignore.append(match[0][0])
        return conj_pairs

    
    def eigs_to_kf_param_state(self,eigs):
        """
        Converts eigenvalue list into all real elements, by writing any complex conjugate pairs in polar coordinates

        Parameters
        ----------
        eigs : numpy.array
            The DMD models eigenvalues at their current stage of filtering
            
        Returns
        -------
        kf_param_state: numpy.array
            The DMD models eigenvalues at their current stage of filtering, with all real elements as described above
        """
        
        #generate copy of eigs to use as kf param state
        kf_param_state = eigs.copy()
        #go through complex conj pairs and replace entries [x,y] with [r(x),theta(x)]
        for pair in self.conj_pair_list:
            kf_param_state[pair[0]] = self.polar_eigs[pair[0]][0]
            kf_param_state[pair[1]] = self.polar_eigs[pair[0]][1]
        return kf_param_state

    
    def kf_param_state_to_eigs(self,kf_param_state):
        """
        Reverses eigs_to_kf_param_state method, turning eigs set to polar form back into their complex coordinates

        Parameters
        ----------
        kf_param_state : numpy.array
            The DMD models eigenvalues at their current stage of filtering, with all complex conjugate pairs in polar form
            
        Returns
        -------
        eigs: numpy.array
            The DMD models eigenvalues at their current stage of filtering, in standard complex form
        """
        
        eigs = kf_param_state.copy().astype('complex128') 
        for pair in self.conj_pair_list:
            eigs[pair[0]] = kf_param_state[pair[0]]*np.exp(kf_param_state[pair[1]]*1j)
            eigs[pair[1]] = kf_param_state[pair[0]]*np.exp(-kf_param_state[pair[1]]*1j)
        return eigs

    
    #Forecasting methods for the DMDEnKF
    '''def predict_from_ensemble(self, n_steps):
        #readable stochastic prediction from the ensemble but very slow
        ensemble_predictions = []
        for ensemble in np.moveaxis(np.array(self.ensembles),0,2):
            ensemble_prediction = [self.stochastic_prediction(e[:self.state_state_size],
                                                              self.kf_param_state_to_eigs(e[-self.param_state_size:])
                                                              , n_steps) for e in  ensemble.T]
            ensemble_prediction = [np.expand_dims(e,1) for e in ensemble_prediction]
            ensemble_prediction = np.hstack(ensemble_prediction)
            ensemble_predictions.append(ensemble_prediction)
            final_predictions = np.mean(np.array(ensemble_predictions),axis=0)
        return final_predictions'''
    
    
    def apply_ensemble_eig(self,pred_member,eig_member):
        """
        Helper function used within fast predict method, that applies eigenvalue eig_member to ensemble state pred_member

        Parameters
        ----------
        pred_member : numpy.array
            An ensemble member, representing a state in the DMD system at a given timestep, in DMD coordinate space
        eig_member : numpy.array
            Eigenvalues associated to the above ensemble member, that are to be used to propagate it forwards in time
            
        Returns
        -------
        state: numpy.array
            The ensemble member with the relevant eigenvalue applied, still in DMD coordinate space
        """
        #helper function for within the fast predictor
        eigs = self.kf_param_state_to_eigs(eig_member)
        state = np.diag(eigs)@pred_member
        return state
    
    
    def fast_predict_from_ensemble(self, n_steps):
        """
        Propagate the ensembles forward n_steps, then take a state point estimate in an efficient manner
        Efficiency is obtained by generating noise in bulk, but not all at once to reduce chance of RAM overload

        Parameters
        ----------
        n_steps : int
            The number of timesteps to propagate your ensemble forward in time
            
        Returns
        -------
        final_pred: numpy.array
            A state point estimate prediction for each ensembles states value n_steps timesteps ahead
        """
        
        #fast stochastic ensemble prediction that is quicker by generating noise for each step all at once
        #if the multiprocessing loop to predict from the ensemble hangs without exception then check the step noise size,
        #as previously this was exceeding the available ram and causing the program to wait indefinitely
        predos = np.array(self.ensembles)
        original_shape = predos.shape
        predos = np.reshape(predos,[-1,self.x.shape[0]])
        for step in range(n_steps):
            step_noise = np.random.multivariate_normal(np.zeros(predos.shape[-1]),self.Q,predos.shape[0])
            eigs = predos[:,-self.param_state_size:]
            one_step_forward_ens = (self.DMD.DMD_inv_modes@(predos[:,:self.state_state_size].T)).T
            one_step_forward_ens = np.array([self.apply_ensemble_eig(pred_member,eig_member) for pred_member,eig_member in zip(one_step_forward_ens,eigs)])
            one_step_forward_ens = (self.DMD.DMD_modes@(one_step_forward_ens.T)).T
            predos = np.hstack([one_step_forward_ens,eigs])
            predos = predos + step_noise
        predos = np.reshape(predos,original_shape)
        final_preds = np.mean(predos,axis=1)
        return final_preds

    
def system_dynamics():
    #used to fix pickle related issue only, so not required for standard usage
    pass