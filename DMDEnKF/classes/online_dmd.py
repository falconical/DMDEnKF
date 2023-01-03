import numpy as np


'''

Within this file lies the Zhang and Rowley's code for the Online DMD class:

Online DMD is an iteraitve DMD variant we compare against in our paper, hence the code was imported to understand it, and potentially tweak it to be applied to our use case, however no changes were subsequently made to their class.

This code is available from the code supplement linked by their paper:

        Hao Zhang, Clarence W. Rowley, Eric A. Deem, and Louis N. Cattafesta,
        "Online Dynamic Mode Decomposition for Time-varying Systems,"
        arXiv preprint arXiv:1707.02876, 2017.

'''


class OnlineDMD:
    """OnlineDMD is a class that implements online dynamic mode decomposition
    The time complexity (multiply–add operation for one iteration) is O(4n^2), 
    and space complexity is O(2n^2), where n is the state dimension.
    
    Algorithm description:
        At time step k, define two matrix X(k) = [x(1),x(2),...,x(k)], 
        Y(k) = [y(1),y(2),...,y(k)], that contain all the past snapshot pairs, 
        where x(k), y(k) are the n dimensional state vector, y(k) = f(x(k)) is 
        the image of x(k), f() is the dynamics. 
        Here, if the (discrete-time) dynamics are given by z(k) = f(z(k-1)), 
        then x(k), y(k) should be measurements correponding to consecutive 
        states z(k-1) and z(k).
        We would like to update the DMD matrix Ak = Yk*pinv(Xk) recursively 
        by efficient rank-1 updating online DMD algrithm.
        An exponential weighting factor can be used to place more weight on
        recent data.
    
    Usage:
        odmd = OnlineDMD(n,weighting)
        odmd.initialize(Xq,Yq)
        odmd.initilizeghost()
        odmd.update(x,y)
        evals, modes = odmd.computemodes()
            
    properties:
        n: state dimension
        weighting: weighting factor in (0,1]
        timestep: number of snapshot pairs processed (i.e., current time step)
        A: DMD matrix, size n by n
        P: Matrix that contains information about past snapshots, size n by n
    methods:
        initialize(Xq, Yq), initialize online DMD algorithm with first q 
                            snapshot pairs stored in (Xq, Yq)
        initializeghost(), initialize online DMD algorithm with epsilon small 
                            (1e-15) ghost snapshot pairs before t=0
        update(x,y), update DMD computation when new snapshot pair (x,y) 
                            becomes available
        computemodes(), compute and return DMD eigenvalues and DMD modes
    
    Authors:
        Hao Zhang
        Clarence W. Rowley
        
    References:
        Hao Zhang, Clarence W. Rowley, Eric A. Deem, and Louis N. Cattafesta,
        "Online Dynamic Mode Decomposition for Time-varying Systems,"
        arXiv preprint arXiv:1707.02876, 2017.
    
    Date created: April 2017
    
    To import the OnlineDMD class, add import online at head of Python scripts.
    To look up this documentation, type help(online.OnlineDMD) or 
    online.OnlineDMD?
    """
    def __init__(self, n=0, weighting=1):
        """
        Creat an object for online DMD
        Usage: odmd = OnlineDMD(n,weighting)
            """
        self.n = n
        self.weighting = weighting
        self.timestep = 0
        self.A = np.zeros([n,n])
        self.P = np.zeros([n,n])

    def initialize(self, Xq, Yq):
        """Initialize online DMD with first q snapshot pairs stored in (Xq, Yq)
        Usage: odmd.initialize(Xq,Yq)
        """
        q = len(Xq[0,:])
        Xqhat, Yqhat = np.zeros(Xq.shape), np.zeros(Yq.shape)
        if self.timestep == 0 and np.linalg.matrix_rank(Xq) == self.n:
            weight = np.sqrt(self.weighting)**range(q-1,-1,-1)
            Xqhat, Yqhat = weight*Xq, weight*Yq
            self.A = Yqhat.dot(np.linalg.pinv(Xqhat))
            self.P = np.linalg.inv(Xqhat.dot(Xqhat.T))/self.weighting
            self.timestep += q
            
    def initializeghost(self):
        """Initialize online DMD with epsilon small (1e-15) ghost snapshot pairs 
        before t=0
        Usage: odmd.initilizeghost()
        """
        epsilon=1e-15
        alpha = 1.0/epsilon
        self.A = np.random.randn(self.n, self.n)
        self.P = alpha*np.identity(self.n)
        
    def update(self, x, y):
        """Update the DMD computation with a new pair of snapshots (x,y)
        Here, if the (discrete-time) dynamics are given by z(k) = f(z(k-1)), 
        then (x,y) should be measurements correponding to consecutive states 
        z(k-1) and z(k).
        Usage: odmd.update(x, y)
        """
        # compute P*x matrix vector product beforehand
        Px = self.P.dot(x)
        # compute gamma
        gamma = 1.0/(1 + x.T.dot(Px))
        # update A
        self.A += np.outer(gamma*(y-self.A.dot(x)),Px)
        # update P, group Px*Px' to ensure positive definite
        self.P = (self.P - gamma*np.outer(Px,Px))/self.weighting
        # ensure P is SPD by taking its symmetric part
        self.P = (self.P + self.P.T)/2
        # time step + 1
        self.timestep += 1

    def computemodes(self):
        """Compute and return DMD eigenvalues and DMD modes at current time step
        Usage: evals, modes = odmd.computemodes()
        """
        evals, modes = np.linalg.eig(self.A)
        return evals, modes