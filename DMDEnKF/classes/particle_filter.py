import numpy as np



'''

This file contains a simple Particle filter class to be used in the DMD particle filter/DMDEnKF synthetic comparisons

'''


#Particle filter implementation for comparison of the DMDEnKF againt the DMD Particle Filter
class ParticleFilter():
    """
    A class used to represent the Particle Filter data assimilation technique.

    ...

    Attributes
    ----------
    particles : Iterable
        The current set of particles represented within the filter of shape, [num_particles, state dimension]
    particles_list : list
        A record of the particles in the filter at all assimilated timesteps thus far
    model : Callable
        Function that applies the model of the system dynamics to a particle, to forecast it's state at the next timestep
    likelihood : Callable
        Function that calculates the likelihood of a particle being in it's proposed state, based on a given measurement
    particle_num : int
        Number of particles to be used in the filter
    X : list
        Record of the particle's point estimate of the state (mean) for all assimilated timesteps so far
    weights : list
        Relative weights of each particle based on their likelihood values
    min_effective_sample_ratio : float
        Threshold that once the effective sample size falls below to resample
    mode : str
        Governs how the model function is applied to particles, "standard" (default) or "vector" receives all particles
    

    Methods
    -------
    fit(self,data)
        takes in all new data and appplies particle filter point by point 
    pf(self,particles,data_point)
        Used in fit method, applies particle filter to a single piece of new data
    """
    
    
    def __init__(self,prior,model,likelihood,particle_num,min_effective_sample_ratio=None,mode='standard'):
        """
        Sets relevant attributes when particle filter is initialised.

        Parameters
        ----------
        prior : Iterable
            The initial set of particles represented within the filter of shape, [num_particles, state dimension]
        model : Callable
            Function that applies the model of the system dynamics to a particle, to forecast it's state at the next timestep
        likelihood : Callable
            Function that calculates the likelihood of a particle being in it's proposed state, based on a given measurement
        particle_num : int
            Number of particles to be used in the filter
        min_effective_sample_ratio : float
            Threshold that once the effective sample size falls below to resample
        mode : str
            Governs how the model function is applied to particles, "standard" (default) or "vector" receives all particles
        """
        
        #model and liklihood are funcs that return probability density of a point
        self.particles = prior
        self.particles_list = [prior]
        self.model = model
        self.likelihood = likelihood
        self.particle_num = particle_num
        self.X = [np.mean(prior,axis=0)]
        self.weights = [1/self.particle_num]*self.particle_num
        self.min_effective_sample_ratio = min_effective_sample_ratio
        self.mode = mode

        
    #loop through the data and apply particle filter at each step 
    def fit(self,data):
        """
        Takes in all new data and appplies particle filter point by point, storing the results

        Parameters
        ----------
        data : numpy.array
            Iterable of data measurements to be looped through and assimilated in chronological order,
            shape [Number of measurements, state dimension]
        """
        
        for data_point in data:
            self.pf(self.particles,data_point)
            #no longer store full particle list to save memory
            #self.particles_list.append(self.particles)
            self.X.append(np.average(self.particles,axis=0,weights=self.weights))

            
    #basic particle filter implementation that is readable but slow
    def pf(self,particles,data_point):
        """
        Used within fit method to assimilate the new data_point received into the current particles

        Parameters
        ----------
        particles : Iterable
            The current set of particles represented within the filter of shape, [num_particles, state dimension]
        data_point : numpy.array
            The newly received data point, to assimilate into the current set of particles using the likelihood function
        """
        
        if self.mode == 'standard':
            #all operations performed one particle at a time (simplest intuitively)
            #apply model to particles to propagate prior
            particles = [self.model(p) for p in particles]
            #use measurement (data_point) to update weights of each particle
            likelihood_weights = [self.likelihood(p,data_point) for p in particles]
        elif self.mode == 'vector':
            #both model and likelihood take in all particles, so can be designed in a vectorised manner) - MUCH QUICKER!
            particles = self.model(particles)
            #use measurement (data_point) to update weights of each particle
            likelihood_weights = self.likelihood(particles,data_point)
        new_weights = [l*w for l,w in zip(likelihood_weights,self.weights)]
        new_weights = new_weights/np.sum(new_weights)
        #if effective sample size is set and too low then resample particles
        effective_sample_size = 1/(np.sum([w**2 for w in new_weights]))
        if (self.min_effective_sample_ratio is not None) and (effective_sample_size < self.min_effective_sample_ratio*self.particle_num):
            #weighted sample from particles to generate new particles that have equal weight again (stops degeneration)
            #particles = np.random.choice(particles,size = self.particle_num,p=weights)
            particle_indexs = np.random.choice(range(len(particles)),size = self.particle_num,p=new_weights)
            particles = [particles[i] for i in particle_indexs]
            #and the weights are reset to be uniform
            new_weights = [1/self.particle_num]*self.particle_num
        #particles and new weights are then taken to be the prior for the next step, ready to be propagated again
        self.particles = particles
        self.weights = new_weights/np.sum(new_weights)
