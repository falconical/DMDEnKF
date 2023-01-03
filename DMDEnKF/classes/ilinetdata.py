#Import neccessary libraries for when placed in seperate module
import pandas as pd
import math
import numpy as np
import pickle
import datetime
from DMDEnKF.classes.DMDEnKF import DMDEnKF


'''

1) ILINetData - to be used to store data downloaded from ILINet website in a helpful format

2) DataTransform - to store transforms and inverse transforms for prepocessing the data
   
3) ModellingChoices - For specifying different options when designing the model, mainly used for experimentation earlier on in the project so contains some redundancy

'''


class ILINetData():
    """
    A class for storing and formatting ILINet data, made to mimic the formatting of previous RCGPData class 

    ...

    Attributes
    ----------
    total_data : dict
        Dictionary of the format {disease: total data}, where total data is an array,
        representing the percentage of GP visits that were ili related each week over the total population
    case_data : dict
        Dictionary of the format {disease: case data}, where case data is an array,
        representing the raw number of ili related GP visits each week split by age and region
    pop_data : dict
        Dictionary of the format {disease: pop data}, where pop data is an array,
        representing the raw number of total GP visits each week split by age and region
    infection_data : dict
        Dictionary of the format {disease: infection data}, where infection data is an array,
        representing the percentage of GP visits that were ili related each week split by age and region
        a.k.a case_data/pop_data
    x_values : list
        A list of the weeks that the above measurements were taken over
    """
        
        
    #class for storing ILINet data and formatting it in a similar way to RCGPData to reuse code
    def __init__(self,ILINetcsv_location):
        """
        Sets relevant attributes upon creation of an ILINet instance.

        Parameters
        ----------
        ILINetcsv_location : str
            file location of the downloaded ILINet data in csv form
        """
        
        #takes in ILINet csv and sets attributes for total, case, pop and infection data
        #this data is stored in dicts of the form total_data['Influenza-like illness']
        #to make it compatible with RCGPData code

        #Import and prepare american ILINet data
        a_df =pd.read_csv(ILINetcsv_location,skiprows=1)
        a_df = a_df[(a_df['YEAR'] >= 2003) & (a_df['YEAR'] <= 2018)]
        a_df = a_df[a_df['WEEK'] != 53]
        #combine future 25-49 and 50-64 columns to keep the data consis
        a_df['AGE 25-64'] = a_df['AGE 25-64'].fillna(0) + a_df['AGE 25-49'].fillna(0) + a_df['AGE 50-64'].fillna(0)

        #linearly interpolate between know values (using year 2000 as index 0)
        x = [0,520,(1040-52)]
        ys = np.array([[6.825,28.525,52.225,12.425],[6.525,27.425,53.025,13.025], [5.97,25.49,52.07,16.47]])
        dates_as_nums = [(y-2000)*52 + w-1 for y,w in zip(a_df['YEAR'].values,a_df['WEEK'].values)]
        interpolated_pop_percents = np.array([np.interp(dates_as_nums,x,y) for y in ys.T])

        #set a_df columns for each ages patient numbers
        a_df['AGE 0-4 PATIENTS'] = np.round(interpolated_pop_percents[0] * a_df['TOTAL PATIENTS'].values/100)
        a_df['AGE 5-24 PATIENTS'] = np.round(interpolated_pop_percents[1] * a_df['TOTAL PATIENTS'].values/100)
        a_df['AGE 25-64 PATIENTS'] = np.round(interpolated_pop_percents[2] * a_df['TOTAL PATIENTS'].values/100)
        a_df['AGE 65 PATIENTS'] = np.round(interpolated_pop_percents[3] * a_df['TOTAL PATIENTS'].values/100)

        #create total data array (as a percentage of total consults)
        total_data = []
        years = range(2003,2018+1)
        weeks = range(1,52+1)
        for year in years:
            for week in weeks:
                relevant_rows = a_df[(a_df['YEAR']== year) & (a_df['WEEK'] == week)]
                total_data.append(relevant_rows['ILITOTAL'].sum()/relevant_rows['TOTAL PATIENTS'].sum()*100)

        #create case data array for each region/age strata       
        case_data = []
        regions = ['Region 1', 'Region 2', 'Region 3', 'Region 4', 'Region 5',
                   'Region 6', 'Region 7', 'Region 8', 'Region 9', 'Region 10']
        ages = ['AGE 0-4', 'AGE 5-24', 'AGE 25-64', 'AGE 65']
        for region in regions:
            relevant_rows = a_df[(a_df['REGION']== region)]
            case_data.append(np.array([relevant_rows[age] for age in ages]))
        case_data = np.array(case_data)

        #create total patient number for each age strata
        pop_data = []
        regions = ['Region 1', 'Region 2', 'Region 3', 'Region 4', 'Region 5',
                   'Region 6', 'Region 7', 'Region 8', 'Region 9', 'Region 10']
        age_patients = ['AGE 0-4 PATIENTS', 'AGE 5-24 PATIENTS', 'AGE 25-64 PATIENTS', 'AGE 65 PATIENTS']
        for region in regions:
            relevant_rows = a_df[(a_df['REGION']== region)]
            pop_data.append(np.array([relevant_rows[age] for age in age_patients]))
        pop_data = np.array(pop_data)

        #create percentage of total consults that were for ILI for each region/age strata
        infection_data = (case_data/pop_data)*100
        
        start_year = 2003
        x_weeks = [w for w in range(1,len(total_data)+1)]
        x_values = [datetime.datetime(start_year + math.floor((w-1)/52),1,4) + datetime.timedelta(weeks=(w-1)%52) for w in x_weeks]
        
        #set attributes in a way compatible with functions that use rcgp data
        self.total_data = {'Influenza-like illness': total_data}
        self.case_data = {'Influenza-like illness': case_data}
        self.pop_data = {'Influenza-like illness': pop_data}
        self.infection_data = {'Influenza-like illness': infection_data}
        self.x_values = x_values
        
        
        
        
class DataTransform():
    """
    A base class for data transforms that can be applied to data within the ModellingChoices class

    ...
    
    Methods
    -------
    apply_the_transformation(self)
        raises error if this method has not been implemented
    invert_the_transformation(self)
        raises error if this method has not been implemented
    """
    
    
    def apply_the_transformation(self):
        #takes in numpy matrix and applies transform assuming samples are arranged in columns
        raise AttributeError('You have not defined how to apply the transformation')
    
    
    def invert_the_transformation(self):
        #takes in numpy matrix and inverts transform assuming samples are arranged in columns
        raise AttributeError('You have not defined how to invert the transformation')
        
        
        
        
class LogTransform(DataTransform):
    """
    A DataTransform class that implements a transform of the form log(x + c) to data x

    ...

    Attributes
    ----------
    constant : int/float
        The small c within transform log(x + c), added to avoid taking log(0) on non-negative data
    log_means : numpy.array
        The mean vector of the log transformed data
    
    Methods
    -------
    apply_the_transformation(self)
        takes x -> log(x + c), then subtracts the transformed data's mean
    invert_the_transformation(self)
        sends log(x + c) - log_means -> x back to the original data
    """
    
    
    def __init__(self,constant=0):
        """
        Sets relevant attributes upon creation of the LogTransform instance.

        Parameters
        ----------
        constant : int/float
            The small c within transform log(x + c), added to avoid taking log(0) on non-negative data
        """
        
        self.constant = constant
        
        
    def apply_the_transformation(self,data):
        """
        Takes x -> log(x + c), then subtracts the transformed data's mean

        Parameters
        ----------
        data : numpy.array
            The data to apply the transform to, arranged such that the rows will have their means subtracted
            
        Returns
        -------
        transformed_data: numpy.array
            log(x + c) - log_means, where x represents the original data
        """
        
        transformed_data = np.log(data+self.constant)
        if not hasattr(self, 'log_means'):
            self.log_means = np.expand_dims(np.mean(transformed_data,axis=1),1)
        transformed_data = transformed_data - self.log_means
        return transformed_data
    
    
    def invert_the_transformation(self,data):
        """
        Takes log(x + c) - log_means, and sends it back to its original form x

        Parameters
        ----------
        data : numpy.array
            The transformed data to apply the inverse transform to
            
        Returns
        -------
        data: numpy.array
            data x in it's pre-transformed form
        """
        
        data = data + self.log_means
        data = np.exp(data) - self.constant
        return data
    
    
    
    
class ModellingChoices():
    """
    A class that implements various modelling choices to the data, for example:
    when to split for the spin-up, whether to include pandemic years/non-flu season weeks, data transforms etc

    ...

    Attributes
    ----------
    data_transform : ilinetdata.DataTransform
        DataTransform class that has implemented methods apply_the_transformation and invert_the_transformation (default None)
    pandemic_years : list
        List of years to exclude in modelling, typically due to seasonally atypical pandemic behaviour (default None)
    flu_season : list
        List of weeks of the year to exclude, typically non-flu season weeks to analyse the flu season only (default None)
    rcgpdata : ilinet.ILINetData
        Ili data object, containing various attributes pertaining to weekly ili consultation numbers
    disease : str
        Disease to investigate, included as other data sources contain non-ili data also
    datetime_to_split : datetime.datetime
        The date at which to split into spin-up data (before), and filtering data (after)
    total_data : numpy.array
        Squeezed total data from rcgpdata attribute into shape [number of weeks] only
    reshaped_data_dim : int
        State dimension of reformatted data
    stratified_data : numpy.array
        Reformatted stratified data from rcgpdata attribute into shape [state dimension, number of weeks]
    pop_data : numpy.array
        Reformatted pop data from rcgpdata attribute into shape [state dimension, number of weeks]
    x_values : numpy.array
        Array of rcgpdata attributes x_values list
    hankel_dim : int
        Number of time delays used to create delay-embedded coordinates
    x_len : int
        Length of the state measurements being fitted
    e_len : int
        Number of parameters (eigenvalues) being fitted
    dmdenkf_container : dict
        Container object that stores fitted DMDEnKF's with other relevant info,
        somewhat depreceated due to rigidity causing issues when used wit Hankel-DMDEnKF
    
    Methods
    -------
    set_variables(self,rcgpdata,disease,datetime_to_split,**kwargs)
        Called by prepare_data method, sets various variables to attributes
    prepare_data(self,rcgpdata,disease,datetime_to_split,**kwargs)
        Formats data under the specified modelling choices
    spinup_filter_split(self,datetime_to_split)
        Splits the data for spin-up and filtering at the specified date
    transform_the_data(self)
        Wrapper that applys the given data_transform
    inverse_transform_the_data(self,data)
        Wrapper that apply the given inverse data_transform
    stratified_to_total_data(self,data,dates)
        Uses an average weighted by population composition to generate total population case estimates from stratified ones
    run_filters_over_section(self,data,f,sys_obs_rat, sys_eig_rat, sys_cov_rat)
        Used within filter_all_sections method to apply DMDEnKF filtering to data over a given period
    post_filter_processing(self,dates,dmdenkf,eig_carry)
        Used within filter_all_sections method to store filters from each period in attributes
    filter_all_sections(self,f,eig_carry=True,sys_obs_rat=3.5,sys_eig_rat=1/400,sys_cov_rat=1)
        Applys DMDEnKF filtering over all periods of the data
    past_moving_average(data,n)
        Calculates the n year moving average for the data
    """
    
    
    def __init__(self,data_transform=None, pandemic_years=None, flu_season=None):
        """
        Sets relevant attributes upon creation of the ModellingChoices instance.

        Parameters
        ----------
        data_transform : ilinetdata.DataTransform
            DataTransform class that has implemented methods apply_the_transformation and invert_the_transformation,
            (default None)
        pandemic_years : list
            List of years to exclude in modelling, typically due to seasonally atypical pandemic behaviour, (default None)
        flu_season : list
            List of weeks of the year to exclude, typically non-flu season weeks to analyse the flu season only, (default None)
        """
            
        #Inputs: data_transform = string that is handled by if statements
                #pandemic years = list of all years that should be removed from the dataset
                #flu_season = list of the weeks that constitute the flu season and should be kept
                
        self.data_transform=data_transform
        self.pandemic_years = pandemic_years
        self.flu_season = flu_season
        
        
    def set_variables(self,rcgpdata,disease,datetime_to_split,**kwargs):
        """
        Called by prepare_data method, sets various variables to attributes.

        Parameters
        ----------
        rcgpdata : ilinet.ILINetData
            Ili data object, containing various attributes pertaining to weekly ili consultation numbers
        disease : str
            Disease to investigate, included as other data sources contain non-ili data also
        datetime_to_split : datetime.datetime
            The date at which to split into spin-up data (before), and filtering data (after)
        kwargs : optional
        hankel_dim : int
        Number of time delays used to create delay-embedded coordinates
        """
        
        #this function is called at the start of prepare data to make overwriting classes easier
        #provides the option of adding hankel dimensions onto data
        self.rcgpdata = rcgpdata
        self.disease = disease
        self.datetime_to_split = datetime_to_split
        self.total_data = np.expand_dims(rcgpdata.total_data[disease],0)
        self.reshaped_data_dim = np.prod(rcgpdata.infection_data[disease].shape[:-1])
        self.stratified_data = np.reshape(rcgpdata.infection_data[disease],[self.reshaped_data_dim,-1])
        self.pop_data = np.reshape(rcgpdata.pop_data[disease],[self.reshaped_data_dim,-1])
        self.x_values = np.array(rcgpdata.x_values)
        #stack via hankel dim and adjust total_data and dates accordingly
        if "hankel_dim" in kwargs:
            self.hankel_dim = kwargs['hankel_dim']
            hankel_list = [self.stratified_data[:,i:-self.hankel_dim + i + 1] if i+1 != self.hankel_dim else self.stratified_data[:,i:] for i in range(self.hankel_dim)]
            self.stratified_data = np.vstack(list(reversed(hankel_list)))
            self.total_data = self.total_data[:,(self.hankel_dim-1):]
            self.x_values = self.x_values[(self.hankel_dim-1):]
    
    
    def prepare_data(self,rcgpdata,disease,datetime_to_split,**kwargs):
        """
        Formats data under the specified modelling choices.

        Parameters
        ----------
        rcgpdata : ilinet.ILINetData
            Ili data object, containing various attributes pertaining to weekly ili consultation numbers
        disease : str
            Disease to investigate, included as other data sources contain non-ili data also
        datetime_to_split : datetime.datetime
            The date at which to split into spin-up data (before), and filtering data (after)
        kwargs : optional
        hankel_dim : int
        Number of time delays used to create delay-embedded coordinates
        """
        
        self.set_variables(rcgpdata,disease,datetime_to_split,**kwargs)        
        
        #REMOVE PANDEMIC YEARS IF SPECIFIED
        if self.pandemic_years is not None:
            split_points = []
            for year in self.pandemic_years:
                #make list of all points that splits will be required
                years_first_index = [date.year for date in self.x_values].index(year)
                years_last_index = years_first_index + 52
                split_points.extend([years_first_index,years_last_index])
            #split the arrays into chunks at the given split points
            split_points = sorted(set(split_points))
            self.x_values = np.split(self.x_values,split_points)
            self.total_data = np.split(self.total_data,split_points,axis=1)
            self.stratified_data = np.split(self.stratified_data,split_points,axis=1)
            #determine sections that can be discarded
            sections_to_discard = []
            for i,section in enumerate(self.x_values):
                for year in self.pandemic_years:
                    if year in [s.year for s in section]:
                        sections_to_discard.append(i)
                        break
            #discard those sections
            self.x_values = [section for i,section in enumerate(self.x_values) if i not in sections_to_discard]
            self.total_data = [section for i,section in enumerate(self.total_data) if i not in sections_to_discard]
            self.stratified_data = [section for i,section in enumerate(self.stratified_data) if i not in sections_to_discard]
            
        else:
            #to keep formatting consistent if pandemic years not removed, enclose relevant variables in a list
            self.x_values = [self.x_values]
            self.total_data = [self.total_data]
            self.stratified_data = [self.stratified_data]
        
        
        
        #REMOVE NON FLU SEASON WEEKS IF SPECIFIED
        if self.flu_season is not None:
            #work section by section
            for section_index,section in enumerate(self.x_values):
                #determine at which points a split is required
                split_points = []
                for i, section_week in enumerate([w.isocalendar()[1] for w in section]):
                    if section_week not in self.flu_season:
                        split_points.extend([i,i+1])
                #split the arrays into chunks at the given split points
                split_points = sorted(set(split_points))
                self.x_values[section_index] = np.split(self.x_values[section_index],split_points)
                self.total_data[section_index] = np.split(self.total_data[section_index],split_points,axis=1)
                self.stratified_data[section_index] = np.split(self.stratified_data[section_index],split_points,axis=1)
                #determine sections that can be discarded
                parts_to_discard = []
                for part_index,part in enumerate(self.x_values[section_index]):
                    if len(part) == 0:
                        parts_to_discard.append(part_index)
                    elif part[0].isocalendar()[1] not in self.flu_season:
                            parts_to_discard.append(part_index)
                #discard given parts
                self.x_values[section_index] = [part for i,part in enumerate(self.x_values[section_index]) if i not in parts_to_discard]
                self.total_data[section_index] = [part for i,part in enumerate(self.total_data[section_index]) if i not in parts_to_discard]
                self.stratified_data[section_index] = [part for i,part in enumerate(self.stratified_data[section_index]) if i not in parts_to_discard]
            #de-nest the results so each section does not include further subsections
            self.x_values = [part for section in self.x_values for part in section]
            self.total_data = [part for section in self.total_data for part in section]
            self.stratified_data = [part for section in self.stratified_data for part in section]
        
        
        #APPLY THE DATA TRANSFORM IF SPECIFIED
        self.transformed_stratified_data = self.transform_the_data()
        
        self.spinup_filter_split(self.datetime_to_split)
        
        
    def spinup_filter_split(self,datetime_to_split):
        """
        Called withing prepare_data method, splits the data for spin-up and filtering at the specified date.

        Parameters
        ----------
        datetime_to_split : datetime.datetime
        The date at which to split into spin-up data (before), and filtering data (after)
        """
        
        #split data into seperate sections, one to spin up DMD on and the other to use for filtering
        spinup_bools = [np.where(section<datetime_to_split,True,False) for section in self.x_values]
        
        #define a splitter to run over dates and all 3 lists of data arrays
        def splitter(sections,spinup_bools):
            spinup_part = []
            filter_part = []
            for section,spinup_bool in zip(sections,spinup_bools):
                spinup_part.append(np.array([a for a,b in zip(section.T,spinup_bool) if b]).T)
                filter_part.append(np.array([a for a,b in zip(section.T,spinup_bool) if not b]).T)
            spinup_part = [s for s in spinup_part if s.size != 0]
            filter_part = [s for s in filter_part if s.size != 0]
            return spinup_part, filter_part
        
        #run splitter over all required lists of arrays, filtering section keep their name and spinup labelled
        self.spinup_x_values, self.x_values = splitter(self.x_values,spinup_bools)
        self.spinup_total_data, self.total_data = splitter(self.total_data,spinup_bools)
        self.spinup_stratified_data, self.stratified_data = splitter(self.stratified_data,spinup_bools)
        self.spinup_transformed_stratified_data, self.transformed_stratified_data = splitter(self.transformed_stratified_data,spinup_bools)

    
    def transform_the_data(self):
        """
        Applies the data_transform provided to the ModellingChoices instance
            
        Returns
        -------
        transformed_stratified_data: numpy.array
            The stratified data with provided data transform applied
        """
        
        #check if there is any transformation to apply to the data
        if self.data_transform is None:
            transformed_stratified_data = self.stratified_data
        else:
            #make a note of the points to resplit arrays after transforamtion
            points_to_resplit = np.cumsum([x.shape[0] for x in self.x_values])[:-1]
            stratified_stacked = np.hstack(self.stratified_data)
            #transform the stacked array using data_transform classes method
            stratified_stacked = self.data_transform.apply_the_transformation(stratified_stacked)
            transformed_stratified_data = np.hsplit(stratified_stacked,points_to_resplit)
        return transformed_stratified_data
    
    
    def inverse_transform_the_data(self,data):
        """
        Inverts the transform applied by data_transform provided to the ModellingChoices instance

        Parameters
        ----------
        data : numpy.array
            Transformed (stratified) data
            
        Returns
        -------
        data: numpy.array
            Data as it was before the transform was applied
        """
        
        #check if there is any inverse transformation and apply to the data
        if self.data_transform is not None:
            data = self.data_transform.invert_the_transformation(data)
        #check if hankel stacking occurred and if so take only most recent week
        if hasattr(self,'hankel_dim'):
            data = np.vsplit(data,self.hankel_dim)[0]            
        return data
    
    
    def stratified_to_total_data(self,data,dates):
        """
        Uses an average weighted by population composition to generate total population case estimates from stratified ones

        Parameters
        ----------
        data : numpy.array
            The stratified (untransformed) data to be converted to a total estimate
        dates : numpy.array
            Dates for each week of the data, so that the correct population composition can be used in the weighted average
            
        Returns
        -------
        total_data: list
            A list of the total estimates for each week of the provided stratified data
        """
        
        #takes in a stratified section with relevant dates and produces a weighted total
        population_data = self.pop_data
        try:
            population_indexs = [self.rcgpdata.x_values.index(d) for d in dates]
        except:
            ValueError('You entered dates that there does not exist population data for')
        population_data = population_data[:,population_indexs]
        return [d@(pop/np.sum(pop)) for d,pop in zip(data.T,population_data.T)]
    
    
    def run_filters_over_section(self,data,f,sys_obs_rat, sys_eig_rat, sys_cov_rat):
        """
        Used within filter_all_sections method to apply DMDEnKF filtering to data over a given period

        Parameters
        ----------
        data : numpy.array
            Data to be filtered
        f : DMDEnKF.DMD
            Spin-up DMD model
        sys_obs_rat : float
            Constant to govern the size of the observation covariance matrix's uncertainty
        sys_eig_rat : float
            Constant to govern the size of the system covariance matrix's eigenvalue uncertainty
        sys_cov_rat : float
            Constant to govern the size of the system covariance matrix's state uncertainty
            
        Returns
        -------
        dmdenkf: DMDEnKF.DMDEnKF
            DMDEnKF instance that has been fitted using DMD model f and filtered over provided data
        """
        
        #General parameters taken as inputs as likely to edit the way these are determined
        system_cov_const = sys_cov_rat**2
        obs_cov_const = sys_obs_rat**2
        eig_cov_const = sys_eig_rat**2
        ensemble_size = 200

        #Shared variables
        Y=data[:,1:]
        self.x_len = f.X.shape[0]
        self.e_len = f.E.shape[0]
        P0 = np.real(np.cov(f.Y-(f.DMD_modes@np.diag(f.E)@np.linalg.pinv(f.DMD_modes)@f.X)))
        #DMDEnKF
        observation_operator = np.hstack((np.identity(self.x_len),np.zeros((self.x_len,self.e_len))))

        system_cov = np.diag([system_cov_const]*self.x_len + [eig_cov_const]*self.e_len)

        observation_cov = obs_cov_const * np.identity(self.x_len)

        x0 = data[:,0]
        P0 = np.block([[P0,np.zeros([self.x_len,self.e_len])],[np.zeros([self.e_len,self.x_len]),np.diag([eig_cov_const]*self.e_len)]])
        eig0 = None
        if hasattr(self,'eig0'):
            eig0 = self.eig0

        dmdenkf = DMDEnKF(observation_operator=observation_operator, system_cov=system_cov,
                              observation_cov=observation_cov,x0=x0,P0=P0,eig0=eig0,DMD=f,ensemble_size=ensemble_size,Y=None)

        dmdenkf.fit(Y)

        return dmdenkf
    
    
    #After discontinuous time section has been filtered, we post process to extra
    def post_filter_processing(self,dates,dmdenkf,eig_carry):
        """
        Used within filter_all_sections method to store filters from each period in attributes.

        Parameters
        ----------
        dates : numpy.array
            Dates for the weeks that the filter has been applied over
        dmdenkf : DMDEnKF.DMDEnKF
            Fitted DMDEnKF to store key attributes of
        eig_carry : bool
            Whether to carry the latest eigenvalue estimate from the previous period to use as the start for the next period,
            (default True)
        """
        
        #DMDEnKF POST PROCESSING
        #stack and invert the transformation to get stratified data and eigenvalues to initialise next filter
        dmdenkf_Xarray = np.hstack([np.expand_dims(x,1) for x in dmdenkf.X])
        states = dmdenkf_Xarray[:self.x_len,:]
        params = dmdenkf_Xarray[self.x_len:,:]
        eigenvalue_estimates = np.apply_along_axis(dmdenkf.kf_param_state_to_eigs,0,params)
        self.dmdenkf_container['eigenvalue estimates'].append(eigenvalue_estimates)
        if eig_carry:
            self.eig0 = eigenvalue_estimates[:,-1]
        stratified_estimates = self.inverse_transform_the_data(states)
        self.dmdenkf_container['stratified estimates'].append(stratified_estimates)
        #use dates to recreate total data from stratified
        total_estimates = self.stratified_to_total_data(stratified_estimates,dates)
        self.dmdenkf_container['total estimates'].append(total_estimates)
    
    
    #FILTER PROCESS OVER ALL SECTIONS
    def filter_all_sections(self,f,eig_carry=True,sys_obs_rat=3.5,sys_eig_rat=1/400,sys_cov_rat=1):
        """
        Applys DMDEnKF filtering over all periods of the data

        Parameters
        ----------
        f : DMDEnKF.DMD
            Spin-up DMD model
        eig_carry : bool
            Whether to carry the latest eigenvalue estimate from the previous period to use as the start for the next period,
            (default True)
        sys_obs_rat : float
            Constant to govern the size of the observation covariance matrix's uncertainty
        sys_eig_rat : float
            Constant to govern the size of the system covariance matrix's eigenvalue uncertainty
        sys_cov_rat : float
            Constant to govern the size of the system covariance matrix's state uncertainty
            
        Returns
        -------
        dmdenkf.container : dict
            Container object that stores fitted DMDEnKF's with other relevant info,
            somewhat depreceated due to rigidity causing issues when used wit Hankel-DMDEnKF
        """
        
        self.dmdenkf_container = {'full_filters':[],'stratified estimates':[],'total estimates':[],'eigenvalue estimates':[]}
        for dates,data in zip(self.x_values,self.transformed_stratified_data):
        #for each section grab data with associated dates
            #apply filters
            dmdenkf = self.run_filters_over_section(data,f,sys_obs_rat,sys_eig_rat,sys_cov_rat)
            self.dmdenkf_container['full_filters'].append(dmdenkf)
            #apply post processing to store stratified/total data and eigs in dmdenkf containers
            self.post_filter_processing(dates,dmdenkf,eig_carry)
            
        return self.dmdenkf_container
    
    
    @staticmethod
    def past_moving_average(data,n):
        """
        Calculates the n year moving average for the data

        Parameters
        ----------
        data : numpy.array
            The data to calculate the n year moving average over, assumed to be weekly and 1-dimensional
        n : int
            The number of previous years to include in the moving average
            
        Returns
        -------
        MA : list
            The n year moving average at each possible week, calculated over the provided data
        """
        
        MA = []
        for i in range(len(data)):
            if i < n*52:
                MA.append(np.nan)
            else:
                n_prev_years = [data[i-52*(year+1)] for year in range(n)]
                MA.append(sum(n_prev_years)/n)
        return MA