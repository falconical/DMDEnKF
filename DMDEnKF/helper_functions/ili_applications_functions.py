import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from KDEpy import FFTKDE
from scipy import integrate
import sys
from pympler import asizeof


'''

Helper functions for the ILI applications section of the DMDEnKF paper

'''


def fast_predict_from_ensemble(dmdenkf, n_steps):
    """
    Predicts n_steps ahead using the the full ensembles at all timesteps.
    This function is almost identical to the DMDEnKF's own ensemble predict method,
    however does not take the point estimate as in the ili case transformations are neccasary before this step.

    Parameters
    ----------
    dmdenkf : DMDEnKF.DMDEnKF
        The DMDEnKF object, whose ensemble we will propagate forward using it's system dynamics operator
    n_steps : int
        The number of timesteps forward to forecast
        
    Returns
    -------
    prediction : numpy.array
        All ensemble members, forecast n_steps forward in time
    """
        
    #prediction as in DMDEnKF class but without taking a mean (point estimate) before returning as full ensemble is required
    #fast stochastic ensemble prediction that is quicker by generating noise for each step all at once
    #if the multiprocessing loop to predict from the ensemble hangs without exception then check the step noise size,
    #as previously this was exceeding the available ram and causing the program to wait indefinitely
    predos = np.array(dmdenkf.ensembles)
    original_shape = predos.shape
    predos = np.reshape(predos,[-1,dmdenkf.x.shape[0]])
    for step in range(n_steps):
        eigs = predos[:,-dmdenkf.param_state_size:]
        one_step_forward_ens = ((dmdenkf.DMD.DMD_inv_modes@(predos[:,:dmdenkf.state_state_size].T)).T).astype('complex64')
        one_step_forward_ens = np.array([dmdenkf.apply_ensemble_eig(pred_member,eig_member) for pred_member,eig_member in zip(one_step_forward_ens,eigs)],dtype='complex64')
        one_step_forward_ens = ((dmdenkf.DMD.DMD_modes@(one_step_forward_ens.T)).T).astype('float32')
        predos = np.hstack([one_step_forward_ens,eigs])
        #FOUL HACK TO GET THINGS DONE QUICK IN VERY SPECIFIC SITCH FOR ILI DATA
        state_sd = np.sqrt(np.diag(dmdenkf.Q)[0])
        param_sd = np.sqrt(np.diag(dmdenkf.Q)[-1])
        state_noise = np.random.normal(0,state_sd,dmdenkf.state_state_size*predos.shape[0])
        state_noise = np.reshape(state_noise,[-1,dmdenkf.state_state_size])
        param_noise = np.random.normal(0,param_sd,dmdenkf.param_state_size*predos.shape[0])
        param_noise = np.reshape(param_noise,[-1,dmdenkf.param_state_size])
        noise = np.hstack([state_noise,param_noise]).astype('float32')
        predos = predos + noise
        '''predos = predos + np.random.multivariate_normal(np.zeros(predos.shape[-1],dtype='float32'),dmdenkf.Q,predos.shape[0]).astype('float32')'''
        
    predos = np.reshape(predos,original_shape)
    return predos


def generate_forecast_df(dmdenkf,m,n_steps_max):
    """
    Creates a dataframe with columns including some basic ili data, to be used to store future forecasts etc

    Parameters
    ----------
    dmdenkf : DMDEnKF.DMDEnKF
        DMDEnKF whose forecasts we wish to record in the dataframe
    m : ilinetdata.ModellingChoices
        ModellingChoices object with ili data stored in chosen manner, with datatransforms applied etc
    n_steps_max : int
        Maximum number of steps ahead we are interested in forecasting

    Returns
    -------
    df : pandas.DataFrame
        A dataframe that includes columns with relevant DMDEnKF forecasts and ili base data
    """
    
    #create a dataframe object with base data, where all subsequent results will be stored
    #then produce stratified and total ensemble and point estimates
    #takes in n_steps_max which is an int looped through the range of, or list to loop through for convenience
    #create initial df and relevant variables
    df = pd.DataFrame(index=m.rcgpdata.x_values)
    
    df['total data'] = m.rcgpdata.total_data[m.disease]
    df['stratified data'] = [x for x in np.reshape(m.rcgpdata.infection_data[m.disease],[m.reshaped_data_dim,-1]).T][-df.shape[0]:]
    df['5 year MA'] = m.past_moving_average(m.rcgpdata.total_data[m.disease],5)[-df.shape[0]:]
    dates = m.x_values[0]

    #for n step predictions in the range of chosen forecast horizons
    for n_steps in range(1,n_steps_max+1):
        print(n_steps)
        #change dates to represent prediction timesteps
        prediction_dates = (dates + datetime.timedelta(weeks=n_steps))[:-n_steps]
        #prediction dates are out by a day or 2 when going over the year barrier, so grab closest available date from dates
        prediction_dates = [d if d in dates else min(dates,key=lambda x: abs(x-d)) for d in prediction_dates]
        #predict the n step ahead state for each ensemble member
        ensemble_pred_states = fast_predict_from_ensemble(dmdenkf,n_steps)[:-n_steps,:,:-dmdenkf.param_state_size]
        #mean, invert transform and destratify to get point estimates
        average_traj = np.mean(ensemble_pred_states,axis=1)
        average_traj = m.inverse_transform_the_data(average_traj.T)
        df[f'{n_steps} strat pred'] = pd.Series({k:v for k,v in zip(prediction_dates,np.real(average_traj.T))})
        average_traj = m.stratified_to_total_data(average_traj,dates)
        df[f'{n_steps} total pred'] = pd.Series({k:v for k,v in zip(prediction_dates,np.real(average_traj))})    
        #invert the transform
        ensemble_pred_states = [m.inverse_transform_the_data(ensemble_state.T) for ensemble_state in np.swapaxes(ensemble_pred_states,0,1)]
        #record the ensemble with transform inverted in dataframe
        ensemble_strat_pred_states = np.swapaxes(np.array(np.real(ensemble_pred_states)),0,2)
        df[f'{n_steps} ensemble strat pred'] = pd.Series({k:v.T for k,v in zip(prediction_dates,ensemble_strat_pred_states)})
        #turn stratified ensemble preds into total ensemble preds
        ensemble_pred_states = np.array([m.stratified_to_total_data(ensemble_state,dates) for ensemble_state in ensemble_pred_states])
        #save these total ensemble preds in dataframe
        df[f'{n_steps} ensemble pred'] = pd.Series({k:v for k,v in zip(prediction_dates,np.real(ensemble_pred_states).T)}) 
    '''    #use the stratified ensemble preds to create a point estimate, by taking the mean then converting back to total data
        average_traj = np.mean(ensemble_strat_pred_states,axis=2)
        #save the stratified predictions
        df[f'{n_steps} strat pred'] = pd.Series({k:v for k,v in zip(prediction_dates,np.real(average_traj))})
        average_traj = m.stratified_to_total_data(average_traj.T,dates)
        #save as total pred in dataframe
        df[f'{n_steps} total pred'] = pd.Series({k:v for k,v in zip(prediction_dates,np.real(average_traj))})'''
    return df


#generate confidence intervals using an ensemble
def gen_ensemble_ci(ci,ensemble_predictions):
    """
    Takes an ensemble of predictions and returns the ensembles members at the top and bottom of the given confidence interval.

    Parameters
    ----------
    ci : float
        The confidence interval in decimal format that you want returned from the ensemble (e.g. 0.95 for 95% C.I.)
    ensemble_predictions : Iterable
        The ensemble from which the confidence interval should be drawn

    Returns
    -------
    top_ci : float
        The ensemble at the top of the given confidence interval's value
    bottom_ci : float
        The ensemble at the bottom of the given confidence interval's value
    mid : float
        The mean of the provided ensemble
       """
    
    top_ci = []
    bottom_ci = []
    mid = []
    for e in ensemble_predictions:
        if not isinstance(e,(list,np.ndarray)):
            if np.isnan(e):
                top_ci.append(e)
                bottom_ci.append(e)
                mid.append(e)
        else:
            sorted_preds = sorted(np.real(e))
            top_ci.append(sorted_preds[round((ci+(1-ci)/2)*len(sorted_preds))])
            mid.append(np.mean(sorted_preds))
            bottom_ci.append(sorted_preds[round(-(ci+(1-ci)/2)*len(sorted_preds))])
    return top_ci, bottom_ci, mid


def gen_eigenvalue_plots(FBeigs,ax=None):
    """
    Create a plot of the eigenvalues generated by the spin up DMD

    Parameters
    ----------
    FBeigs : numpy.array
        The eigenvalues produced by fitted DMD model in complex form
    ax : matplotlib.pyplot.Axes
        Axis upon which the eigenvalue graph should be plotted
       """
    
    #Plot eigenvalues generated by the spin up DMD
    #Eigenvalue Plots
    if ax is None:
        fig, ax = plt.subplots()
    plt.sca(ax)
    #add unit circle
    circle1 = plt.Circle((0, 0), 1, fill=False,color='black',alpha=0.6)
    ax.add_artist(circle1)
    plt.scatter(FBeigs.real,FBeigs.imag,color='tab:red')
    plt.axis('scaled')
    plt.xlim(-1.1,1.1)
    plt.ylim(-1.1,1.1)
    plt.xlabel('Real')
    plt.ylabel('Imaginary')
    tickos = [-1,-0.5,0,0.5,1]
    plt.xticks(tickos,tickos)
    plt.yticks(tickos,tickos)
    

def week_round(x_values,date):
    """
    A utility function that takes a date and sends it to the closest date available in a provide list

    Parameters
    ----------
    x_values : list
        The list of dates from which to find the closest match
    date : datetime.datetime
        The date which you want to find the closest match for

    Returns
    -------
    closest date : datetime.datetime
        The closest available date from the list provided
    """
    
    #utility function that returns closest date of the available dates provided
    distance_from_date = np.abs([mx - date for mx in x_values[0]])
    closest_date_ind = list(distance_from_date).index(np.min(distance_from_date))
    return x_values[0][closest_date_ind]


def gen_step_dmd_preds(date,tot_weeks_ahead,m,dmdenkf,df):
    """
    Used in wrapper_gen_step_dmd_preds, generates/retrieves the next forecast over a time period for naive DMD and the DMDEnKF

    Parameters
    ----------
    date : datetime.datetime
        Date to start forecasting from
    tot_weeks_ahead : int
        Number of future weeks to generate forecasts for
    m : illinetdata.ModellingChoices
        ModellingChoices object that contains fitted DMDEnKF model etc
    dmdenkf : DMDEnKF.DMDEnKF
        DMDEnKF for use of it's system dynamics model
    df : pandas.DataFrame
        Dataframe of the DMDEnKF forecasts and ili data

    Returns
    -------
    plot_dates: list
        List of dates that the state predictions are for
    step_preds : list
        The naive DMD model predictions
    dmdenkf_preds : list
        The DMDEnKF model's predictions
    """
    
    #Used in wrapper_gen_step_dmd_preds()
    #generates/retrieves the next forecasts over a period of time for naive DMD and DMDEnKF as new data becomes available
    trans_strat_ind = list(m.x_values[0]).index(date)
    step_start = m.transformed_stratified_data[0][:,trans_strat_ind]
    dmdenkf_init = m.dmdenkf_container['total estimates'][0][trans_strat_ind]
    dmdenkf_preds = [dmdenkf_init]
    step_init = df['total data'][date]
    step_preds = [step_init]
    for w in range(1,tot_weeks_ahead+1):
        dmdenkf_pred = (df[f'{w} total pred'][week_round(m.x_values,date + datetime.timedelta(weeks=w))])
        dmdenkf_preds.append(dmdenkf_pred)
        predict_operator = dmdenkf.DMD.DMD_modes@np.linalg.matrix_power(np.diag(dmdenkf.DMD.E),w)@dmdenkf.DMD.inv_DMD_modes
        step_pred = predict_operator@np.hstack(step_start)
        step_pred = m.inverse_transform_the_data(step_pred)
        step_pred = m.stratified_to_total_data(step_pred,[date])
        step_preds.append(step_pred[0])
    plot_dates = m.x_values[0][trans_strat_ind:trans_strat_ind+tot_weeks_ahead+1]
    return plot_dates, step_preds, dmdenkf_preds


def wrapper_gen_step_dmd_preds(date,tot_weeks_ahead,m,dmdenkf,df):
    """
    Generates/retrieves the next forecast over a time period for naive DMD and the DMDEnKF,
    as new data becomes available over that period

    Parameters
    ----------
    date : datetime.datetime
        Date to start forecasting from
    tot_weeks_ahead : int
        Number of future weeks to generate forecasts for
    m : illinetdata.ModellingChoices
        ModellingChoices object that contains fitted DMDEnKF model etc
    dmdenkf : DMDEnKF.DMDEnKF
        DMDEnKF for use of it's system dynamics model
    df : pandas.DataFrame
        Dataframe of the DMDEnKF forecasts and ili data

    Returns
    -------
    plot_dates: list
        List of dates that the state predictions are for
    super_step_preds : list
        The naive DMD model predictions, and how they are altered as each new piece of data emerges
    super_dmdenkf_preds : list
        The DMDEnKF model's predictions, and how they are altered as each new piece of data emerges
    """
    
    #generates/retrieves the next forecasts over a period of time for naive DMD and DMDEnKF as new data becomes available
    plot_dates, step_preds, dmdenkf_preds = gen_step_dmd_preds(date,tot_weeks_ahead,m,dmdenkf,df)
    super_dmdenkf_preds = [dmdenkf_preds]
    super_step_preds = [step_preds]
    for i in range(1,tot_weeks_ahead+1):
        w = tot_weeks_ahead -i
        d = week_round(m.x_values,date + datetime.timedelta(weeks=i))
        a,b,c = gen_step_dmd_preds(d,w,m,dmdenkf,df)
        dmdenkf_predos = [None]*i
        dmdenkf_predos.extend(c)
        super_dmdenkf_preds.append(dmdenkf_predos)
        step_predos = [None]*i
        step_predos.extend(b)
        super_step_preds.append(step_predos)
    return plot_dates, super_step_preds, super_dmdenkf_preds


def gen_no_kde_total_accuracy_prob(true_series,epreds_series):
    """
    Calculates an accuracy score, based on the number of ensemble members that are "close enough" to the true state value

    Parameters
    ----------
    true_series : list
        The list of baseline truth data
    epreds_series : list/nump.array
        The ensemble prediction at each corresponding timestep

    Returns
    -------
    output_accuracys : list
        The list of the fraction of ensembles that were close enough to the truth at each timestep
    """
    
    #func that creates an accuracy probability for the ensemble without using kde
    #and just via no. accurate ensemble members/total ensemble members
    output_accuracys = []
    for truth, epreds in zip(true_series.values,epreds_series.values):
        if not isinstance(epreds,(list,np.ndarray)):
            if np.isnan(epreds):
                output_acc = np.nan
        else:
            accuracte_preds = [epred for epred in epreds if (epred >=truth-0.5 and epred <=truth+0.5)]
            output_acc = len(accuracte_preds)/(np.array(epreds).shape[0])
        output_accuracys.append(output_acc)
    return output_accuracys


def calculate_final_probability_score(series):
    """
    Calculates a final probaility score as used in previous ILINet competitions

    Parameters
    ----------
    series : pandas.Series
        A series of accuracy scores over all weeks of the year,
        that should include at least some data from weeks >=40 or <=20 in the years from 2012 to 2018

    Returns
    -------
    final_prob_score : float
        Overall probability score
    seasonal_prob_score : numpy.array
        A probability score for each flu season
    """
    
    #calculate the final probability score from a series of accuracy probabilities
    #filter down to only include weeks of the year > 40 or less than 20 for flu season only
    series = series[(series.index.isocalendar().week>=40) | (series.index.isocalendar().week<=20)]
    #remove end half of 2011/2012 season as dont have full season available
    series = series[series.index>=datetime.datetime(year=2012,month=8,day=1)]
    #remove start half of 2018/2019 season as dont have full season available
    series = series[series.index<=datetime.datetime(year=2018,month=8,day=1)]
    accuracy_probs = series.values
    final_prob_score = np.exp(np.mean([np.log(p) if p>0 else -10 for p in accuracy_probs]))
    seasonal_prob_score = [np.exp(np.mean([np.log(p) if p>0 else -10 for p in acc_prob])) for acc_prob in np.split(accuracy_probs,int(2018-2012))]
    return final_prob_score, np.array(seasonal_prob_score)


def restrict_flu_season(series):
    """
    Restricts a series to only data during what we consider flu seasons in the relevant years

    Parameters
    ----------
    series : pandas.Series
        A series that should include at least some data from weeks >=40 or <=20 in the years from 2012 to 2018

    Returns
    -------
    series : pandas.Series
        A series that only includes data from weeks >=40 or <=20 in the years from 2012 to 2018
    """
    
    #restrict a series down to relevant flu seasons
    #filter down to only include weeks of the year > 40 or less than 20 for flu season only
    series = series[(series.index.isocalendar().week>=40) | (series.index.isocalendar().week<=20)]
    #remove end half of 2011/2012 season as dont have full season available
    series = series[series.index>=datetime.datetime(year=2012,month=8,day=1)]
    #remove start half of 2018/2019 season as dont have full season available
    series = series[series.index<=datetime.datetime(year=2018,month=8,day=1)]
    return series


def MSE_residuals(n_steps_max, na_free_forecasts):
    """
    Calculates the means squared errors over a forecast dataframe with any na's pre-removed

    Parameters
    ----------
    n_steps_max : int
        The maximum number of timesteps ahead to calculate the forecast MSE's for
    na_free_forecasts : pandas.DataFrame
        Forecast dataframe that has had any weeks with na entries removed

    Returns
    -------
    MSEs : dict
        A dictionairy containing the mean squared errors for each forecast horizon up to n_steps_max
    """
    
    #calculate the MSE residuals over the na removed dataframe
    MSEs = {'dmdenkf':[]}
    na_free_forecasts = restrict_flu_season(na_free_forecasts)
    total_data = na_free_forecasts['total data'].values
    for weeks_ahead in range(1,n_steps_max+1):
        total_forecast = np.real(na_free_forecasts[f'{weeks_ahead} total pred'].values)
        MSEs['dmdenkf'].append(mean_squared_error(total_data,total_forecast))
    return(MSEs)


def MSE_residuals_with_hb(m,n_steps_max,df):
    """
    Calculates the means squared errors over a forecast dataframe for the DMDEnKF and historical baseline prediction,
    Applied later in the code where the hb predicitons are now available

    Parameters
    ----------
    m : ilinetdata.ModellingChoices
        Modelling choices object with relevant data transforms etc
    n_steps_max : int
        The maximum number of timesteps ahead to calculate the forecast MSE's for
    df : pandas.DataFrame
        Forecast dataframe

    Returns
    -------
    MSEs : dict
        A dictionairy containing the mean squared errors for each forecast horizon up to n_steps_max,
        and those for the historical baseline prediciton
    """
    
    #calculate the MSE residuals over the na removed dataframe, including those for the historical baseline which is now available
    MSEs = {'dmdenkf':[]}
    na_free_forecasts = df.dropna()
    na_free_forecasts = restrict_flu_season(na_free_forecasts)
    total_data = na_free_forecasts['total data'].values
    hb = na_free_forecasts['total hb point estimate'].values
    MSEs['hb'] = mean_squared_error(total_data,hb)
    for n_steps in range(1,n_steps_max+1):
        total_forecast = np.real(na_free_forecasts[f'{n_steps} total pred'].values)
        MSEs['dmdenkf'].append(mean_squared_error(total_data,total_forecast))
    return(MSEs)


def gen_prev_year_weeks_list(series,pandemic_years):
    """
    Creates an ensemble using the values for a week in previous years

    Parameters
    ----------
    series : pandas.Series
        A series of the total data going back as many years as are wanted to be included in the ensemble
    pandemic_years : list
        A list of years not to use in the ensemble creation e.g. because they are atypical pandemic years

    Returns
    -------
    output_infs : list
        The list of ensembles for each week as they become available
    """
    
    #create ensemble for historical baseline
    #collects outputs in a list
    output_infs = []
    #loop through the dfs values
    for i, item in enumerate(series.iteritems()):
        #if no previous years data for that week, write NaN
        index_of_interest = i - 52
        if index_of_interest < 0:
            infs_entry = np.nan
        else:
            infs_entry = []
            while index_of_interest >= 0:
                if series.index[index_of_interest].year not in pandemic_years:
                    infs_entry.append(series.values[index_of_interest])
                index_of_interest -=52
        output_infs.append(infs_entry)
    return output_infs


def median_point_estimate(ensemble_preds,bw='silverman'):
    """
    Generates a point estimate from the historical baseline ensemble using their median values
    
    Parameters
    ----------
    ensemble_preds : list
        List of ensembles estimates for each timestep
    bw : str
        The method used to estimate the bandwidth in the kernel density estimation
    
    Returns
    -------
    output_pes : list
        The list of median estimates from the ensembles
    """
    
    #generate a point estimate for hb total infections to use in graphs to reflect new hb
    output_pes = []
    for epreds in ensemble_preds:
        if not isinstance(epreds,(list,np.ndarray)):
            if np.isnan(epreds):
                output_pe = np.nan
        else:
            kde = FFTKDE(kernel='gaussian', bw=bw)
            kde.fit(np.vstack(epreds))
            axis, density = kde.evaluate()
            cdf = np.cumsum(density)
            cdf = np.cumsum(density)/cdf[-1]
            above_med = [c for c in cdf if c >=0.5][0]
            below_med = [c for c in cdf if c <=0.5][-1]
            val_above_med = axis[list(cdf).index(above_med)]
            val_below_med = axis[list(cdf).index(below_med)]
            med = np.interp(0.5,[below_med,above_med],[val_below_med,val_above_med])
            output_pe = med
        output_pes.append(output_pe)
    return output_pes


def calc_accuracy_prob(truth,ensemble_preds,bw='silverman'):
    """
    Used in gen_total_accuracy_prob, uses Kernel Density Estimation to calculate an accuracy probability from an ensemble
    
    Parameters
    ----------
    truth : float
        True state value of the data at a specific timestep
    ensemble_preds : list
        List of ensembles estimates at that timestep
    bw : str
        The method used to estimate the bandwidth in the kernel density estimation
    
    Returns
    -------
    accuracy_prob : float
        The probability of being accurate based on the KDE's probability density
    """
       
    #func that uses KDE to calculate an accuracy probability from an ensemble
    kde = FFTKDE(kernel='gaussian', bw=bw)
    kde.fit(np.vstack(ensemble_preds))
    axis, density = kde.evaluate()
    relevant_densities = [(x,y) for x,y in zip(axis,density) if x>=truth-0.5 and x<truth+0.5]
    accuracy_prob = integrate.simps(y=[r[1] for r in relevant_densities],x=[r[0] for r in relevant_densities])
    return accuracy_prob


def gen_total_accuracy_prob(truth_series, ensemble_pred_series,bw='silverman'):
    """
    Generates the accuracy probability at all timesteps of an ensemble prediciton being accurate to the true data, using KDE
    
    Parameters
    ----------
    truth_series : pandas.Series
        Series of true state value of the data at all timesteps
    ensemble_preds : pandas.Series
        Series of ensembles at all timestep
    bw : str
        The method used to estimate the bandwidth in the kernel density estimation
    
    Returns
    -------
    output_accuracys : list
        The probability of being accurate based on the KDE's probability density at each timestep
    """
    
    #func that applies accuracy probability calc over a truth and ensemble prediction series
    truths = truth_series.values
    ensemble_preds = ensemble_pred_series.values
    output_accuracys = []
    for truth,epreds in zip(truths,ensemble_preds):
        if not isinstance(epreds,(list,np.ndarray)):
            if np.isnan(epreds):
                output_acc = np.nan
        else:
            try:
                output_acc = calc_accuracy_prob(truth,epreds,bw)
            except IndexError:
                #output_acc = 'Truth out of pdf range'
                #this occurs when the true value is outside the range of the pdf
                output_acc = 0
        output_accuracys.append(output_acc)
    return output_accuracys


def select_age_region(entry,age_region_index):
    """
    Utility to check for given entries if the stratified data exists or is np.nan,
    and if it does extract it from the strata vector
    
    Parameters
    ----------
    entry : list/numpy.array
        The data entries to check for np.nans
    age_region_index : int
        The index of the relevant strata that we want to check is not np.nan
    
    Returns
    -------
    entries age region index : list
        List of all values for the given strata in the provided entries
    """
    
    #func that turn a 40 age region state into correct series format for series accuracy probability calcs
    if not isinstance(entry,(list,np.ndarray)):
        if np.isnan(entry):
            return np.nan
    else:
        return [a[age_region_index] for a in entry]