# DMDEnKF

Code accompaniment to: Combining Dynamic Mode Decomposition with Ensemble Kalman filtering for tracking and forecasting

## Installation

There are 2 levels of installation for the DMDEnKF package:

1. First, a bare bones installation that installs all the relevant code, and will allow the running of the dmdenkf example notebook.

2. Second, is the downloading of all the pre-fitted models and data used in the paper which this repository is the accompaniment for. These total around 15gb in memory, hence why the installation is seperated into 2 steps, and allow for the the running of the synthetic and ILI applications notebooks.

### Bare Bones Installation - Allows the running of dmdenkf example notebook only.

Ensure you have git installed, then download the code using:

 `git clone https://gitlab.surrey.ac.uk/sf00173/dmdenkf.git`

We recommend setting up a fresh Python environment so that this code can be run without issue. This can be done by installing the package manager Anaconda, then running the following commands in a terminal window, which create a new environment called "myenv", activates it, and then equips it with pip:

`conda create --name myenv`

`conda activate myenv`

`conda install pip`

Move to main project folder you downloaded using git titled "dmdenkf" if you have not done so already, and install the DMDEnKF package with all of it's dependencies via:

 `pip install -e .`

You now have all the code and dependencies installed, and can run the dmdenkf example notebook in your freshly activated Python environment via the command:

`jupyter notebook`

and then clicking on the notebook titled "dmdenkf example".

This notebook guides you through what an application of the DMDEnKF could look like on a simple synthetic dataset.


### Full Installation - Required to generate all graphs in the DMDEnKF paper.

The notebooks that generate the graphs used in the paper use large save files that store the results from the fitted models, to save the user rerunning the time consuming fitting process. As these save files are large, they will not have been downloaded by deafult as they are stored using the git LFS protocol.

First, ensure you have followed all the steps in the Bare Bones Installation section (although you need not open the dmdenkf example notebook).

Next, ensure git lfs is installed via:

`git lfs install`

Now, making sure you are still in the main project folder titled "dmdenkf", download the large save files required to generate the graphs used in the paper with:

 `git lfs pull`

NOTE: For this step I recommend you use a wired ethernet connection, as the size of this download is ~15gb.

Once the large files have downloaded, you are now able to run all the examples in the synthetic applications folder.

The final step to be able to run the ILI applications code is to download the ILINet data from the CDC's website at:

https://gis.cdc.gov/grasp/fluview/fluportaldashboard.html

On the website, hit the download data button in the top right hand corner of the page, and select the following options in the resultant popup box.

![](assets/ILINet_screenshot.png)

Then hit download data in the popup, and extract the received ILINet.csv file into the folder "ili_applications/data/american_flu" within the dmdenkf project.

With this step complete, you now have everything you need to run the ILI applications code, and hence all the code used in the DMDEnKF paper.


## Example Usage

A simple example of how the DMDEnKF can be applied is available in full in the dmdenkf example notebook, and we highlight the main steps below. First, some simple syntheic data is generated. In this case we use a 2-dimensional sin wave with an increasing frequency, and add normally distributed measurement noise, a plot of which can be seen here:

![](assets/synthetic_data.png)

Fitting the DMDEnKF to the noisy measurement data then constitutes of 2 parts. Fitting a DMD model of truncation rank r to the spin up data, via the code:

```
#Fit a DMD model to the spin up data
f = TDMD()
f.fit(data[:,:num_for_spin_up],r=2)
```

Then applying the filtering step over the rest of the data (Y), with all relevant variables provided:

```
#Fit the DMDEnKF to the rest of the data
dmdenkf = DMDEnKF(observation_operator=observation_operator, system_cov=system_cov,
                      observation_cov=observation_cov,P0=P0,DMD=f,ensemble_size=ensemble_size,Y=None)
dmdenkf.fit(Y=Y)
```

Once this has been performed, we can then view the DMD model's reconstruction over the spin up period, and how the DMDEnKF tracks the data over the course of the remaining timesteps:

![](assets/DMD_reconstruction_DMDEnKF_tracking.png)

To produce forecasts at each filtered timestep using the DMDEnKF, one simply provides it with the number of timesteps ahead to forecast (n_steps_ahead):

```
#create a forecast from the dmdenkfs ensemble
dmdenkf_forecast = dmdenkf.fast_predict_from_ensemble(n_steps_ahead)
```

The resultant forecasts can then be plotted to compare how accurately they predict the true data:

![](assets/4-step_ahead_DMDEnKF_forecast.png)

## Authors and acknowledgment
#### Authors:
Stephen A Falconer, David J.B. Lloyd, and Naratip Santitissadeekorn

Department of Mathematics, University of Surrey, Guildford, GU2 7XH, UK

#### Acknowledgments
Stephen Falconer acknowledges the UKRI, whose Doctoral Training Partnership Studentship helped fund his PhD.

He would also like to thank for their valuable discussions, Nadia Smith and Spencer Thomas from the National Physics Laboratory.
