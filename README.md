# Understanding the Impact of COVID-19 on Local Air Pollution


## What can you do with this code?
This respository contains code and examples to (1) train a long-term air pollution prediction model with weather variables, (2) estimate the impact of COVID-19 related lockdowns on air pollution and (3) investigate the air pollution reduction potential in traffic. It can be applied to datasets for different countries.



## The Paper
This code was created in the writing of a master thesis. The resulting paper can be found here: There you can find additional details on the theory behind this code and some applications.


[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/johanna-einsiedler/covid-19-air-pollution/HEAD)

## Data 
In order to run these scripts you need a dataset containing daily observations of the air pollutant you want to investigate (e.g. NO2, PM2.5), and the weather variables windspeed, wind direction, humidity and temperature. Additional weather data can easily be incorporated. 
For demonstration purposes, you can find data for 5 stations in East Switzerland in this repository ([Raw Swiss Data](./che/df_che.csv)).
Besides these variables the model uses additional lagged versions, long-term averages and other derivative variables. These can be calculated using the [Generate Variables](Generate_Variables.ipynb) Notebook. 


## Model Selection Algorithm
To predict air pollution concentrations we use a **Generalized Additive Model**. In order to make the decision which variables to include in the model, a model selection algorithm is used. The algorithm is explained in more detail and implemented in [Model Selection](Model_Selection.ipynb).

## Model Training & Evaluation
The selected models (one for every station) are then trained on data before the COVID-19 Lockdown (*pre-LD model*). The trained model is evaluated using cross validation. Training and evaluation are performed in [Model Evaluation](Model_Evaluation.ipynb).

## Prediction & Transfer Model
In the script [Change Estimation Transfer](Change_Estimation_Transfer.ipynb) the following operations are implemented:
### Prediction of the Lockdown Period
In the next step, the trained model is used to predict air pollution during the lockdown period (Lockdown Dates for Austria, Switzerland & China are specified in the file [Dictionaries](dictionaries.py), sources are given in the paper). The difference to the observed values can be used as an estimate for the change of air pollution levels during the lockdown period.

### Transfer Learning - Lodckdown Model
The timeframe of the lockdowns (usually about 1 months) is too short to train a seperate GAM model on the data. However, we can create a model for the lockdown period (*LD-model*) using transfer learning, i.e. we keep all relationships between the pollutatnt and the weather variables fixed (it can be assumed that the influence of weather on pollution will not change significantly within one month) but do retraining for the variable 'weekday' which is used as a proxy for traffic.

### Model Evaluation - Lockdown Model
To evaluate the model performance of the lockdown model we again perform cross validation with a 3 day time periods as test sets.

### Post Lockdown Period
We further use the *pre-LD model* and the *LD model* to predict the period after the lockdown. These estimates can be used to infer how much a city has gone 'back to normal'.

### Predict 2019
Using the *LD-model* we make 'predictions' for the whole year of 2019, thus it can be estimated how air pollution levels would have changed if the city had been under lockdown for a whole year.


## Plots & Analysis
The outputs of all this scripts can be visualised using the [Plots](Plots.ipynb) script. It contains coe to obtain the following statistics and graphics

### Model Selection Outcome
Summarizes the outcome of the model selection algorithm, grouping of the variables can be inferred from [Dictionaries](dictionaries.py).

![alt text](/che/plots/table_model_selection.png)



