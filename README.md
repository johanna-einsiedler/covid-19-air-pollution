# Understanding the Impact of COVID-19 on Local Air Pollution


## What can you do with this code?
This respository contains code and examples to (1) train a long-term air pollution prediction model with weather variables, (2) estimate the impact of COVID-19 related lockdowns on air pollution and (3) investigate the air pollution reduction potential in traffic. It can be applied to datasets for different countries.



## The Paper
This code was created in the writing of a master thesis. The resulting paper can be found here:


## Data 
In order to run these scripts you need a dataset containing daily observations of the air pollutant you want to investigate (e.g. NO2, PM2.5), and the weather variables windspeed, wind direction, humidity and temperature. Additional weather data can easily be incorporated.
Besides these variables the model uses additional lagged versions, long-term averages and other derivative variables. These can be calculated using the ['Generate Variables'](Generate_Variables.ipynb) Notebook.
