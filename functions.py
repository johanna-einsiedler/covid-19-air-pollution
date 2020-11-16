
# import libraries
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import shapefile as shp
import matplotlib.pyplot as plt
import geopandas as gpd
import seaborn as sns
from pygam import LinearGAM, LogisticGAM, s, f,l, GAM, te
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from datetime import datetime
import math
import sklearn.mixture as mix
import ipywidgets as widgets
from ipywidgets import interact, interact_manual
import re
from scipy import stats
from dateutil.relativedelta import relativedelta



def GAMf(df, in_var, ex_vars, city, cut, pred_end = 'one_month', train_duration = 'all'):
    """
    Parameters
    ----------
    df: 
        dataframe containing all variables of interest for the whole time of measurement
    in_var: 
        independent variable
    ex_vars: 
        list of explanatory variables
    city: 
        name of specific city
    cut: 
        string of the format '%m/%d/%Y' indicating the date where training set ends & test set starts
    pred_end:
        end of the prediction period
         if 'one_month' pred_end is set to one month after the cut
    train_duration:
        int, indicating the number of months that should be used for training
        defaults to 'all' -> all available data before the cut date will be used as training data
        
    Returns
    -------
    gam:
        fitted gam model instance
        
        
    model_statistics:
        vector containing the following information about the fitted model
        
        rmse:
            RMSE for test set
        r_squared:
            pseudo R-squared for the fitted GAM model
        fac2:
            fraction of predictions that lies between 50% and 200% of the corresponding measurements
        test_len:
            number of observations in the test set
        train_len:
            number of observations in the training set
        ratio:
            ratio of prediction to true values for test set
        avg_err:
        
    preds:
        a dataframe containing all explanatory variables, the independent variable, the predicted values & 
        the absolute error divided by the average value of the pollution variables in the training set
    """
    
    # drop rows with NAN values for explantory variables
    df =df.dropna(subset=ex_vars)
    
    # subset dataset to given city
    df = df[df['city']== city]
    
    # convert cut variable to datetime object
    cut = datetime.strptime(cut, '%m/%d/%Y')
    
    # if pred_end has the default value add one month to cut date to calculate end of the test dataset
    # else convert given string to datetime
    if(pred_end == 'one_month'):
        pred_end = cut+relativedelta(months=+1)
    else:
        pred_end = datetime.strptime(pred_end, '%m/%d/%Y')
        
    # determine subset of dataset used for training based on the given value for training duration
    if (train_duration == 'all'):
        df_train = df[df.index<cut]
    else:
        train_start = cut -relativedelta(months=+train_duration)
        df_train = df[df.index<cut]
        df_train = df_train[df_train.index>train_start]
    df_train = df_train.replace([np.inf, -np.inf], np.nan)
    df_train = df_train.dropna(subset = ex_vars)

    # determine subset of dataset used for test
    df_test = df[df.index>cut]
    df_test = df_test[df_test.index<pred_end]
    
    # extract values for independent and explanatory variables
    train_X = df_train[ex_vars].values
    train_y = np.log(df_train[in_var].values)
    test_X = df_test[ex_vars].values
    test_y = np.log(df_test[in_var].values)
    
    # check if test and training set contain sufficient observations
    if ((len(test_y) !=0) and (len(train_y) != 0)):
        
        # generate TermList for GAM
        string = str()
        if isinstance(ex_vars,str):
            length = 1
        else:
            length = len(ex_vars)
        for i in  range(0,length):
            if (ex_vars[i] in ['weekday', 'month','season','hour','season', 'new_year', 'daytime']) and (len(train_y)>300):
                string = string + "+f(" + str(i) + ")"
          #  else:  
            elif ('ws'in ex_vars[i]):
                string = string + '+l('+ str(i) +')'
            else:
                string = string + '+s(' + str(i) + ", lam = 0.6, basis = 'ps')"
            
        string = string[1:]
        
        # specify and fit GAM model
        gam = LinearGAM(eval(string))
        gam.fit(train_X, train_y)
        y_pred = gam.predict(test_X)

        # get max observed value for y
        max_value = train_y.max()

        # cut prediction to not get higher than maximum value in the training dataset
        y_pred[y_pred>max_value] = max_value
        
        # calculate model statistics
        ratio = np.mean(y_pred/test_y)
        rmse = np.sqrt(metrics.mean_squared_error(np.exp(test_y), np.exp(y_pred)))
        avg_err = np.mean(np.exp(test_y)-np.exp(y_pred))
        r_squared = list(gam.statistics_['pseudo_r2'].items())[0][1]
        fac2 = np.mean(test_y/y_pred<2)
        
        
        # dataframe with independent & dependent variables, prediction and prediction error
        preds = df_test.copy()[ex_vars]
        preds['true'] = np.exp(test_y)
        preds['y_pred']= np.exp(y_pred)
        preds['err'] = abs(preds['true'] - preds['y_pred'])/(np.mean(train_y))
        
        confidence = gam.prediction_intervals(test_X)

        preds['lower'] = np.exp(confidence[:,0])
        preds['upper'] = np.exp(confidence[:,1])
    else:
        # return Nan and give a warning if the training set is very small
        print('Problem with test and/or training data length for the station ' + city + 'in the month of ' + str(cut.month))
        print('Training Length: '+str(len(train_y)) + ' Test Length: ' + str(len(test_y)))
        rmse = gam = ratio = preds = avg_err = r_squared = fac2 = float("NaN")
                               
    # calculate length of test & training set
    test_len = len(test_X)
    train_len = len(train_X)
    model_statistics = [rmse,r_squared, fac2, test_len, train_len, ratio, avg_err]
    
    return(gam, model_statistics , preds)


def GAMf_train_test(df, in_var, ex_vars, city, cut, pred_end = 'one_month', train_duration = 'all'):
    """
    Parameters
    ----------
    df: 
        dataframe containing all variables of interest for the whole time of measurement
    in_var: 
        independent variable
    ex_vars: 
        list of explanatory variables
    city: 
        name of specific city
    cut: 
        string of the format '%m/%d/%Y' indicating the date where training set ends & test set starts
    pred_end:
        end of the prediction period
    train_duration:
        int, indicating the number of months that should be used for training
        defaults to 'all' -> all available data before the cut date will be used as training data
        
    Returns
    -------
    gam:
        fitted gam model instance
        
        
    model_statistics:
        vector containing the following information about the fitted model
        
        rmse:
            RMSE for test set
        r_squared:
            pseudo R-squared for the fitted GAM model
        fac2:
            fraction of predictions that lies between 50% and 200% of the corresponding measurements
        test_len:
            number of observations in the test set
        train_len:
            number of observations in the training set
        ratio:
            ratio of prediction to true values for test set
        avg_err:
        
    preds:
        a dataframe containing all explanatory variables, the independent variable, the predicted values & 
        the absolute error divided by the average value of the pollution variables in the training set
    """
    
    # drop rows with NAN values for explantory variables
    df =df.dropna(subset=ex_vars)
    
    # subset dataset to given city
    df = df[df['city']== city]
    
    # convert cut variable to datetime object
    cut = datetime.strptime(cut, '%m/%d/%Y')
    
    # if pred_end has the default value add one month to cut date to calculate end of the test dataset
    # else convert given string to datetime
    if(pred_end == 'one_month'):
        pred_end = cut+relativedelta(months=+1)
    else:
        pred_end = datetime.strptime(pred_end, '%m/%d/%Y')
        
    # determine subset of dataset used for training based on the given value for training duration
    if (train_duration == 'all'):
        df_train = df[df.index<cut]
    else:
        train_start = cut -relativedelta(months=+train_duration)
        df_train = df[df.index<cut]
        df_train = df_train[df_train.index>train_start]
    df_train = df_train.replace([np.inf, -np.inf], np.nan)
    df_train = df_train.dropna(subset = ex_vars)

    # determine subset of dataset used for test
    df_test = df[df.index>cut]
    df_test = df_test[df_test.index<pred_end]
    
    # extract values for independent and explanatory variables
    train_X = df_train[ex_vars].values
    train_y = np.log(df_train[in_var].values)
    test_X = df_test[ex_vars].values
    test_y = np.log(df_test[in_var].values)
    
    # check if test and training set contain sufficient observations
    if ((len(test_y) !=0) and (len(train_y) != 0)):
        
        # generate TermList for GAM
        string = str()
        if isinstance(ex_vars,str):
            length = 1
        else:
            length = len(ex_vars)
        for i in  range(0,length):

          #  else:  
            if ('ws'in ex_vars[i]):
                string = string + '+l('+ str(i) +')'
            else:
                string = string + '+s(' + str(i) + ", lam = 0.6, basis = 'ps')"
            
        string = string[1:]
        
        # specify and fit GAM model
        gam = LinearGAM(eval(string))
        gam.fit(train_X, train_y)
        y_pred = gam.predict(test_X)

        # get max observed value for y
        max_value = train_y.max()

        # cut prediction to not get higher than maximum value in the training dataset
        y_pred[y_pred>max_value] = max_value
        
        
        # calculate fitted prediction for training data
        y_pred_train = gam.predict(train_X)
        y_pred_train[y_pred_train>max_value] = max_value
        
        # calculate model statistics
        ratio = np.mean(y_pred/test_y)
        rmse_test = np.sqrt(metrics.mean_squared_error(np.exp(test_y), np.exp(y_pred)))
        avg_err = np.mean(np.exp(test_y)-np.exp(y_pred))
        r_squared = list(gam.statistics_['pseudo_r2'].items())[0][1]
        fac2 = np.mean(test_y/y_pred<2)
        rmse_train = np.sqrt(metrics.mean_squared_error(np.exp(train_y), np.exp(y_pred_train)))
        
        
        # dataframe with independent & dependent variables, prediction and prediction error
        preds = df_test.copy()[ex_vars]
        preds['true'] = np.exp(test_y)
        preds['y_pred']= np.exp(y_pred)
        preds['err'] = abs(preds['true'] - preds['y_pred'])/(np.mean(train_y))
        
        confidence = gam.prediction_intervals(test_X)

        preds['lower'] = np.exp(confidence[:,0])
        preds['upper'] = np.exp(confidence[:,1])
    else:
        # return Nan and give a warning if the training set is very small
        print('Problem with test and/or training data length for the station ' + city + ' in the month of ' + str(cut.month))
        print('Training Length: '+str(len(train_y)) + ' Test Length: ' + str(len(test_y)))
        rmse_train = rmse_test = gam = ratio = preds = avg_err = r_squared = fac2 = float("NaN")
                               
    # calculate length of test & training set
    test_len = len(test_X)
    train_len = len(train_X)
    model_statistics = [rmse_train, rmse_test,r_squared, fac2, test_len, train_len, ratio, avg_err]
    
    return(gam, model_statistics , preds)




def time_plot_conf(df, gam, ex_vars, in_var, city, cut, pred_end = 'one_month', train_duration = 'all' ):

    
    # get respective data for city and pred_mon
    df = df[df['city']== city]

    cut = datetime.strptime(cut, '%m/%d/%Y')
    
    # determine subset of dataset used for training based on the given value for training duration
    if (train_duration == 'all'):
        df_train = df[df.index<cut]
    else:
        train_start = cut -relativedelta(months=+train_duration)
        df_train = df[df.index<cut]
        df_train = df_train[df_train.index>train_start]
    
    # determine subset of dataset used for test
    df_test = df[df.index>cut]
    df_test = df_test[df_test.index<pred_end]
    
    # drop NAN values
    df_train = df_train.replace([np.inf, -np.inf], np.nan)
    df_train = df_train.dropna(subset = ex_vars)
    
    # extract values for independent and explanatory variables
    train_X = df_train[ex_vars].values
    train_y = df_train[in_var].values
    test_X = df_test[ex_vars].values
    test_y = df_test[in_var].values
    
    # stack values for training and test data
    stack_X = np.concatenate((train_X, test_X), axis=0)
    stack_y = np.concatenate((train_y, test_y))
    
    # put date & true values in dataframe
    preds = pd.DataFrame(stack_X)
    preds['true'] = stack_y
    preds['date'] = pd.concat((df_train,df_test)).index
    
    preds = preds.replace([np.inf, -np.inf], np.nan)
    preds = preds.dropna()
    
    # predict training and test data
    y_pred =  gam.predict(stack_X)

    # cut prediction to not get higher than max of training data
    max_value = train_y.max()
    y_pred[y_pred> np.log(max_value)] = np.log(max_value)
    
    # calculate confidence interval
    confidence = gam.prediction_intervals(stack_X)
    
    # add prediction and difference to dataframe
    preds['pred'] = np.exp(y_pred)
    preds['diff'] = abs(preds['pred'] - preds['true'])
    preds['lower'] = np.exp(confidence[:,0])
    preds['upper'] = np.exp(confidence[:,1])
    
    
    # plot true value, prediction & difference
    plt.figure(figsize=(150,15), dpi= 80)
    ax = plt.gca()
    preds.plot(kind='line',x='date',y='true',ax=ax,  label = 'True Values')
    preds.plot(kind='line',x='date',y='pred', color='orange', ax=ax, label = 'Prediction')
    preds.plot(kind='line',x='date',y='diff', color='green', ax=ax, label = 'Difference')
    preds.plot(kind='line',x='date',y='lower', color='grey', ax=ax, label = 'Lower Bound', linestyle = 'dashed')
    preds.plot(kind='line',x='date',y='upper', color='grey', ax=ax, label = 'Upper Bound', linestyle = 'dashed')

    plt.ylim(top=350, bottom = 0)
    plt.axvline(cut, color = 'red', linewidth = 2) # Cut of train and test dataset
    plt.legend()
    

    
    
def curves(gam, ex_vars):
    """
    Parameters
    ----------
    gam: fitted gam model instance of interest
    
    Returns
    ----------
    Plot of estimated relationshps of explanatory variables to independent variables
    
    """

    
    titles = ex_vars
    plt.figure()

    fig, axs = plt.subplots(1,len(gam.terms)-1,figsize=(40, 5))

    for i, ax in enumerate(axs):
        term = gam.terms[i]
        if term.isintercept:
            continue
        XX = gam.generate_X_grid(term=i)
        pdep, confi = gam.partial_dependence(term=i, X=XX, width=0.95)
        ax.plot(XX[:, term.feature], pdep)
        ax.plot(XX[:, term.feature], confi, c='r', ls='--')
        ax.set_title(titles[i])
