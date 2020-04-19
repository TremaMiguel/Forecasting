from scipy import stats
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.api import SimpleExpSmoothing, ExponentialSmoothing, Holt
from preprocess import pp_transforms, pp_tests, pp_processes
import pandas as pd 
import numpy as np
import itertools


def arima(obs, p_mean:bool, boxcox:bool, n_forecast:int):
        '''
           Implement a Seasonal Arima Model with Grid Search to find the model with the best
           combination of parameters based on a MSE minimization.
           
           Input:
           :param obs: sequential data for forecasting
           :param p_mean: wether or not to apply penalized_mean
           :param boxcox: wether or not to apply boxcox transformation
           :param n_forecast: number of observations to forecast
          
           Output:
           forecast: Forecast for the next n_forecast observations
           aic_min: AIC of model
           bic: BIC of model
           mse_min: MSE of model
           sse: SSE of model
        '''                  
        assert type(obs) == pd.core.series.Series, "Data must be of pandas Series type"
        
        p = q = d = range(0,3)
        pdq  = list(itertools.product(p, d, q))
        seasonal_pdq = [(x[0], x[1], x[2], 52) for x in list(itertools.product(p, d, q))]
        
        if p_mean == True:
            y_prc = pp_transforms().penalized_mean(obs)
        else:
            y_prc = obs

        untransform = False
        if boxcox == True:
            y_prc, lmbd = pp_transforms().boxcox(y_prc)
            untransform = True        

        # Perform a Grid Search to find the best model
        aic_min = mse_min = 1000000
        bic = sse = 0    
        order = seasonal = []
        iterables = [[p, s] for p in pdq for s in seasonal_pdq] 
        for i in iterables:
            p, s = i
            try:
                mod = sm.tsa.statespace.SARIMAX(y_prc,
                                                order=p,
                                                seasonal_order=s,
                                                enforce_stationarity=False,
                                                enforce_invertibility=False)
                results = mod.fit()
                aic, bic, pred  = results.aic, results.bic, results.get_prediction(dynamic=False)
                y_pred = pred.predicted_mean
                mse, sse = ((y_pred - y_prc) ** 2).mean(), np.sum(((y_pred - y_prc)**2).mean())
                if (mse <= mse_min) & (~np.isnan(aic)):
                    order = seasonal =  []
                    mse_min, aic_min = mse, aic
                    print('MSE '+ str(mse_min))
                    order.append(p)
                    seasonal.append(s)
            except:
                continue


        # Fit the best model
        try:
            mod = sm.tsa.statespace.SARIMAX(y_prc,
                                    order=(order[0][0],order[0][1],order[0][2]),
                                    seasonal_order=(seasonal[0][0],seasonal[0][1],seasonal[0][2],52),
                                    enforce_stationarity=False,
                                    enforce_invertibility=False)
            results = mod.fit()
        except Exception as e:
            print("Error while fitting the model","\n",e)
        


        # Untransform Box-Cox Data
        if untransform:      
            y_pred = pp_transforms().boxcox_untransform(y_pred,lmbd)  
            mse_min, forecast = ((y_pred - y.astype(float).values) ** 2).mean(), pp_transforms().boxcox_untransform(results.forecast(n_forecast), lmbd)
        else:
            mse_min, forecast = ((y_pred - y.astype(float).values) ** 2).mean(), results.forecast(n_forecast)
      
        return (forecast, aic_min, bic,  mse_min, sse)
        
        
        
def sem(obs, p_mean:bool, boxcox:bool, n_forecast:int):
        '''
           Implement a Simple Exponential Smoothin with Grid Search to find the model with the best
           combination of parameters based on an SSE minimization.
           
           Input:
           :param obs: sequential data for forecasting
           :param p_mean: wether or not to apply penalized_mean
           :param boxcox: wether or not to apply boxcox transformation
           :param n_forecast: number of observations to forecast
          
           Output:
           forecast: Forecast for the next n_forecast observations
           aic_min: AIC of model
           bic: BIC of model
           mse: MSE of model
           sse_min: SSE of model
        '''
        assert type(obs) == pd.core.series.Series, "Data must be of pandas Series type"
        
        if p_mean == True:
            y_prc = pp_transforms().penalized_mean(obs)
        else:
            y_prc = obs

        untransform = False
        if boxcox == True:
            y_prc, lmbd = pp_transforms().boxcox(y_prc)
            untransform = True   

        # Perform a Grid Search to find the best model
        sse_min, ss = 1000000000, 0
        print(obs)
        print(y_prc)
        try:
            for i in [.95,.9,.85,.8,.75,.7,.65,.6,.55,.5,.45,.4,.35,.3,.25,.2,.15,.1,.05]:
                mdl = SimpleExpSmoothing(y_prc).fit(smoothing_level = i, optimized = False)
                sse = np.sum((y_prc.astype(float) - mdl.fittedvalues)**2)
                if sse <= sse_min:
                    sse_min, sl = sse, i
        except Exception as e:
            print("Error while fitting the model","\n",e)
        
        # Best model
        mdl = SimpleExpSmoothing(y_prc).fit(smoothing_level = sl,optimized=False)
        aic, bic = mdl.aic, mdl.bic

        # Untransform Box-Cox Data
        if untransform:
            pred = pp_transforms().boxcox_untransform(mdl.fittedvalues, lmbd)
            forecast = pp_transforms().boxcox_untransform(mdl.forecast(n_forecast), lmbd)
            mse = ((obs.astype(float).values - pred)**2).mean()
        else:
            pred = mdl.fittedvalues
            forecast = mdl.forecast(n_forecast)
            mse = ((obs.astype(float).values - pred)**2).mean()

        return (forecast, aic, bic, mse, sse_min)

def holt(obs, p_mean:bool, boxcox:bool, n_forecast:int):
    '''
           Implement a Holt Model with Grid Search to find the model with the best
           combination of parameters based on a SSE minimization.
           
           Input:
           :param obs: sequential data for forecasting
           :param p_mean: wether or not to apply penalized_mean
           :param boxcox: wether or not to apply boxcox transformation
           :param n_forecast: number of observations to forecast
          
           Output:
           forecast: Forecast for the next n_forecast observations
           aic_min: AIC of model
           bic: BIC of model
           mse_min: MSE of model
           sse: SSE of model
    	  '''                  
    assert type(obs) == pd.core.series.Series, "Data must be of pandas Series type"
        
    if p_mean == True:
        y_prc = pp_transforms().penalized_mean(obs)
    else:
        y_prc = obs

    untransform = False
    if boxcox == True:
        y_prc, lmbd = pp_transforms().boxcox(y_prc)
        untransform = True  

    # Fit model and forecast
    sse_min = 1000000000
    alpha = beta = phi = 0
    values = [.95,.9,.85,.8,.75,.7,.65,.6,.55,.5,.45,.4,.35,.3,.25,.2,.15,.1,.05]
    combinations = [[v1, v2, v3] for v1 in values for v2 in values for v3 in values]
    
    try:
        for c in combinations: 
            sl,ss,ds = c
            mdl = Holt(y_prc, damped = True).fit(smoothing_level=sl, 
                                                   smoothing_slope=ss,
                                                   damping_slope=ds,
                                                   optimized=False)           
            sse = np.sum((y_prc - mdl.fittedvalues)**2)
            if sse <= sse_min:
                sse_min, alpha, beta, phi = sse, sl, ss, ds
    except Exception as e:
            print("Error while fitting the model","\n",e)
        
    # Best model
    mdl = Holt(y_prc, damped=True).fit(smoothing_level=alpha, 
                                           smoothing_slope=beta,
                                           damping_slope=phi, 
                                           optimized=False)
        
    if untransform:
        pred = pp_transforms().boxcox_untransform(mdl.fittedvalues, lmbd)
        forecast = pp_transforms().boxcox_untransform(mdl.forecast(n_forecast), lmbd)
    else:
        pred = mdl.fittedvalues
        forecast = mdl.forecast(n_forecast)
    
    # Metrics
    aic, bic, mse = mdl.aic, mdl.bic, ((obs.astype(float).values - pred)**2).mean()

    return (forecast, aic, bic, mse, sse_min)  


def holt_winters(obs, p_mean:bool, boxcox:bool, n_forecast:int):
    '''
       Implement a Holt Additive Model, abrupt changes decides the seasonal component based 
       on the number of changes (y_{i+1} - y_{i}) that are greater than the median of the observations (obs)
           
       Input:
       :param obs: sequential data for forecasting
       :param p_mean: wether or not to apply penalized_mean
       :param boxcox: wether or not to apply boxcox transformation
       :param n_forecast: number of observations to forecast

       Output:
       forecast: Forecast for the next n_forecast observations
       aic_min: AIC of model
       bic: BIC of model
       mse_min: MSE of model
       sse: SSE of model
    '''  

    assert type(obs) == pd.core.series.Series, "Data must be of pandas Series type"
        
    if p_mean == True:
        y_prc = pp_transforms().penalized_mean(obs)
    else:
        y_prc = obs
    
    #Abrupt Changes
    mean = obs.mean()
    dac = uac = sl = 0
    for i in range(len(obs) - 1):
        if obs.values[i] - obs.values[i+1] > mean:
            dac += 1
    for i in range(len(obs) - 1):
        if obs.values[i+1] - obs.values[i] >= mean:
            uac += 1
    if (dac == 1) & (uac == 1):
        sl = dac + uac
    else:
        sl = max(dac,uac)
    
    # Fit model and forecast
    alpha = beta = phi = 0 
    error = ''
    try:
        mdl = ExponentialSmoothing(y_prc, 
                                   seasonal_periods=sl, 
                                   trend='add', 
                                   seasonal='add', 
                                   damped=True).fit(use_boxcox=boxcox)
    except:
        mdl = ExponentialSmoothing(y_prc, 
                                   seasonal_periods=dac + uac, 
                                   trend='add', 
                                   seasonal='add', 
                                   damped=True).fit(use_boxcox=boxcox)
  
    
    # Metrics
    aic, bic = mdl.aic, mdl.bic
    mse = ((obs.astype(float).values - mdl.fittedvalues.values)**2).mean()
    sse = np.sum((obs.astype(float).values - mdl.fittedvalues.values)**2)  
    forecast = mdl.forecast(n_forecast)
    
    return (forecast, aic, bic, mse, sse)
