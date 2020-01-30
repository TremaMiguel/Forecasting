from scipy import stats
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.api import SimpleExpSmoothing, ExponentialSmoothing, Holt
from process_for_AR import *
from collections.abc import Sequence


def arima(obs, p_mean = False, boxcox = False, n_forecast = 2):
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
        assert type(obs) == Sequence, "Data must be of seq type"
        
        p = q = d = range(0,3)
        pdq  = list(itertools.product(p, d, q))
        seasonal_pdq = [(x[0], x[1], x[2], 52) for x in list(itertools.product(p, d, q))]
        
        if p_mean == True:
            y_prc = preprocess_AR().penalized_mean(obs)
        else:
            y_prc = obs

        untransform = False
        if boxcox == True:
           y_prc, lmbd = preprocess_AR().boxcox(y_prc)
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
            mod = sm.tsa.statespace.SARIMAX(y_pm,
                                    order=(order[0][0],order[0][1],order[0][2]),
                                    seasonal_order=(seasonal[0][0],seasonal[0][1],seasonal[0][2],52),
                                    enforce_stationarity=False,
                                    enforce_invertibility=False)
            results = mod.fit()
        except:
            print("Error with the model")  


        # Untransform Box-Cox Data
        if untransform:      
            y_pred = preprocess_AR().boxcox_untransform(y_pred,lmbd)  
            mse_min, forecast = ((y_pred - y.astype(float).values) ** 2).mean(), preprocess_AR().boxcox_untransform(results.forecast(n_forecast), lmbd)
        else:
            mse_min, forecast = ((y_pred - y.astype(float).values) ** 2).mean(), results.forecast(n_forecast)
      
        return (forecast, aic_min, bic,  mse_min, sse)
        
        
        
def sem(obs, p_mean = False, boxcox = False, n_forecast = 2):
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
        
        if p_mean == True:
            y_prc = preprocess_AR().penalized_mean(obs)
        else:
            y_prc = obs

        untransform = False
        if boxcox == True:
           y_prc, lmbd = preprocess_AR().boxcox(y_prc)
           untransform = True   

        # Perform a Grid Search to find the best model
        sse_min, ss = 1000000000, 0
        try:
            for i in [.95,.9,.85,.8,.75,.7,.65,.6,.55,.5,.45,.4,.35,.3,.25,.2,.15,.1,.05]:
                mdl = SimpleExpSmoothing(y_pm).fit(smoothing_level = i, optimized = False)
                sse = np.sum((y_prc.astype(float) - mdl.fittedvalues)**2)
                if sse <= sse_min:
                    sse_min, sl = sse, i
        except:
            print("Error while Runnig SEM")
        
        # Fit the best model
        mdl = SimpleExpSmoothing(y_prc).fit(smoothing_level = sl,optimized=False)
        aic, bic = mdl.aic, mdl.bic

        if untransform:
            pred = preprocess_AR().boxcox_untransform(mdl.fittedvalues, lmbd)
            forecast = preprocess_AR().boxcox_untransform(mdl.forecast(n_forecast), lmbd)
            mse = ((obs.astype(float).values - pred)**2).mean()
        else:
            pred = mdl.fittedvalues
            forecast = mdl.forecast(n_forecast)
            mse = ((obs.astype(float).values - pred)**2).mean()

        return (forecast, aic, bic, mse, sse_min)
