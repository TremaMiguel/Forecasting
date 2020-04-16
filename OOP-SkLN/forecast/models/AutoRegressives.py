from scripts import *
from preprocess import pp_tests

class AutoRegressive():

    def __init__(self, df:bool, dt:'pd.DataFrame'=None, target_variable:str=None, target:'pd.Series'=None, *args):
        '''
	    Initialization Parameters. It assummes your target values are in a columnar form. 
            :param df: Is the data a dataframe of a series.
	    :param dt: Data with all the observations
	    :param target_variable: variable that you would like to forecast
            :param target: series of data
    	'''   
        if df:
            self.dt = dt 
            self.target = dt[target_variable]
            self.dependent = dt.drop(columns=[target_variable])
        else:
            self.target = target

    def AR_model(self, n_forecast:int=2):
        '''
	  First test for seasonality and trend in data with an Augmented Dicker Fuller Test and KPSS test. Then, based on the 
          choosen model ('arima' for Seasonal Arima, 'sem' for Simple Exponential Smoothing, 'holt' for Holt,
          'holt-winters' for Holt-Winters) implements an AutoRegressive model for Forecasting.
          
          Input:
	  :param model: which of the four ML models to implement according to the documentation
	  :param n_forecast: number of desired forecasted values 
          
          Output:
          forecast: Forecast for the next n_forecast observations
          aic: AIC of model
          bic: BIC of model
          mse: MSE of model
          sse: SSE of model
          model: model choosen for forecasting
    	'''
      
        # Test Seasonality and Trend in Data
        model, regularization = pp_tests().test(self.target)
      
        # Fit model for forecast
        if model == 'arima':
            forecast, aic, bic, mse, sse = arima(obs=self.target, 
                                                 p_mean = regularization, 
                                                 boxcox = regularization,  
                                                 n_forecast = n_forecast)
        elif model == 'sem':
            forecast, aic, bic, mse, sse  = sem(obs=self.target,
                                                p_mean = regularization, 
                                                boxcox = regularization,  
                                                n_forecast = n_forecast)
        elif model == 'holt':
            forecast, aic, bic, mse, sse = holt(obs=self.target,
                                                p_mean = regularization, 
                                                boxcox = regularization,  
                                                n_forecast = n_forecast)
        elif model == 'holt-winters':
            forecast, aic, bic, mse, sse = holt_winters(obs=self.target,
                                                p_mean = regularization, 
                                                boxcox = regularization,  
                                                n_forecast = n_forecast)
       
        return (forecast, aic, bic, mse, sse, model)
