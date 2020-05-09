from scripts import *
from preprocess import pp_tests

class AutoRegressive():

    def __init__(self, target:'pd.Series'=None):
        '''
            :param target: series of data
    	'''   
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
