from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri, numpy2ri
import rpy2.robjects as ro 
import numpy as np 
import rpy2 

# Activate automatic conversion
pandas2ri.activate()
numpy2ri.activate()

# R libraries
tsintermittent, tscount = importr('tsintermittent'), importr('tscount')
base, forecast = importr('base'), importr('forecast')

# R functions 
tsb, crost = tsintermittent.tsb, tsintermittent.crost
tsglm, predict_tsglm = tscount.tsglm, tscount.predict_tsglm

# R objects 
ts=ro.r('ts')


class Intermittent():

    def __init__(self, target:'pd.Series'=None):
        '''
           :param target: series of data
    	'''   
        self.target = target


    def Croston(self, n_forecast:int=2, cost_function:str='mar'):
        '''
	      Implement the crost function from the tsintermittent R package. 
          
          Input:
          :param n_forecast: number of steps ahead forecast.
          :param cost_function: metric to evaluate the performance of the model.
          
          Output:
          pred_croston: forecast for the n_forecast steps ahead specified.
    	'''

        # To avoid R-error
        obs = ro.FloatVector(self.target)
        
        # Fit a Croston model 
        try:
            mdl_croston = crost(ts(obs), h=n_forecast, cost=cost_function)
            pred_croston =  np.array(mdl_croston.rx2('frc.out'))
        except rpy2.rinterface.RRuntimeError as e:
            print(e)
        
        return pred_croston 

    def TSB(self, n_forecast:int=2, cost_function:str='mar'):
        '''
	      Implement the tsb function from the tsintermittent R package. 
          
          Input:
          :param n_forecast: number of steps ahead forecast.
          :param cost_function: metric to evaluate the performance of the model.
          
          Output:
          pred_tsb: forecast for the n_forecast steps ahead specified.
    	'''

        
        # To avoid R-error
        obs = ro.FloatVector(self.target)
        
        # Fit a Croston modified (TSB)
        try:
            mdl_tsb = tsb(ts(obs), h=n_forecast, cost=cost_function)
            pred_tsb =  np.array(mdl_tsb.rx2('frc.out'))
        except rpy2.rinterface.RRuntimeError as e:
            print(e)
        
        return pred_tsb

    def GLM(self, n_forecast:int=2, past_obs:int=2, past_mean:int=1, link:str="log", distr:str="nbinom"):
        '''
	      Fit a GLM model for time series of counts based on the tsglm function of the tscount
          R package. By default is assumes that the model will take two past observations and one
          past mean to predict the value. 

          Input:
          :param n_forecast: number of steps ahead forecast.
          :param past_obs: previous observations to be regressed on. 
          :param past_mean: previous conditional means to be regressed on. 
          :param link: Specify the link function 
          :param distr: Distribution of the model "poisson" for Poisson, 
                        "nbinom" for negative binomial. 
          
          Output:
          pred_glm: forecast for the n_forecast steps ahead specified.
    	'''

        # To avoid R-error
        obs = ro.FloatVector(self.target)
        
        # Fit an INGARCH model 
        try:
            mdl_glm = tsglm(ts(obs), model=ro.r(f'list(past_obs = {past_obs}, past_mean = {past_mean})'), link=link, distr=distr)
            pred_glm = predict_tsglm(mdl_glm, n_ahead=n_forecast)
            pred_glm = np.array(pred_glm.rx2('pred'))
        except rpy2.rinterface.RRuntimeError as e:
            print(e)
        
        return pred_glm

    