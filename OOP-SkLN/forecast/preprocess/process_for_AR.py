from scipy import stats
from statsmodels.tsa.stattools import adfuller,kpss

class preprocess_AR():

    @staticmethod
    def test(obs:'pd.Series') -> 'pd.Series, boolean':
        '''
	        Perform an Augmented-Dicker-Fuller and KPSS-test to test seasonality and trend in time series. Depending on the four different
          results it chooses an ARIMA, Holt-Winters, Holt or Exponential Smoothing from the statsmodels library.
          Input:
	        :param obs: sequential data to analyze
          Output:
          :model: Choosen model according to behavior of data
          :regularization: Regularize data in case there is no stationarity 
    	'''
        assert type(obs) is pd.Series, "Data must be a pd.Series type"
      
        model, regularization = '', False
        
        # ADF-test
        ADF_statistic, ADF_p_value, *rest =  adfuller(obs)
        cv_1, cv_5, cv_10 = [value for key,value in rest[2].items()]
        rt_ADF = ''
        
        if ADF_statistic < cv_5:                      # If the ADF-statistic is less than the 5% confidence value the series is Stationary
            rt_ADF = rt_ADF.join('SS').strip()
        else:
            rt_ADF = rt_ADF.join('SnS').strip()

        # KPSS-test    
        kpss_statistic, kpss_p_value, lags_used, *rest = kpss(obs, regression='c')
        cv_10, cv_5, cv_25, cv_1 = [value for key,value in rest[0].items()]
      
        # Model Selection
        if (kpss_p_value < .1) & (rt_ADF == 'SnS'):     #Trend Not Stationary 
            model, regularization = model.join('holt').strip(), True
        elif (kpss_p_value < .1) & (rt_ADF == 'SS'):    #Trend and Stationary
            model = model.join('holt_winters').strip()
        elif (kpss_p_value >= .1) & (rt_ADF == 'SnS'):  #No trend no stationary
            model, regularization = model.join('sem').strip(), True
        elif (kpss_p_value >= .1) & (rt_ADF == 'SS'):   #No Trend and stationary
            model = model.join('arima').strip()
        return model, regularization

    @staticmethod
    def penalized_mean(obs:'pd.Series') -> 'pd.Series':
        '''
	        Implement Penalized Mean
	        :param obs: sequential data to analyze
    
    	'''
        if isinstance(obs, pd.Series):
           values = list(obs)
        else:
           values = obs
        
        obs_pm, mean, std = [], obs.mean(), obs.std()
        
        for v in values:
            if v > mean+2*std:         
                v = v*(1/(np.abs(mean-v)/(2*std)))
                y_pm.append(v)
            elif v > mean+std:
                v = v**2/np.abs(v+std/(std**2/mean-v))
                y_pm.append(v)
            elif v < mean-std:
                v = v**2/np.abs(v-std/((std**2/mean-v)))
                y_pm.append(v)
            elif v < mean-2*std:
                v = v*(1/((mean-v)/(2*std) - 1))
                y_pm.append(v)
            else:
                y_pm.append(v)    
 
        return pd.Series(y_pm)

    @staticmethod
    def boxcox_untransform(obs:'pd.Series', lmbd:float) -> 'pd.Series':
        '''
	        Untransform data after a Box-Cox transformation have been implemented.
          Input:
	        :param obs: sequential data to analyze
          :param lmbd: lambda coefficient of box-cox
          
    	'''
      
      assert type(obs) is pd.Series, "Data must be a pd.Series type"
      
        if lmbd == 0:
            obs = np.exp(obs)
        else:
            obs = np.exp((1/lmbd)*np.log(lmbd*obs+1))
        return obs  
