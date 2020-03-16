from scipy import stats
from statsmodels.tsa.stattools import adfuller,kpss
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
import rpy2.robjects as ro 
import numpy as np 
import matplotlib.pyplot as plt

# Activate automatic conversion
pandas2ri.activate()

# Import R libraries
strucchange, tsoutliers, base = importr('strucchange'), importr('tsoutliers'), importr('base')
zoo, forecast, imputeTS = importr('zoo'), importr('forecast'), importr("imputeTS")
tsoutliers, TSA, stats = importr('tsoutliers'), importr('TSA'), importr('stats')
pracma = importr('pracma')

class pp_transforms():

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
    def boxcox(obs:'pd.Series') -> 'pd.Series':
	'''
	   Implement a Box-Cox Transformation of Data
	   
       Input:
	   :param obs: sequential data to analyze
	   
       Output:
	   obs_bc: Data transformed
           lmbd: Optimal lambda
    	'''
        obs_bc, lmbd = stats.boxcox(obs)
	return (obs_bc, lmbd)

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


class pp_tests():

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

class pp_procedures_AR():

    def __init__(self, obs:'Rtype'):
        self.obs = obs
        self.obs_t = pd.Series()


    @staticmethod 
    def interpolation(self, method:str, plot:bool, params:dict) -> 'pd.Series':
        '''
          Perform interpolation to the given data (it should show missing values with NA) methods 
          from the R libraries forecast, R and zoo. Among them are NA, Kalman, Moving Average, 
          Seasonal Decompose, Seasonal Splitted or StructTS.
          
          Input:
            :param method: Interpolation method to apply
          
          Output:
            :res: Interpolated data
            :method: Regularize data in case there is no stationarity 
        '''

        # Interpolation Method
        if method == 'NaInterpolation':
            res = forecast.na_interp(self.obs, **params)
        elif method == 'KalmanInterpolation':
            res = imputeTS.na_kalman(self.obs, **params)
        elif method == 'MAInterpolation:
            res = imputeTS.na_ma(self.obs, **params)
        elif method == 'SDInterpolation':
            res = imputeTS.na_seadec(self.obs, **params)
        elif method == 'SSInterpolation':
            res = imputeTS.na_seasplit(self.obs, **params)  
        else:
            res = zoo.na_StructTS(self.obs, **params)       

        # Visualize Results
        if plot:
            plt.figure(figsize = (12, 6))
            plt.style.use('fivethirtyeight')
            plt.xticks(rotation=70)
            plt.plot(res,color = 'blue')
            plt.grid(False)
            plt.title(f"Interpolated data with method {method}")

        self.obs_t = res 
        return (res, method)

    @staticmethod
    def outlier_detection(self, method:str, plot:bool):
        '''
          Perform outlier detection applying the locate_outliers function from the tsoutliers
          package or detectAO from the TSA package.
          
          Input:
            :param method: Detection method to apply.
            :param plot: Wether or not to visualize the results.
          
          Output:
            :res: results of the applied method.
            :idx: Indexes of the type of outlier.
            :type: Type of outliers in the series.

        Note: Currently plot only works when you call the method locate_outliers
        '''


        # Fit an Arima model 
        fit = forecast_automarima(self.obs, max_p=3, max_q=3,
                                  ic = ro.StrVector(["bic"]),
                                  trace = True, nmodels = 15,
                                  stepwise = True)

        resid, pars = forecast.residuals_forecast(fit), tsoutliers.coefs2poly(fit)
        fitted = fit.rx2('fitted')

        if method == 'locate_outliers':
            res = tsoutliers.locate_outliers(resid, pars)
            idx, type = res['ind'] - 1, res['type']

        elif method == 'TSA':
            res = pd.DataFrame(TSA.detectAO(fit, robust = True))
            idx, type = res.iloc[0], 'AO'

        if plot:
            plt.figure(figsize = (12, 6))
            plt.style.use('fivethirtyeight')
            plt.plot(self.obs, color = 'blue')
            arrow = dict(arrowstyle='->', color='red', linewidth=5, mutation_scale=15)
            plt.plot(fitted, color = 'black')
            plt.grid(False)
            # Annotate arrows with type of outlier
            for i in range(0,len(idx)):
                plt.annotate(type[i],xy=(idx[0], fitted[idx[0]]), xytext=(idx[0], fitted[idx[0] + 3]), arrowprops=arrow)


        return (res, idx, type)


    def outlier_replacement(self, method:str, plot:bool):

        '''
          Perform outlier replacements implementing the function tsoutliers from the forecast package, by a Hampel
          filter or by an interpolation technique, see the method interpolation for more details about them. 
          
          Input:
            :param method: Outlier replacement method to apply.
            :param plot: Wether or not to visualize the results.
          
          Output:
            :res: results of the applied method.
            :idx: Indexes of the values to replace.
            :rep: Values that were replaced according to idx.
        '''

        res = org = self.obs
        
        # Option 1: Tsoutliers
        output, other = forecast_tsoutliers(self.obs), False 
        idx, rep = output.rx2('index'), output.rx2('replacements')
        idx = idx - 1 # Python indexes start in 0
        
        # Verify output
        if np.any(res.rx2('index')):
            other = True
        else:
            res[idx] = rep 

        # Other options
        if other:
            
            # Option 2: Hampel Filter
            if method == "Hampel":
                res_hmp = pracma_hampel(self.obs, k = 4, t0 = 2.5)
                res, idx, rep = res_hmp.rx2('y'), res_hmp.rx2('ind'), None
            
            # Option 3: Interpolation Methods
            else:
                output, idx, *rest = self.outlier_detection(method = 'locate_outliers', plot=False)
                self.obs[idx] = np.nan 
                res, method = self.interpolation(method = method, plot = False)
                rep = res[idx]

        if plot:
            f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(12,6))
            plt.style.use('fivethirtyeight')
            ax1.plot(org, color = 'blue')
            ax1.grid(False)
            ax1.title("Original Time Series")
            ax2.plot(res, color = 'red')
            ax2.grid(False)
            ax2.title("Modified Time Series") 

        self.obs_t, self.obs = res, org 

        return (res, idx, rep)
