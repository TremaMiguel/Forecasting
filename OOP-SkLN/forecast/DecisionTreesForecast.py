import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt

class EnsembleTrees():
    
    def __init__(self, dt:'pd.DataFrame', target_variable:str, X_train='pd.DataFrame', X_test='pd.DataFrame', y_train='pd.DataFrame', y_test='pd.DataFrame', train = True, *args):
      '''
	         Initialization Parameters. The train parameter controls if you're going to use a train/test focus or use your whole observations 
           for forecasting. 
           
           * Parameters when train = False
	        :param dt: Data with all the observations
	        :param target_variable: variable that you would like to forecast
	        
          * Parameters when train = True
          :param X_train: Data to train the model
          :param X_test: Data to test the model
          :param y_train: Objective variable to train the data
          :param y_test: Objective variable to test the data
	        :param target_variable: variable that you would like to forecast
    	'''    
        if train:
            self.X_train = X_train
            self.X_test = X_test
            self.y_train = y_train
            self.y_test = y_test
            self.train = train
            self.args = args
        else:
            self.dt = dt 
            self.target = dt[target_variable]
            self.dependent = dt.drop(columns=[target_variable]) 
            self.train = train
            
        
    def tree_model(self, q, model:str, mdl_params:dict, criteria = 'mae'):
    	'''
	        Based on the choosen model ('rfr' for RandomForest, 'gbr' for GradientBoosting, 'adr' for AdaBoost and 'xgbr' for XG-Boost)
          implement a Decision Tree for Forecasting one week of data.
	        :param model: which of the four ML models to implement according to the documentation
	        :param mdl_params: parameters of the ML model
	        :param criteria: metric to fit data
    	'''
      
        # Train-test Model
        if self.train:
            if model == 'rfr':
                mdl = RandomForestRegressor(**mdl_params)
            elif model == 'adr':
                mdl = AdaBoostRegressor(base_estimator = DecisionTreeRegressor(criterion = criteria,
                                                             max_features = "auto",
                                                             max_depth = 4,
                                                             random_state = 0), **mdl_params)
            elif model == 'gbr':
                mdl = GradientBoostingRegressor(**mdl_params)
            elif model == 'xgbr':
                mdl = xgb.XGBRegressor(**mdl_params)
            
            # Fit model
            mdl.fit(self.X_train, self.y_train)
            
            # Accuracy of model
            y_true = self.y_test.reset_index(drop = True)
            y_pred = pd.Series(mdl.predict(self.X_test))
            if criteria == 'mse':
                 rms  = sqrt(mean_squared_error(y_true, y_pred))
                 print(f"The RMSE is {rms}")
            else:      
                mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
                print(f"The MAPE is {mape}")
            
            # Visualization of Real vs Prediction
            plt.plot(y_pred, color = 'red')
            plt.plot(y_true, color = 'blue')
        
        # Forecast
        else:  
            if model == 'rfr':  
                mdl = RandomForestRegressor(**mdl_params)
            elif model == 'adr':
                mdl = AdaBoostRegressor(base_estimator = DecisionTreeRegressor(criterion = criteria,
                                                             max_features = "auto",
                                                             max_depth = 4,
                                                             random_state = 0), **mdl_params)
            elif model == 'gbr':
                mdl = GradientBoostingRegressor(**mdl_params)
            elif model == 'xgbr':
                mdl = xgb.XGBRegressor(**mdl_params)
            
            # Fit model
            mdl.fit(self.dependent, self.target)
            
            # Accuracy and Forecast
            y_true = self.target.reset_index(drop = True)
            y_pred = pd.Series(mdl.predict(self.dependent))
            
            # Take the last observation for forecasting
            X_for_forecast = self.dt.head(1).drop(columns = "lag_1")
            X_for_forecast.columns = list(self.dependent.columns)
            forecast = float(mdl.predict(X_for_forecast))
            
            # Evaluate Model
            if criteria == 'mse':
                metric  = sqrt(mean_squared_error(y_true, y_pred))
                print(f"The RMSE is {metric}", forecast)
            else:      
                metric = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
                print(f"The MAPE is {metric}", forecast)
                
            # Visualization of Real vs Prediction
            plt.plot(y_pred, color = 'red')
            plt.plot(y_true, color = 'blue')
            plt.plot(y_pred, color = 'red')
            plt.plot(y_true, color = 'blue')
        return (forecast, metric)
