# Forecasting  

Currently, it implements Decision Tree Models from the SKlearn Ensemble library and XGBoost library (see [1] and [2]) and AutoRegressive Models like Simple Exponential Smoothing, Holt, Holt-Winters and Arima from the StatsModels library for forecasting. 

For the decision trees the data is transformed with a Window Slide, for the Auto Regressive Models it is performed a Dicker Fuller Test and KPSS test to test for seasonality and trend in data to choose the right model to fit to the time series.



***References***

[1] Scikit-Learn Ensemble. URL: https://scikit-learn.org/stable/modules/ensemble.html

[2] XGBoost. URL:https://xgboost.readthedocs.io/en/latest/index.html


