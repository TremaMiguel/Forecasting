# Forecasting  

Currently, it implements Decision Tree Models from the SKlearn Ensemble library and XGBoost library (see [1] and [2]) and AutoRegressive Models like Simple Exponential Smoothing, Holt, Holt-Winters and Arima from the StatsModels library for forecasting. 

For the decision trees the data is transformed with a Window Slide, for the Auto Regressive Models it is performed a Dicker Fuller Test and KPSS test to test for seasonality and trend in data to choose the right model to fit to the time series.

# Components of a Time Series (Theory)

A Time Series has three basic components, which are helpful to understand in order to grasp the concepts of structural breaks.

1. Trend. They are up or down changes (steep upward slope, plateauing downward slope).
2. Seasonality. The effect on the time series by the season (measured by time).
3. Noise. It is composed of:
   *   White Noise. If the variables are independent and identically distributed with a mean of zero. This means that all variables have the same variance (sigma^2) and each value has a zero correlation with all other values in the series. See [5] for more details. 
   *   Random Walk. A random walk is another time series model where the current observation is equal to the previous observation with a random step up or down. Checkout [6].

# Preprocess Data (Functions)

### Transform Data 

It implements Penalized Mean or Box Cox Transform to normalize data. 


### Test Data

Implements an augmented Dicker-Fuller test (unit root) or a KPSS-test in order to determine which type of model to apply between Simple Exponential Moving Average, Holt, Holt-Winters Additive or Seasonal Arima from the statsmodels library. 

### Process Data

Through the ```rpy2``` library we call an R enviroment to use distinct statistical packages as ```strucchange```, ```TSA```, ```zoo```, ```tsoutliers```, ```pracma```, ```imputeTS``` or ```forecast```.

##### 1. Interpolation.

It implements these types of interpolation: NA, Kalman, Moving Average, Seasonal Decompose, Seasonal Splitted or StructTS. In general, these methods are used to replace NA values in a time series. 

##### 2. Outlier Detection and Replacement.

An outlier is understood as an observation that is not explained by the model, so their role in forecast is limited in the sense that the presence of new outliers will not be predicted. Time Series data often presents outliers due to influence of non-usual events. Forecast accuracy in such situtations is reduced due to:

* Carry-over effect of the outlier on the point forecast
* Bias in the estimatess of model parameters.

In fact, there are different kinds of outliers:

* Additive Outlier. They do affect only a single observation.
* Innovative Outlier. An unusual observation that affects all later observations.
* Level Shift Outlier. Abrupt and permanent step change in the series.

So as different ways of detecting them, such as:

* Studentized Residuals / Bonferroni test. Normally, discrepant observations have large residuals
* F-statistic. Test for a single shift in the series.
* Fluctuation Tests. Based on the cumulative sum of the residuals.

Normally, to treat this unusual observations (outliers) you could implement the following steps with the aid of the ```outlier_dectection``` and ```outlier_replacement``` functions:

1. Identify the location and presence of outliers. (```outlier_dectection``` detects outliers applying the locate_outliers function from the tsoutliers package or detectAO from the TSA package).

2. Omit the outliers and do interpolation, for example, through a Kalman Filter or a Moving Average interpolation (```outlier_replacement``` implements this through the function tsoutliers from the forecast package, by a Hampel filter or by an interpolation technique).


##### 3. Structural Changes

With structural changes we seek to determine if the parameters of the model are not stable throughout the
sample period but change over time, that is, a parametric time series model. According to [3], Bruce Hansen recommends to
proceed as follows:

1. Test for structural breaks using the Andrews or Bai/Perron tests
2. If there is evidence of a break, estimate its date using Baiís
least-squares estimator
3. Calculate a confidence interval to assess accuracy (calculate both Bai
and Elliott-Muller for robustness)
4. Split the sample at the break, use the post-break period for estimation

However, you could also proceed by simply detecting the structural breaks and don't considering that observation for forecasting, the function ```structural_change``` is helpful for this, because it goes through fluctutation process or F-statistic test according to the methods of the strucchange R package. Besides, it plots the boundary with a 95% confidence interval to determine this points in the time series. These methods work as follows:

* Fluctuation Test. Based on the residuals, if they do change on time then it is necessary to change the parameters of the model.
* F-statistic . Null hypothesis against a single shift alternatives. Thus, compute an F statistic (or Chow statistic) for each conceivable breakpoint in a certain interval and reject the null hypothesis of structural stability if any of these statistics (or some other functional such as the mean) exceeds a certain critical value, when calling the method ```Ftest``` on the function ```structural_change``` it takes as interval the 20% to 80% of the data for testing, this parameters could be modified. 

##### 4. Outlier or Structural Change
 
The kind of perturbation that shifts cause on the observed time series can be classified as an outlier, when the shift affects the noise component, or as a structural change, when the shift affects one of the signal components.




***References***

[1] Scikit-Learn Ensemble. URL: https://scikit-learn.org/stable/modules/ensemble.html

[2] XGBoost. URL:https://xgboost.readthedocs.io/en/latest/index.html

[3] Hansen, Bruce. Advanced Time Series and Forecasting Lecture 5 Structural Breaks. The University of Wisconsin, 2012.

[4] Rousseeuyy, Peter; Leroy, Annick. Robust Regression and Outlier Detection. John Wiley & Sons, 1987.

[5] Stackoverflow. https://stats.stackexchange.com/questions/289349/why-do-we-study-the-noise-sequence-in-time-series-analysis 

[6] Quantstart. https://www.quantstart.com/articles/White-Noise-and-Random-Walks-in-Time-Series-Analysis/



