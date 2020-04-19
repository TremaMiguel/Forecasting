# Forecasting  

This repository is intended to be a recopilaton of different techniques and models that you can perform while forecasting an univariate time series using both Python libraries and R packages together. From **autoregressive models** (`Simple Exponential Smoothing`, `Holt`, `Holt-Winters` or `Arima`), **ensemble trees** (`Ada Boost`, `Gradient Boosting`, `Random Forest`, `XGBoost`) and **neural networks** (LSTM, CNN). Additionaly, you can perform **outlier detection**, **interpolation** and **structural change tests** based on R packages like `tsoutliers` and `strucchange`. There is the option to implement the **tasks in parallel**, check section F `Parallel Computation`. Finally, for **setting up a virtual environment with R and Python** checkout the setting up instrucions under the `setup` folder.  

> **NOTE.** To visualize the latex code in this file add the MathJax plugin for github avaible on this [link](https://chrome.google.com/webstore/detail/mathjax-plugin-for-github/ioemnmodlmafdkllaclgeombjnmnbima/related)

## A. Autoregressive Models

The selection of the parameters ```$p$```, ```$d$``` or ```$q$``` for the Arima model or the ```$\alpha$```, ```$\beta$``` for the other models is based on a grid search. See section E for more details.

## B. Decision Tree Models

First, the data is transformed with the function ```window_slide```. This is done, in order to be able to forecast when calling the ```predict``` method of each model, that is, for constructing the variable ```y_{t+1}``` we consider n past observations ```$y_{t+1} = y_{t} + y_{t-1} + ... + y_{t-n+1}$```. Then, call the desired model (```'rfr'``` for RandomForest, ```'gbr'``` for GradientBoosting, ```'adr'``` for AdaBoost and ```'xgbr'``` for XG-Boost) with the function ```tree_model```. Finally, the parameters of each model where choosen according to [3].


## C. Neural Networks

Currently it implements a stacked LSTM. To specify an LSTM specify `model == 'Stacked-LSTM'` when calling the function `nn_model` of the class `NeuralNetworks`. Furthermore, you can specify the number of stacked layers and units by setting the integer parameters `layers` and `units` respectively, for example, if you set `layers=4` and `units=8` each layer will have (units / number of hidden layer) as units.  


## D. Theory

> **A. Components of a Time Series**

A Time Series has three basic components, which are helpful to understand to identify appropiately a forecasting method that is capable of capturing the patterns of the time series data.

1. **Trend**. They are up or down changes (steep upward slope, plateauing downward slope).
2. **Seasonality**. The effect on the time series by the season (measured by time).
3. **Noise**. It is composed of:
   *   White Noise. If the variables are independent and identically distributed with a mean of zero. This means that all variables have the same variance ($\sigma^2$) and each value has a zero correlation with all other values in the series. See [6] for more details. In other words, the series shows no autocorrelation.
   *   Random Walk. A random walk is another time series model where the current observation is equal to the previous observation with a random step up or down. Checkout [7].

4. **Cycles**. It happens when the time series exhibits rises and fall that aren't of fixed frequency. It is not important not to confuse this concept with seasonality. When the frequency is unchanging and associated with some calendar date then there is seasonality. On the other hand, the fluctuations are cyclic when there are not of a fixed frequency. 

> **B. Autocorrelation**

Autocorrelation measure the **linear relationship between lagged values** in a time series. The autocorrelation coefficient is given by the formula 

$$r_{k} = \dfrac{\sum_{t=k+1}^{L}(y_{t}-\overline{y})(y_{t-k}-\overline{y})}{\sum_{t=1}^{L}(y_{t}-\overline{y})^2}, $$

in other words $r_{1}$ measure the relationship between $y_{t}$ and $y_{t-1}$ and so with $r_{2}, r_{3},...r_{L-1}$.

These coefficients are plot to show the autocorrelation function or `ACF`.

> **C. Interpreting an ACF plot**

The ACF plot allow us to identify trend, seasonality or a mixture of the both in the time series.

 When data have trend, the autocorrelation for the first lags is large and positive (the nearer the data in time the similar they'll be in size/value) and it slowly decrease as the lag increase. By contrast, when the data is seasonal we will see larger values appear every certain lag, that is, the autocorrelation is largeer at multiples of the seasonal frequency than  other lag values. Finally, when data is both trended and seasonal, it is likely to see a combination of these effects. The above image, taken from [9], shows an example of this behavior  

 ![Seasonal_trend_ACF](figs/Seasonal_trend_ACF.png)

> **D. Statistical tests for autocorrelation**

We would like to test the different $r_{k}$ coefficients. But doing multiple single test could yield a false positive, in other words, we could get to conclude that there is autocorrelation in the residuals when that is not true. We could overcome this with a `Portmanteau test` setting the null hypothesis as following 

$$H_{0} = \text{Autocorrelations come from a white noise.}$$

One such test is the `Ljung-Box test`

$$Q= L(L+2)\sum_{k=1}^h (L-k)^{-1} \: r_{k}^2,$$

where $h$ is the maximum lag considered. Thus, a large value of $Q$ implies that the autocorrelations do not come from a white noise series. Another test that can be considered is the `Breusch-Godfrey` test, both of them are implemented in the function `checkresiduals()` of the `forecast` R package. 


> **E. Model Diagnosis**

The residual is the difference between the fitted value and the real value, in mathematical terms

$$e_{t} = y_{t} - \hat{y_{t}}.$$

To check wheter a model has captured the information adequately one should check that the residuals follow the next properties:

1. `The residuals are uncorrelated`. When correlations is present it means that there is information left in the residuals which should be used in computing forecasts.

2. `The residuals have mean zero`. When the mean of the residuals is different from zero, the forecasts are biased. 

3. `The residuals have constant variance`. 

4. `The residuals are normally distributed`. 

Thus, if the residuals of a model does not satisfy these properties it can be improved. For example, to fix the bias problem one just add the mean of the residuals to all the points forecasted.

## E. Preprocess Data (Functions)

### E.1 Transform Data 

It implements `Penalized Mean`, `Box Cox Transform` or `Yeo-Johson` transformation to normalize data. In case you only have positive value use the `Box Cox Transform` and transform the data according to the next table

| Lambda Value | Transformation of response variable |
| -------------| ------------------------------------|
|   -3         |    y^{-3}                           |
|   -2         |    y^{-2}                           |
|   -1         |    y^{-1}                           |
|   -.5        |    y^{-.5}                          |
|    0         |    log(y)                           |
|    0.5       |    y^{.5}                           |   
|    1         |    y^1                              |  
|    2         |    y^2                              |
|    3         |    y^3                              |

In case of negative values use the `Yeo-Johnson` transformation. `Penalized Mean` is more suitable when you have high leverage points or heavy outliers. 


### E.2 Test Data

Implements an augmented `Dicker-Fuller test` (unit root) or a `KPSS-test` in order to determine which type of model to apply between `Simple Exponential Moving Average`, `Holt`, `Holt-Winters Additive` or `Seasonal Arima` from the `statsmodels` library. 

### E.3 Process Data

Through the ```rpy2``` library we call an R enviroment to use distinct statistical packages as ```strucchange```, ```TSA```, ```zoo```, ```tsoutliers```, ```pracma```, ```imputeTS``` or ```forecast```.

##### E.3.1. Interpolation.

It implements different types of interpolation such as: `NA`, `Kalman`, `Moving Average`, `Seasonal Decompose`, `Seasonal Splitted` or `StructTS`. In general, these methods are used to replace NA values in a time series. 

##### E.3.2. Outlier Detection and Replacement.

An outlier is understood as an observation that is not explained by the model, so their role in forecast is limited in the sense that the presence of new outliers will not be predicted. **Time Series data often presents outliers due to influence of non-usual events**. Forecast accuracy in such situtations is reduced due to:

* Carry-over effect of the outlier on the point forecast
* Bias in the estimatess of model parameters.

In fact, there are different kinds of outliers:

* **Additive Outlier**. They do affect only a single observation.
* **Innovative Outlier**. An unusual observation that affects all later observations.
* **Level Shift Outlier**. Abrupt and permanent step change in the series.

So as different ways of detecting them, such as:

* **Studentized Residuals / Bonferroni test**. Normally, discrepant observations have large residuals
* **F-statistic**. Test for a single shift in the series.
* **Fluctuation Tests**. Based on the cumulative sum of the residuals.

Normally, to treat this unusual observations (outliers) you could implement the following steps with the aid of the ```outlier_dectection``` and ```outlier_replacement``` functions:

1. Identify the location and presence of outliers. (```outlier_dectection``` detects outliers applying the locate_outliers function from the tsoutliers package or detectAO from the TSA package).

2. Omit the outliers and do interpolation, for example, through a Kalman Filter or a Moving Average interpolation (```outlier_replacement``` implements this through the function tsoutliers from the forecast package, by a Hampel filter or by an interpolation technique).


##### E.3.3. Structural Changes

With **structural changes we seek to determine if the parameters of the model are not stable** throughout the
sample period but change over time, that is, a parametric time series model. According to [4], Bruce Hansen recommends to
proceed as follows:

1. Test for structural breaks using the Andrews or Bai/Perron tests
2. If there is evidence of a break, estimate its date using Baiís
least-squares estimator
3. Calculate a confidence interval to assess accuracy (calculate both Bai
and Elliott-Muller for robustness)
4. Split the sample at the break, use the post-break period for estimation

However, you could also proceed by simply detecting the structural breaks and don't considering that observation for forecasting, the function ```structural_change``` is helpful for this, because it goes through **fluctutation process** or **F-statistic test** according to the methods of the strucchange R package. Besides, it plots the boundary with a 95% confidence interval to determine this points in the time series. These methods work as follows:

* **Fluctuation Test**. Based on the residuals, if they do change on time then it is necessary to change the parameters of the model.
* **F-statistic**. Null hypothesis against a single shift alternatives. Thus, compute an F statistic (or Chow statistic) for each conceivable breakpoint in a certain interval and reject the null hypothesis of structural stability if any of these statistics (or some other functional such as the mean) exceeds a certain critical value, when calling the method ```Ftest``` on the function ```structural_change``` it takes as interval the 20% to 80% of the data for testing, this parameters could be modified. 

##### E.3.4. Outlier or Structural Change
 
The kind of perturbation that shifts cause on the observed time series can be classified as an outlier, when the shift affects the noise component, or as a structural change, when the shift affects one of the signal components.


## F. Parallel Computation 

```Pool``` and ```Process``` are two ways of executing tasks parallelly, but in a different way. It works by mapping the input to the different processors, then distributes the processes among the avaible cores in ```FIFO``` manner. Finally, it waits for all the tasks to finish to return an output. One particularity is that the processes in execution are stored in memory. By contrast, the ```Process``` class assign all the processes in memory and schedules execution using FIFO policy. When the process ends it schedules a new one for execution. I suggest to read [8] and the official documentation of the Multiprocessing library for better comprehension.

##### F.1 When to use Pool or Process?

Basically, when you have a several tasks to execute in parallel use the ```Pool```. On the other hand,  when you have a small number of tasks to execute in parallel and you only need each task done once use the ```Process```. For example, if I would like to run forecasting for 100 different indicators for 5 different countries, I could use both assigning to each ```Process``` each country and then the forecast task of each indicator to the ```Pool```. 

##### F.2 How to implement it?

The class ```parallel_process``` should be provided with three arguments to initialize it:

  * function: The function that you would like to implement in parallel
  * func_args: The arguments of the function
  * elements: List containing the elements to which you would like to implement in parallel the function. Considering the above example elements could be a list with the different countries or the 100 indicators.
  
If you would like to implement both the ```Pool``` and the ```Process``` at the same time, like in the example of countries and indicators, you should initialize two classes because for the moment it does not support to call both at the same time.  Finally, either you call the function ```pp_Queue``` or ```pp_Pool``` it would return as a tuple two objects: ```res``` a list with the results of the tasks and the time it took the whole process to run. 


## G. Set up a virtual environment with R and Python 

To set up a virtual environment with Jupyter Notebook and the required R packages and Python libraries, see the `README.md` file under the folder `setup`. 

***References (Books)***

[1] Scikit-Learn Ensemble. URL: https://scikit-learn.org/stable/modules/ensemble.html

[2] XGBoost. URL:https://xgboost.readthedocs.io/en/latest/index.html

[3] Friedman, Jerome; Hastie, Trevor; Tibshirani, Robert. The elements of Statistical Learning. Springer, 2008.

[4] Hansen, Bruce. Advanced Time Series and Forecasting Lecture 5 Structural Breaks. The University of Wisconsin, 2012.

[5] Rousseeuyy, Peter; Leroy, Annick. Robust Regression and Outlier Detection. John Wiley & Sons, 1987.

[6] Stackoverflow. https://stats.stackexchange.com/questions/289349/why-do-we-study-the-noise-sequence-in-time-series-analysis 

[7] Quantstart. https://www.quantstart.com/articles/White-Noise-and-Random-Walks-in-Time-Series-Analysis/

[8] Mane, Priyanka. Python Multiprocessing: Pool vs Process – Comparative Analysis. URL: https://www.ellicium.com/python-multiprocessing-pool-process/

[9] Stats-Stackexchange. URL: https://stats.stackexchange.com/questions/263366/interpreting-seasonality-in-acf-and-pacf-plots 

[10] Hyndman, Rob; Athanasopoulos, George. Forecasting: Principles and Practice. O Texts, 2018. 
