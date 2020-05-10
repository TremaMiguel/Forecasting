## Preprocess Data (Functions)

### 1. Transform Data (`pp_transforms()`)

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


### 2. Test Data (`pp_tests()`)

Implements an augmented `Dicker-Fuller test` (unit root) or a `KPSS-test` in order to determine which type of model to apply between `Simple Exponential Moving Average`, `Holt`, `Holt-Winters Additive` or `Seasonal Arima` from the `statsmodels` library. 

Furthermore, you can compare the forecast performace of two models through the function `diebold_mariano`.

### 3. Process Data (`pp_functions()`)

Through the ```rpy2``` library we call an R enviroment to use distinct statistical packages as ```strucchange```, ```TSA```, ```zoo```, ```tsoutliers```, ```pracma```, ```imputeTS``` or ```forecast```.

##### 3.1. Interpolation.

It implements different types of interpolation such as: `NA`, `Kalman`, `Moving Average`, `Seasonal Decompose`, `Seasonal Splitted` or `StructTS`. In general, these methods are used to replace NA values in a time series. 

##### 3.2. Outlier Detection and Replacement.

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


##### 3.3. Structural Changes

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

##### 3.4. Outlier or Structural Change
 
The kind of perturbation that shifts cause on the observed time series can be classified as an outlier, when the shift affects the noise component, or as a structural change, when the shift affects one of the signal components.

## Parallel Computation (`preprocess_multiprocessing`)

```Pool``` and ```Process``` are two ways of executing tasks parallelly, but in a different way. It works by mapping the input to the different processors, then distributes the processes among the avaible cores in ```FIFO``` manner. Finally, it waits for all the tasks to finish to return an output. One particularity is that the processes in execution are stored in memory. By contrast, the ```Process``` class assign all the processes in memory and schedules execution using FIFO policy. When the process ends it schedules a new one for execution. I suggest to read [8] and the official documentation of the Multiprocessing library for better comprehension.

##### When to use Pool or Process?

Basically, when you have a several tasks to execute in parallel use the ```Pool```. On the other hand,  when you have a small number of tasks to execute in parallel and you only need each task done once use the ```Process```. For example, if I would like to run forecasting for 100 different indicators for 5 different countries, I could use both assigning to each ```Process``` each country and then the forecast task of each indicator to the ```Pool```. 

##### How to implement it?

The class ```parallel_process``` should be provided with three arguments to initialize it:

  * function: The function that you would like to implement in parallel
  * func_args: The arguments of the function
  * elements: List containing the elements to which you would like to implement in parallel the function. Considering the above example elements could be a list with the different countries or the 100 indicators.
  
If you would like to implement both the ```Pool``` and the ```Process``` at the same time, like in the example of countries and indicators, you should initialize two classes because for the moment it does not support to call both at the same time.  Finally, either you call the function ```pp_Queue``` or ```pp_Pool``` it would return as a tuple two objects: ```res``` a list with the results of the tasks and the time it took the whole process to run. 
