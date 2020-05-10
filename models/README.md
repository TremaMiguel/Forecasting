> NOTE. To visualize the TeX code, activate the [MathJax](https://chrome.google.com/webstore/detail/mathjax-plugin-for-github/ioemnmodlmafdkllaclgeombjnmnbima) plugin for Github.

## A. Exponential Smoothing Models

This methods produce forecasts by weighting average of past observations, this weights decay exponentially as the observations get older. To select between `Simple Exponential Smoothing`, `Holt` or `Holt-Winters` one must be able to recognise the component of the time series (see section D) and the way this components interact with the smoothing method (additive, damped or multiplicative). the image below shows the appropiate model selection based on these conditions, image is taken from chapter 7 of [2].

![ES_models](figs/Classification_ES_models.png)

> **A.1 Simple Exponential Smoothing**

When there is no clear trend or seasonality this model is ideal. It works under assumption of giving less weigth to distant observations through a parameter $\alpha \in [0,1]$ and the following expression

$$\hat{y}_{t+1|t} = \alpha y_{t} + \alpha(1-\alpha)y_{t-1} + \alpha(1-\alpha)^2y_{t-2} + ...,$$

notice the decay in the coefficient associated to each $y_{t-k}$, the more distant the observation the larger the value $\alpha(1-\alpha)^k$, thus, the less weight the value $y_{t-k}$ takes.

One can choose the right $\alpha$ value by minimizng the `SSE` or sum of squared residuals 

$$SSE = \sum_{t=1}^{L}(y_{t} - \hat{y}_{t|t-1})^2.$$

> **A.2 Holt Linear Trend Method**

To forecast data with trend, besides the forecast equation 

$$\hat{y}_{t+h|t} = l_{t} + hb_{t}, $$

we now include two smoothing equations

$$\text{Level equation} \:\: l_{t} = \alpha y_{t} + (1-\alpha)(l_{t-1} + b_{t-1}),$$

$$\text{Trend equation} \:\: b_{t} = \beta^* (l_{t} - l_{t-1}) + (1-\beta^*)b_{t-1},$$

where $\alpha \in [0,1]$ is the smoothing parameter for the level
and $\beta^* \in [0,1]$ is the smoothing parameter for the trend. Notice that the level equation is weighted average between the past observation $y_{t}$ and the first equation. By contrast, the trend equation is a weighted average between the difference in the level equation and the previous trend equation. 

In addition, one can also **dampen the trend** to a flat line. To control this behavior the damping parameter $\phi \in [0,1]$ is included. Thus,  

$$\hat{y}_{t+h|t} = l_{t} + (\phi + \phi^2 + ... + \phi^h)b_{t}, $$

$$l_{t} = \alpha y_{t} + (1-\alpha)(l_{t-1} + \phi b_{t-1}),$$

$$b_{t} = \beta^* (l_{t} - l_{t-1}) + (1-\beta^*)\phi b_{t-1},$$

one can easily see that the dumping parameter is only affecting the trend component. So, an effective damped model would consider $0.8\leq \phi < 1$ because $\phi$ has a strong effect for smaller values. 

> **A.3 Holt-Winters Seasonal method**

This method extends the  Holt model to capture seasonality by additionally incorporating a smoothing equation for the seasonal component. Depending on the nature of the seasonal component there are two variations of this method. When the seasonal variation is changing proportional to the level of the series the `multiplicative method` is choosen. Whereas, when the seasonal variations are constant through the series the `additive method` is the choice. Let's see this two methods in more detail, below the `Holt-Winters additive method`

$$\hat{y}_{t+h|t} = l_{t} + hb_{t} + s_{t+h-m(k+1)}, $$

$$l_{t} = \alpha(y_{t} - s_{t-m}) + (1-\alpha)(l_{t-1} + b_{t-1}),$$

$$b_{t} = \beta^* (l_{t} - l_{t-1}) + (1-\beta^*)b_{t-1},$$

$$s_{t} = \gamma(y_{t}-l_{t-1}-b_{t-1}) + (1-\gamma)s_{t-m},$$

where $m$ denotes the frequency of the seasonality ($m$=12 for monthly data, 
$m$=4 for quarterly, $m$=52 for weekly data and $m=7$ for daily data) and $\gamma \in [0, 1-\alpha]$. Note that the seasonal equation is a weighted average between the current season and the same season of $m$ periods ago. In the multiplicative case, the series is seasonally adjusted by dividing through the seasonal component. 

$$\hat{y}_{t+h|t} = (l_{t} + hb_{t}) s_{t+h-m(k+1)}, $$

$$l_{t} = \alpha\dfrac{y_{t}}{s_{t-m}} + (1-\alpha)(l_{t-1} + b_{t-1}),$$

$$b_{t} = \beta^* (l_{t} - l_{t-1}) + (1-\beta^*)b_{t-1},$$

$$s_{t} = \gamma\dfrac{y_{t}}{l_{t-1}-b_{t-1}} + (1-\gamma)s_{t-m}.$$

It is also possible to consider damped versions of this models, check out [2] for more details. A summary of this models is provided in the image below, again taken from [2]

![Trend_Seasonal](figs/Trend_and_Seasonal.png)

Finally, the models `sem`, `holt` and `holt-winters` under the `AutoRegressiveModels.py` file use a grid search to fit the best parameters to each model.  

***

## B. Understanding the ARIMA model

$ARIMA$ stands for autoregressive integrated moving average. But, what does it mean autoregressive or moving average? Basically, it depends on the term that you're considering to forecast the variable of interest. For example, if you're considering the prior observation $y_{t-1}, y_{t-2}, ...$ it is autoregressive and if you're taking into account the forecast errors $\epsilon_{t}, \epsilon_{t-1},...$ it is moving average. 

> **B.1 The autoregressive part**

Autoregression means a regression of the variable against itself, in other words, we forecast the variable of interest $y_{t}$ by taking a linear combination of past values 

$$y_{t} = c + \phi_{1}y_{t-1} + \phi_{2}y_{t-2} + ... + \phi_{p}y_{t-p} + \epsilon_{t},$$

where $c$ is a constant and $\epsilon_{t}$ is white noise. We can refer to this model as an autoregressive model of order $p$ or $AR(p)$ model.

> **B.2 The Movingaverage part**

If we now consider the prior errors of the forecast we have 

$$y_{t} = c + \epsilon_{t} + \alpha_{1}\epsilon_{t-1} + ... + \alpha_{q}\epsilon_{t-q},$$

where $\epsilon_{t}$ is white noise. We can refer to this model as moving average model of order $q$ or $AR(q)$ model.

> **B.3 The non-seasonal $ARIMA$ model**

An $ARIMA$ model is expressed as 

$$y_{t}' = c + \phi_{1}y_{t-1}' + ... + \phi_{p}y_{t-p}' + \alpha_{1}\epsilon_{t-1} + ... + \alpha_{q}\epsilon_{t-q} + \epsilon_{t},$$

where $y_{t}'$ is the degree of differencing in the series, if it is one time is $y_{t}'$ and if it is second time is $y_{t}''$. Therefore when combining the autoregression, the moving average and the differencing we obtain a non-seasonal $ARIMA(p,d,q)$ model, where $p$ is the order or the autoregressive part, $d$ the degree of first differencing and $q$ the order of the moving average part.

***

## C. Count Data Models

The prior models consider that the sample space of the data is continuous. But, data could contain small counts $\{0,1,2,3,..\}$ and often a high proportion of zeros. This models are most commonly known as **Intermittent** and are used to forecast the demand of products with several periods of zero demand. 

> **C.1 Ad-hoc models**

**C.1.1 Croston Model**

The time series is constructed by noting which time periods contain zero values and which non-zero values. The `demand` is the $y$ component (the non-zero quantity) and the `inter-arrival time` the $a$ component (the time between $q_{i-1}$ and $q_{i}$). Thus, Croston method implies exponential smoothing forecasts for this components 

$$\hat{y}_{i+1|i} = (1-\alpha)\hat{y}_{i|i-1} + \alpha y_{i}$$ 

$$\hat{a}_{i+1|i} = (1-\alpha)\hat{a}_{i|i-1} + \alpha a_{i},
$$

where $\alpha \in [0,1]$ and $\hat{q}_{i+1|i}$, $\hat{a}_{i+1|i}$ denote the one step forecast of the components respectively. So, the $h-step$ forecast for the demand is given by the ratio 

$$\hat{Y}_{T+h|T} = \dfrac{\hat{y}_{j+1|j}}{\hat{a}_{j+1|j}},$$

where $j$ is the time of the last observed positive observation. Thus, basically the Croston method considers what would the next value be with the $q$ component (expected amount of the items sold) and what would be the interval time of zero values wiht the component $a$ (the time lag between two periods of consecutive demand). In this way, by taking the quotient one is "smoothing" the demand by taking into account that they are peridos of zero values. For a detail about the expectation and the variance of the forecast, see [1].

There are some issues with this method, for example, updating the forecast only in periods of positive demand has a negative effect for periods of highly intermittent demand because this lead to long periods without updating the parameters. 

**C.1.2 Modified Croston Models**

**C.1.2.1 Syntetos and Boylan**

In 2001 it was proven by Syntetos and Boylan that the estimator $\hat{y}_{T+h|T}$ is positively bias, that is, the forecast is greater than the actual demand. To correct for this bias, consider 

$$\hat{Y}_{T+h|T} = \bigg(1-\dfrac{\alpha}{2}\bigg)\dfrac{\hat{y}_{j+1|j}}{\hat{a}_{j+1|j}},$$

where $\alpha$ is as before. We notice that this modification reduces the forecast between 0.5 ($\alpha = 1$) and 1 ($\alpha = 0$). 

**C.1.2.2  TSB method**

Instead of updating the demand interval, we now consider updating a demand probability element $\pi$, we'll later see why this is an advantage. The model formulation is 

$$\hat{y}_{i+1|i} = (1-\alpha)\hat{y}_{i|i-1} + \alpha y_{i}$$

$$\hat{\pi}_{i+1|i} = (1-\beta)\hat{\pi}_{i|i-1} + \beta x_{i},$$

where 

$$
x_{i} =  \begin{cases} 
      0 & \text{if} \: y_{i} =  0 \\
      1 & \text{in any other case}
   \end{cases}
$$

is an indicator of a non-zero demand at time $i$ and $\hat{\pi}_{i+1|i}$ is the probability of positive demand. Now the $h-$ step forecast is given by the formula 

$$\hat{y}_{T+h|T} = \hat{y}_{i+1|i}\hat{\pi}_{i+1|i}.$$

Notice that the parameters $\alpha$ and $\beta$ intent to smooth demand size and demand probability respectively. 

The key advantage regarding the Croston model is the possibility to update the parameters in every period, therefore, the estimate of the demand probability approaches zeros when there is a long period without demand.  

**C.1.2.3  HES method**

Hyperbolic Exponential Smoothing is an hybrid of Croston's method with a Bayesian approach. Compare to the $TSB$, it decays hyperbolically over a period of zeros (obsolescence). Also, it is unbiases on non-intermittent and stochastic intermittent demand according to the authors, see [14].

$$Y_{i|i-1} = \dfrac{\hat{y}_{i|i-1}}{\hat{a}_{i|i-1} + \beta (T_{i|i-1} / 2)},$$

where $T$ is the current number of periods since a demand was last seen and $\beta$ smooths the interval length.  

> **C.2 Model based models**

This approach considers using statistical models, particularly take an autoregressive and moving average point of view to model intermittent data. 

**C.2.1 INARMA model**

The `INARMA` process stands the idea that the current demand in period $t$ comes from the demand of the prior periods (lingering demand) and an [innovation process](https://en.wikipedia.org/wiki/Innovation_(signal_processing)) (an error). An $INARMA(p,q)$ process is formulated as 

$$Y_{t} = \sum_{i=1}^{p}\alpha_{i}\circ Y_{t-i} + \epsilon_{t} + \sum_{i=1}^{q}\beta_{i}\epsilon_{t-i},$$

where $\epsilon_{t}$ is the error term and $\circ$ denotes the binomial thinning operator

$$\alpha \circ Y = \sum_{i=1}^{Y}B_{i}$$

$$B_{i} \sim Ber(\alpha),$$

where $\alpha \in [0,1]$. Notice that $\alpha \circ Y$ follows a `binomial distribution`. To gain further insight about parameter estimation see [13]. $INARMA$ is suitable for the distribution of the demand during the lead time. Thus, they are helpful for defining an optimal inventory policy. 


> **C.3 Regression models**


**C.3.1 Poisson Regression**

This distribution is useful if we could satisfy the assumption of [equidispersion](http://www.eco.uc3m.es/~ricmora/miccua/materials/S17T33_Spanish_handout.pdf). Suppose that the variable of interest $Y$ follows a Poisson distribution then

$$Pr(Y=y|\lambda) = \dfrac{e^{-\lambda}\lambda^y}{y!}.$$

Poisson regression is not suitable when we have unusual observations or excess of zeros because [overdispersion](https://en.wikipedia.org/wiki/Overdispersion) could be present. Overdispersion happens when there is greater variability than expected. 

**C.3.2 Negative Binomial Regression**

As we have mentioned before Poisson Regression fails because it doesn't consider the `heterogeneity` in the data, that is, there are unobserved factors that leverage the variability related with the response variable $Y$. One way to pass through this is the `Negative Binomial` distribution 

$$f(y;\mu,\theta) = \dfrac{\Gamma(y+\theta)}{\Gamma(\theta)y!}\dfrac{\mu^{y}\theta^{\theta}}{(\mu+\theta)^{y+\theta}},$$

where $\mu$ is the mean and $\theta$ is the parameter of overdispersion. One alternative to deal with the overdispesion in the standar Poisson model is to estimate $\theta$ with the data, this approach is known as `Quasi-Poisson` regression. 

**C.3.3 Zero Inflated Regression**

When the presence of zeros is significant, the prior models fail. Thus, we consider a Zero Inflated model which assumes that the zeros in the data come from two distinct processes. One comes from a Poisson, Geometric or Negative Binomial distribution (`count component`) and the other are `point mass zeros`.

$$Pr(y_{i}=0) = g_{i} + (1-g_{i})f(0),$$

where $f$ is a probability function. 

**C.3.3 Hurdle Regression**

In this model there is a `binary decision` between zero or non-zero and other component that determines the values greater than zero when the `Hurdle` (zero or non-zero) is cross


$$f_{\text{hurdle}}(y;x,z,\beta,\gamma) = \begin{cases} 
      y=0 & f_{\text{zero}(0;z,\gamma)} \\
      y > 0 & 1-\dfrac{f_{zero}(0;z,\gamma)f_{\text{count}}(y;x,\beta)}{1 - f_{\text{count}(0;x,\beta)}} 
   \end{cases}$$

**C.3.4 Poisson-Tweedie Regression**

The $PT$ have a great flexibility because they allow us to have a good behavior with data that have a great presence of zeros, overdispersion and extreme observations that make the distribution to be heavy tailed. A variable $Y \sim PT(a,b,c)$ with probability function 

$$F_{Y}(y|a,b,c) = \exp \bigg\{\dfrac{b}{a}[(1-c)^{a} - (1-cy)^{a}]\bigg\}, $$

when $a\neq 0$. In other case, a limit should be calculated, see [16].


> **C.4 Evaluation Metrics**

In normal Time Series, it is usual to consider the $RMSE$ or the $MAE$ to assess the fit of the model. However, for intermittent demand they tend to bias forecast in favour of zero demand forecast. In other words, by considering these measures it is easier to minimize its values by forecasting zero values. To overcome this, one considers `scale performance measures`, let's consider the scaled error

$$\epsilon_{t} = \dfrac{y_{t} - \hat{y_{t}}}{\frac{1}{T}\sum_{t=2}^{T}|y_{t} - y_{t-1}|},$$

which can be considered for the measures

$$
\text{MASE} = mean(|\epsilon_{t}|), \\
$$

$$
\text{MSSE} = mean(\epsilon_{t}^2), \\
$$

$$
\text{MdASE} = median(|\epsilon_{t}|), \\
$$

$$
\text{MdSSE} = median(\epsilon_{t}^2), \\
$$

where $MASE$ stands for mean absolute scaled error, $MSSE$ for Mean Squared Scaled Error and $Md$ refers to the median like Median Absolute Scaled Error. $MASE$ is recommended as metric for intermittent demand. 

---

## D. Decision Tree Models

First, the data is transformed with the function ```window_slide```. This is done, in order to be able to forecast when calling the ```predict``` method of each model, that is, for constructing the variable $y_{t+1}$ we consider n past observations $y_{t+1} = y_{t} + y_{t-1} + ... + y_{t-n+1}$. Then, call the desired model (```'rfr'``` for RandomForest, ```'gbr'``` for GradientBoosting, ```'adr'``` for AdaBoost and ```'xgbr'``` for XG-Boost) with the function ```tree_model```. Finally, the parameters of each model where choosen according to [3].




---

## E. Neural Networks

Currently it implements a stacked LSTM. To specify an LSTM specify `model == 'Stacked-LSTM'` when calling the function `nn_model` of the class `NeuralNetworks`. Furthermore, you can specify the number of stacked layers and units by setting the integer parameters `layers` and `units` respectively, for example, if you set `layers=4` and `units=8` each layer will have (units / number of hidden layer) as units.  



## References

[1]: Engelmeyer, Torben. Managing Intermittent Demand. Springer, 2016. 

[2]: Hyndman, Rob; Athanasopoulos, George. Forecasting: Principles and Practice. O Texts, 2018. 

[3]: Peña Carrasco, Manuel. Modelización de conteos mediante la
distribución Poisson-Tweedie (PT):
aplicación en datos de
ultrasecuenciación. [URL](https://ddd.uab.cat/pub/tfg/2013/125799/TFG_ManuelCarrasco.pdf)

[4]: Prestwich, S.D.; Tarim, S.A.; Rossi, R; Hnich, B. Forecasting Intermittent Demand by Hyperbolic-Exponential Smoothing. [URL](https://arxiv.org/pdf/1307.6102.pdf)

[5]: Waller, Daniel. Methods for Intermittent Demand Forecasting. [URL](https://www.lancaster.ac.uk/pg/waller/pdfs/Intermittent_Demand_Forecasting.pdf)

[6]: Zivot, Eric. Notes on Unit Root Tests. URL: https://faculty.washington.edu/ezivot/econ584/notes/unitroot.pdf

[7]: Zivot, Eric. Notes on Forecasting. URL: https://faculty.washington.edu/ezivot/econ584/notes/forecasting.pdf

