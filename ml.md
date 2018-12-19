# Machine Learning

## Baeysian Statistics


### How would you use Naive Bayes classifier for categorical features? What if some features are numerical?

Naive Bayes:
The first assumption in the NB approach is that the features are independent. 
Then use Bayes theorem to determine the probability.
For numerical data:
A common technique in NBC is to recode the feature (variable) values into quartiles, such that values less than the 25th percentile are assigned a 1, 25th to 50th a 2, 50th to 75th a 3 and greater than the 75th percentile a 4. Thus a single object will deposit one count in bin Q1, Q2, Q3, or Q4. Calculations are merely done on these categorical bins. Bin counts (probabilities) are then based on the number of samples whose variable values fall within a given bin.


### Is Naïve Bayes bad? If yes, under what aspects.

Naive Bayes
	Pros
		Computationally fast
		Simple to implement
		Works well with high dimensions
	Cons
		Relies on independence assumption and will perform badly if this assumption is not met
https://github.com/ctufts/Cheat_Sheets/wiki/Classification-Model-Pros-and-Cons



### What do you understand by conjugate-prior with respect to Naïve Bayes?

Posterior = Likelihood * prior/denominator 
If prior is taken from a normal dist, posterior is also a nornal dist
more on this https://www.youtube.com/watch?v=aPNrhR0dFi8


## Bias Variance Tradeoff

### Bootstrapping - how and why it is used?

The bootstrap method is a resampling technique used to estimate statistics on a population by sampling a dataset with replacement.
It can be used to estimate summary statistics such as the mean or standard deviation.


### Define Bias and Variance.

Bias refers to the error that is introduced by approximating a real-life problem, which may be extremely complicated, by a much simpler model. So, if the true relation is complex and you try to use linear regression, then it will undoubtedly result in some bias in the estimation of f(X).

Variance refers to the amount by which your estimate of f(X) would change if we estimated it using a diﬀerent training data set. Since the training data is used to ﬁt the statistical learning method, diﬀerent training data sets will result in a different estimation.

A general rule is that, as a statistical method tries to match data points more closely or when a more flexible method is used, the bias reduces, but variance increases.


### How does the variance of the error term change with the number of predictors, in OLS?

Y = f(X) + e
Here we can estimate f(x) to the best and the error associated with it is called reducible error. While e here is the irreducible error.
We can reduce the bias by choosing complex models and including more number of predictors. But that increases the variance of the model while reducing the bias.

### What is overfitting and how to reduce them?

As we make the model more complex, the bias reduces but the model closely learns from the training data thereby overfitting the same. 
Steps to reduce overfitting:
	- Regularization: Adding a penalty term for addition of features. Ridge and Lasso regularization, Pruning in case of trees, dropouts on NNs
	- Using more training data
	- Cross validation: Preventive against overfitting.
	- Ensembling: Boosting and bagging techniques
	- Removing features: PCA and variable selection techniques

## Linear Regression

### Why is R2 horrible for determining the quality of a model and name at least two better metrics. 

R-squared is a statistic that often accompanies regression output. It ranges in value from 0 to 1 and is usually interpreted as summarizing the percent of variation in the response that the regression model explains. So an R-squared of 0.65 might mean that the model explains about 65% of the variation in our dependent variable. 

other metrics
http://www.sthda.com/english/articles/38-regression-model-validation/158-regression-model-accuracy-metrics-r-square-aic-bic-cp-and-more/

One pitfall of R-squared is that it can only increase as predictors are added to the regression model.

To remedy this: 

Adjusted R-squared, incorporates the model’s degrees of freedom. Adjusted R-squared will decrease as predictors are added if the increase in model fit does not make up for the loss of degrees of freedom. Likewise, it will increase as predictors are added if the increase in model fit is worthwhile. Adjusted R-squared should always be used with models with more than one predictor variable. It is interpreted as the proportion of total variance that is explained by the model.

There are situations in which a high R-squared is not necessary or relevant. When the interest is in the relationship between variables, not in prediction, the R-square is less important

Additionally, there are four other important metrics - AIC, AICc, BIC and Mallows Cp - that are commonly used for model evaluation and selection. These are an unbiased estimate of the model prediction error MSE. The lower these metrics, he better the model.

AIC stands for (Akaike’s Information Criteria) The basic idea of AIC is to penalize the inclusion of additional variables to a model. It adds a penalty that increases the error when including additional terms. The lower the AIC, the better the model.
BIC (or Bayesian information criteria) is a variant of AIC with a stronger penalty for including additional variables to the model.
Mallows Cp: A variant of AIC developed by Colin Mallows.


### Why is linear regression called linear?

Linear regression is called linear because you model your output variable (lets call it f(x)) as a linear combination of inputs and weights (lets call them x and w respectively). 


### What are the assumptions that standard linear regression models with standard estimation techniques make? How can some of these assumptions be relaxed?

Linear regression is an analysis that assesses whether one or more predictor variables explain the dependent (criterion) variable.  The regression has five key assumptions:

Linear relationship: First, linear regression needs the relationship between the independent and dependent variables to be linear.  It is also important to check for outliers since linear regression is sensitive to outlier effects.  The linearity assumption can best be tested with scatter plots.

Multivariate normality: This assumption can best be checked with a histogram or a Q-Q-Plot.  Normality can be checked with a goodness of fit test, e.g., the Kolmogorov-Smirnov test.  When the data is not normally distributed a non-linear transformation (e.g., log-transformation) might fix this issue.

No or little multicollinearity : Multicollinearity occurs when the independent variables are too highly correlated with each other. Multicollinearity may be tested with three central criteria:

	1) Correlation matrix – when computing the matrix of Pearson’s Bivariate Correlation among all independent variables the correlation coefficients need to be smaller than 1.

	2) Tolerance – the tolerance measures the influence of one independent variable on all other independent variables; the tolerance is calculated with an initial linear regression analysis.  Tolerance is defined as T = 1 – R² for these first step regression analysis.  With T < 0.1 there might be multicollinearity in the data and with T < 0.01 there certainly is.

	3) Variance Inflation Factor (VIF) – the variance inflation factor of the linear regression is defined as VIF = 1/T. With VIF > 10 there is an indication that multicollinearity may be present; with VIF > 100 there is certainly multicollinearity among the variables.
The simplest way to address the problem is to remove independent variables with high VIF values.

No auto-correlation:  Autocorrelation occurs when the residuals are not independent from each other.  For instance, this typically occurs in stock prices, where the price is not independent from the previous price. While a scatterplot allows you to check for autocorrelations, you can test the linear regression model for autocorrelation with the Durbin-Watson test.  
If the error terms are correlated, the estimated standard errors tend to underestimate the true standard error.

Homoscedasticity: The scatter plot is good way to check whether the data are homoscedastic (meaning the residuals are equal across the regression line). The Goldfeld-Quandt Test can also be used to test for heteroscedasticity.
https://www.analyticsvidhya.com/blog/2016/07/deeper-regression-analysis-assumptions-plots-solutions/

### What do the following parts of a linear regression signify? p-value, coefficient, R-Squared value. 

R-square value tells you how much variation is explained by your model. The greater R-square the better the model.
P-value is the "probability" attached to the likelihood of getting your data results (or those more extreme) for the model you have. It is attached to the F statistic that tests the overall explanatory power for a model based on that data (or data more extreme).
Regression coefficients represent the mean change in the response variable for one unit of change in the predictor variable while holding other predictors in the model constant.

### Could you explain some of the extension of linear models like Splines or LOESS/LOWESS?

- Regression Splines use a combination of linear/polynomial functions to fit the data. 
- Polynomial regression has a tendency to drastically over-fit, even on this simple one dimensional data set. In order to overcome the disadvantages of polynomial regression, we can use an improved regression technique which, instead of building one model for the entire dataset, divides the dataset into multiple bins and fits each bin with a separate model.
- The points where the division occurs are called Knots. Functions which we can use for modelling each piece/bin are known as Piecewise functions.
- To maintain the continuity of the piecewise polynomoial functions along various knots, first/second derivative constraints can be imposed. These are cubic splines. 
- To check the erratic behaviour of splines near the boundaries, Natural splines can be considered which adds a linear constraint near the boundaries to reduce the variance.

### In linear regression, under what condition R^2 always equals a perfect 1?

Mathematically R square = 1 - (SSE/SST)
Where SSE is the sum of squared errors of our regression model
And SST is the sum of squared errors of our baseline model.

If SSE is equal to 0, i,e the model perfectly fits the data, then SSE=0 , making the R square perfectly equal to 1

