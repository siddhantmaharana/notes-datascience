# Statistics

## Definitions and theory

- Definition of Population, Sample, Mean, Median, Variance, SD, Standard Error
- T test(One sample):
	- Test whether a mean of a normally distributed population is different from a specified value
	- Null hypothesis states that the population mean is equal to the value specified
	- T-statistic= xbar minus mu divided by std error. standardizes the mean 
	- Check corresponding p value for some degree of freedom (determined from sample size). If p value is less than the significant value set, then reject the null hypothesis.
- Two Sample T test:
	- Tests whether the means of the two population are significantly different from each other or not.
	- Paired: Each value in the group directly corresponds to some value in another group.
	- Unpaired: The two populations are independent.
- Chi Squared Fitness of test:
	- Checks whether or not an observed pattern of data fits some given distribution.
- Chi Squared to measure independence of categories
	- Checks whether two categorical variables are related or not
	- Calculate Expected value- (rowtotal* column total)/grand total
	- Calculate Chi square. If p less than threshold, then reject Ho( categories are independent )
- Hypothesis testing: It is a way of systematically quantifying how certain you are of the result of a certain statistical test
- Confidence Interval: It is the level at which you reject the null hypothesis. If there's a 95% chance that there is real difference in your observations given the null hypothesis, then you are confident in rejecting this.
 
http://web.mit.edu/~csvoss/Public/usabo/stats_handout.pdf

## Basic Stats

### Are expected value and mean value different?

The expected value is numerically the same as the average value, but it is a prediction for a specific future occurrence rather than a generalization across multiple occurrences.

The concept of expectation value or expected value may be understood from the following example. Let X represent the outcome of a roll of an unbiased six-sided die. The possible values for X are 1, 2, 3, 4, 5, and 6, each having the probability of occurrence of 1/6. The expectation value (or expected value) of X is then given by

(X)expected=1(1/6)+2⋅(1/6)+3⋅(1/6)+4⋅(1/6)+5⋅(1/6)+6⋅(1/6)=21/6=3.5
Suppose that in a sequence of ten rolls of the die, if the outcomes are 5, 2, 6, 2, 2, 1, 2, 3, 6, 1, then the average (arithmetic mean) of the results is given by

(X)average=(5+2+6+2+2+1+2+3+6+1)/10=3.0

We say that the average value is 3.0, with the distance of 0.5 from the expectation value of 3.5. If we roll the die N times, where N is very large, then the average will converge to the expected value, i.e.,(X)average=(X)expected. This is evidently because, when N is very large each possible value of X (i.e. 1 to 6) will occur with equal probability of 1/6, turning the average to the expectation value.


### Do you know what Type-I/Type-II errors are?

Type I:

	- In statistical hypothesis testing, a type I error is the rejection of a true null hypothesis (also known as a "false positive" finding)
	- To lower this risk, you must use a lower value for α.  "Setting it lower" means you need stronger evidence against the null hypothesis

Type II:

	- While a type II error is failing to reject a false null hypothesis (also known as a "false negative" finding).
	- To reduce the probability of a type 2 error (because the consequences could be severe as well), you can either increase the sample size or choose an alternative value of the parameter in question that is further from the null value.


### What are p-values and confidence intervals?

p value:

	- The P value, or calculated probability, is the probability of finding the observed, or more extreme, results when the null hypothesis (H0) of a study question is true.
	- Eg: p valus of 0.18 says that there is 18% prob that the mean obtained is not equal to that claimed mean.
	- A small p-value (typically ≤ 0.05) indicates strong evidence against the null hypothesis, so you reject the null hypothesis. A large p-value (> 0.05) indicates weak evidence against the null hypothesis, so you fail to reject the null hypothesis.
	- You can use either P values or confidence intervals to determine whether your results are statistically significant. If a hypothesis test produces both, these results will agree.

Confidence Intervals:

	- A confidence interval is a range of values that is likely to contain an unknown population parameter. 
	- The confidence level is equivalent to 1 – the alpha level. So, if your significance level is 0.05, the corresponding confidence level is 95%.

		- If the P value is less than your significance (alpha) level, the hypothesis test is statistically significant.
		- If the confidence interval does not contain the null hypothesis value, the results are statistically significant.
		- If the P value is less than alpha, the confidence interval will not contain the null hypothesis value.

### What do you do when n is small? How do you quantify uncertainty? Pick one strategy and explain how to make decisions under uncertainty?

For comaparing means, two sample t-test can perform accurate. However for comparing variance, the Fisher Exact Test tends to perform better.

Using confidence intervals provides a better method for estimating the unknown population average.


### What is the Central Limit Theorem and why is it important in data science?

Let’s say you are studying the population of beer drinkers in the US. You’d like to understand the mean age of those people but you don’t have time to survey the entire US population.

Instead of surveying the whole population, you collect one sample of 100 beer drinkers in the US. With this data, you are able to calculate an arithmetic mean. 
The sampling distribution is the distribution of the samples mean.
The statement of the theorem says that the sampling distribution, the distribution of the samples mean you collected, will approximately take the shape of a bell curve around the population mean. This shape is also known as a normal distribution. 

CLT is essential to make statistical assumptions about the data. Concepts of confidence intervals and hypothesis testing are based on CLT.

### What is the distribution of p-value’s, in general?

 P-values under the null are uniformly distributed.


### What is the normal distribution? Give an example of some variable that follows this distribution

A normal distribution has a bell-shaped density curve described by its mean mu and standard deviation s  . The density curve is symmetrical, centered about its mean, with its spread determined by its standard deviation. 
The Standard Normal curve, has mean 0 and standard deviation 1. 


### What is t-Test/F-Test/ANOVA? When to use it?

The F-test is designed to test if two population variances are equal. It does this by comparing the ratio of two variances. So, if the variances are equal, the ratio of the variances will be 1. 

Analysis of Variance (ANOVA) is a statistical method used to test differences between two or more means.
Eg: Mean IQ scores for 3 different schools (m= groups, n = data points in each grp)

- Calcualate total sum of square(SST): summation(val - grand mean)^2; df = mn-1
- Sum of squares within (ssw) = summation(val-grp mean)^2 ;df = m(n-1)
- Sum of square between (SSB) = summation(grp mean - grand mean)^2 ; df = m-1
- SST = SSW + SSB
- Null Hyp - School doesn't make a difference
- F statistic = Ratio of two chi square distributions with different degrees of freedom (SSB/m-1)/(SSW/m(n-1))
- If numerator is much more it tells us that the variation is due to variation between them than within them. If the denom is larger suggesting that the variation comes mostly from the variation within the group making it difficult to reject H0
- Calculate the critical f value or p value and then conclude.

### What summary statistics do you know?

Measure of Central tendency: 

	- Mean, median, mode
	- mean-mode = 3(mean -median)

Measure of spread: 

	- range, IQR, SD
	- quartiles
	- skewness - right skewed when mean>median>mode
	- skewness tells you the amount and direction of skew 
	- kurtosis tells you how tall and sharp the central peak is, relative to a standard bell curve.
	- skewness: g1 = m3 / m2^ 3/2
	 where m3 = ∑(x−x̅)3 / n   and   m2 = ∑(x−x̅)2 / n
	 x̅ is the mean and n is the sample size, as usual. m3 is called the third moment of the data set. m2 is the variance, the square of the standard deviation.
 	

### How would you calculate needed sample size?

- Need to determine confidence level and margin of error(5 or 10%)
- Can calculate z score from it.

Necessary Sample Size = (Z-score)2 * StdDev * (1-StdDev) / (margin of error)2


### how would you calculate the degrees of freedom of an interaction

The degrees of freedom in a statistical calculation represent how many values involved in a calculation have the freedom to vary.
The degrees of freedom can be calculated to help ensure the statistical validity of chi-square tests, t-tests and even the more advanced f-tests. These tests are commonly used to compare observed data with data that would be expected to be obtained according to a specific hypothesis.


### How would you find the median of a very large dataset?


If the values are discrete and the number of distinct values isn't too high, you could just accumulate the number of times each value occurs in a histogram, then find the median from the histogram counts (just add up counts from the top and bottom of the histogram until you reach the middle). Or if they're continuous values, you could distribute them into bins - that wouldn't tell you the exact median but it would give you a range, and if you need to know more precisely you could iterate over the list again, examining only the elements in the central bin.


### How would you measure distance between data points?

Euclidean distance is the most common use of distance. In most cases when people said about distance, they will refer to Euclidean distance. Euclidean distance is also known as simply distance. When data is dense or continuous, this is the best proximity measure.The Pythagorean theorem gives this distance between two points.

Manhattan distance is a metric in which the distance between two points is the sum of the absolute differences of their Cartesian coordinates. In a simple way of saying it is the total sum of the difference between the x-coordinates  and y-coordinates.

The Minkowski distance is a generalized metric form of Euclidean distance and Manhattan distance

Cosine similarity metric finds the normalized dot product of the two attributes. By determining the cosine similarity, we would effectively try to find the cosine of the angle between the two objects. The cosine of 0° is 1, and it is less than 1 for any other angle.

The Jaccard similarity measures the similarity between finite sample sets and is defined as the cardinality of the intersection of sets divided by the cardinality of the union of the sample sets. It is the ratio of cardinality of A ∩ B and A ∪ B

https://tekmarathon.com/2015/11/15/different-similaritydistance-measures-in-machine-learning/


### How would you remove multicollinearity?

In regression, "multicollinearity" refers to predictors that are correlated with other predictors.  Multicollinearity occurs when your model includes multiple factors that are correlated not just to your response variable, but also to each other. In other words, it results when you have factors that are a bit redundant.

Multicollinearity increases the standard errors of the coefficients. Increased standard errors in turn means that coefficients for some independent variables may be found not to be significantly different from 0. In other words, by overinflating the standard errors, multicollinearity makes some variables statistically insignificant when they should be significant.

One way to measure multicollinearity is the variance inflation factor (VIF), which assesses how much the variance of an estimated regression coefficient increases if your predictors are correlated.  If no factors are correlated, the VIFs will all be 1. A VIF between 5 and 10 indicates high correlation that may be problematic

If you have two or more factors with a high VIF, remove one from the model. Because they supply redundant information, removing one of the correlated factors usually doesn't drastically reduce the R-squared.  Consider using stepwise regression, best subsets regression, or specialized knowledge of the data set to remove these variables. Select the model that has the highest R-squared value. 

Use Partial Least Squares Regression (PLS) or Principal Components Analysis, regression methods that cut the number of predictors to a smaller set of uncorrelated components.


### What is collinearity and what to do with it?

Collinearity occurs when two predictor variables (e.g., x1 and x2) in a multiple regression have a non-zero correlation. Multicollinearity occurs when more than two predictor variables (e.g., x1, x2 and x3) are inter-correlated. 

Multicollinearity has no impact on the overall regression model and associated statistics such as R2, F ratios and p values.
Multicollinearity is a problem if you are interested in the effects of individual predictors.

http://psychologicalstatistics.blogspot.com/2013/11/multicollinearity-and-collinearity-in.html



### What is the difference between squared error and absolute error?

In squared error, you are penalizing large deviations more. Square a big number, and it becomes much larger, relative to the others. Root Mean Square Error (RMSE) basically tells you to avoid models that give you occasional large errors.

Mean absolute deviation (MAD) says that being one standard deviation away and five standard deviations away “averages out” to 3 SDs away, even though being 5 away is astronomically unlikely, even compared with 3.

MSE has nice mathematical properties which makes it easier to compute the gradient.

### What is the null hypothesis? How do we state it?

A null hypothesis is a precise statement about a population that we try to reject with sample data.
Often -but not always- the null hypothesis states there is no association or difference between variables or subpopulations.

### When do we need the intercept term and when do we not?

The intercept (often labeled the constant) is the expected mean value of Y when all X=0.

Basically there is only one reason to perform a regression without using the intercept: Whenever your model is used to describe a process which is known to have a zero-intercept.



## Distributions

### What is a random variable? PMF, CDF?

A random variable, usually written X, is a variable whose possible values are numerical outcomes of a random phenomenon. There are two types of random variables, discrete and continuous.

The probability distribution of a discrete random variable is a list of probabilities associated with each of its possible values. It is also sometimes called the probability function or the probability mass function.

All random variables (discrete and continuous) have a cumulative distribution function. It is a function giving the probability that the random variable X is less than or equal to x, for every value x. For a discrete random variable, the cumulative distribution function is found by summing up the probabilities.

A continuous random variable is one which takes an infinite number of possible values. Continuous random variables are usually measurements. Examples include height, weight, the amount of sugar in an orange, the time required to run a mile.


### Describe a non-normal probability distribution and how to apply it.

Many data sets naturally fit a non normal model. 
For example, the number of accidents tends to fit a Poisson distribution and lifetimes of products usually fit a Weibull distribution.
Several tests, including the one sample Z test, T test and ANOVA assume normality. 
You may still be able to run these tests if your sample size is large enough (usually over 20 items).
Otherwise non parametric tests (https://www.statisticshowto.datasciencecentral.com/parametric-and-non-parametric-data/)

Mann-Whitney U test is the nonparametric equivalent of the two sample t-test. While the t-test makes an assumption about the distribution of a population (i.e. that the sample came from a t-distributed population), the Mann Whitney U Test makes no such assumption.


### Do you know the Dirichlet distribution? How does it differ from the multinomial distribution?

The dirichlet distribution is a probability distribution as well - but it is not sampling from the space of real numbers. Instead it is sampling over a probability simplex.

And what is a probability simplex? It’s a bunch of numbers that add up to 1. For example:

(0.6, 0.4)
(0.1, 0.1, 0.8)
(0.05, 0.2, 0.15, 0.1, 0.3, 0.2)

These numbers represent probabilities over K distinct categories. In the above examples, K is 2, 3, and 6 respectively. That’s why they are also called categorical distributions.

Binomial distribution: the number of successes in a sequence of independent yes/no experiments (Bernoulli trials).
Multinomial: suppose that each experiment results in one of k possible outcomes with probabilities p1, . . . , pk


The primary difference between Dirichlet and multinomial distributions is that Dirichlet random variables are real-valued, where each element of the vector is in the interval [0, 1], and multinomial random variables are integer-valued.

### Explain what a long-tailed distribution is and provide three examples of relevant phenomena that have long tails. Why are they important in classification and prediction problems?

Generally, this describes a feature of a distribution where the probability of increasingly large numbers (generally) monotonically decreases.

Two such examples of such distributions include a power-law distribution(long tail) and an exponential distribution. 

A good example of a power law distribution is the number of airports a uniformly random chosen airport has direct flights to. 

### Give examples of data that does not have a Gaussian distribution, or log-normal.

There are quite a few different reasons why non-normal distributions many occur. One of the main reasons involves extreme values. Logically, a few extreme values can really offset data, in order to fix this scenario the data must be checked and should be evaluated for things like data-entry errors and measurement errors. If errors are found, those pieces of data should be removed. Another cause of non-normal distribution could include insufficient data discrimination; this means that there are an insufficient number of different values.
Examples: people's incomes; mileage on used cars for sale; reaction times in a psychology experiment; house prices; number of accident claims by an insurance customer; number of children in a family.


### How would you check if a distribution is close to Normal? Why would you want to check it? What is a QQ Plot?

Visual methods: Density plot and Q-Q plot can be used to check normality visually.
Statistical tests:  Kolmogorov-Smirnov (K-S) normality test and Shapiro-Wilk’s test.
- (shapiro test -- p val < 0.05 suggests non normality)

A Q-Q plot is a scatterplot created by plotting two sets of quantiles against one another. Q-Q plots take your sample data, sort it in ascending order, and then plot them versus quantiles calculated from a theoretical distribution. 


### How would you find an anomaly in a distribution?

Detecting anlomalies:
- Outlier detection: Almost impossible values can be considered anomalies. When the value deviates too much from the mean, let’s say by ± 4σ, then we can considerate this almost impossible value to be anomalous.
- Detecting a change in the normal distribution. To detect this kind of anomaly we use a “window” containing the n most recent elements. If the mean and standard derivation of this window change too much from their expected values, we can deduce an anomaly. 


### What is an interaction?

Interaction effects occur when the effect of one variable depends on the value of another variable.
http://statisticsbyjim.com/regression/interaction-effects/

### What is Power analysis?

Power analysis is an important aspect of experimental design. It allows us to determine the sample size required to detect an effect of a given size with a given degree of confidence. 



### What are confounding variables?

A confounding variable is an “extra” variable that you didn’t account for. They can ruin an experiment and give you useless results. They can suggest there is correlation when in fact there isn’t. They can even introduce bias. 


### What is a selection bias?

Selection bias is the term used to describe the situation where an analysis has been conducted among a subset of the data (a sample) with the goal of drawing conclusions about the population, but the resulting conclusions will likely be wrong (biased), because the subgroup differs from the population in some important way. Selection bias is usually introduced as an error with the sampling and having a selection for analysis that is not properly randomized



## A/B testing
It is a way of conducting an experiment where you compare a control group to the performance of one or more test groups.
- Eg: Conversion rate of a web page: 
	- Subjected to 3 treatments(Adding layout, change in headings, Changing the color/text)
	- When a new user visits, we randomly assign him to one of the group( control , treatment A, treatmemt B)
	- Get the data after the experiment. Sample size matters(> sample size means more confidently we can say about the result)
	- Null hypothesis is that there is no difference between control group and treatment group.
- **Multivariate testing** uses the same core mechanism as A/B testing, but compares a higher number of variables, and reveals more information about how these variables interact with one another. Think of it as multiple A/B tests layered on top of each other.

	- Multivariate tests can take longer to achieve results of statistical significance. (More data to get better confidence)


