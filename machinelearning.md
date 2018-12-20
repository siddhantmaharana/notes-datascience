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



## Logistic Regression

- * Logistic regression *  is a technique that is well suited for examining the relationship between a categorical response variable and one or more categorical or continuous predictor variables. The big difference from the linear regression is that logistic regression used the log(odds) on the y-axis.
- General format log(odds)=β0+β1∗x1+...+βn∗xn
- The log(odds), or log-odds ratio, is defined by ln[p/(1−p)] and expresses the natural logarithm of the ratio between the probability that an event will occur, p(Y=1), to the probability that it will not occur. 
- The estimates from logistic regression characterize the relationship between the predictor and response variable on a log-odds scale. So taking the exponential of the log odds gives a good representation. Eg: We can then say that one unit increase in Age, the odds of having good credit increases by a factor of 1.01.

### How would you evaluate a logistic regression model?

Evaluation
	- AIC, Confusion Matrix, ROC-AUC Curve
	- Null and Residual Deviance: The difference between the null deviance and the residual deviance shows how our model is doing against the null model(a model with only the intercept).
	- Pseudo R2- Logistic regression models are fitted using the method of maximum likelihood - i.e. the parameter estimates are those values which maximize the likelihood of the data which have been observed. McFadden's R squared measure is defined as
	R2McFadden=1−log(Lc)/log(Lnull)
	where Lc denotes the (maximized) likelihood value from the current fitted model, and Lnull denotes the corresponding value but for the null model - the model with only an intercept and no covariates.


### How MLE works in Logistic Regression?
	
We can transform the y axis from the probability of independent variable to the log odds of IV. We can then draw a candidate best fitting line on the graph. But the transformation pushes the data to positive and negative infinities, making the residuals to infinities. Thus we can't use OLS estimation to get the best fitting line. 
Come MLE. First thing we do is project the original data onto the candidate line which gives each sample a log odds value. Then we transform the candidate log odds to candidate probabilities using p = exp(log(odds))/1+exp(log(odds)). 
Then we plot these on the squiggle and calculate the likelihood of all these samples for the two classes. So we try to find the squiggle that maximizes this likelihood.

## Classification


### How would you deal with unbalanced binary classification?

This is a scenario where the number of observations belonging to one class is significantly lower than those belonging to the other classes. Eg: Fraudulent transactions are significantly lower than normal healthy transactions i.e. accounting it to around 1-2 % of the total number of observations.

Standard classifier algorithms like Decision Tree and Logistic Regression have a bias towards classes which have number of instances. They tend to only predict the majority class data. 

Techniques to deal with them
1. Resampling:
	- Random Undersampling aims to balance class distribution by randomly eliminating majority class examples.  This is done until the majority and minority class instances are balanced out.
	- Random Oversampling
	- Clustered based oversampling: In this case, the K-means clustering algorithm is independently applied to minority and majority class instances. This is to identify clusters in the dataset. Subsequently, each cluster is oversampled such that all clusters of the same class have an equal number of instances and all classes have the same size. 
	- Synthetic Minority Over-sampling Technique:  A subset of data is taken from the minority class as an example and then new synthetic similar instances are created. These synthetic instances are then added to the original dataset.
	- Modified synthetic minority oversampling technique: SMOTE does not consider the underlying distribution of the minority class and latent noises in the dataset. This algorithm classifies the samples of minority classes into 3 distinct groups – Security/Safe samples, Border samples, and latent nose samples. The algorithm randomly selects a data point from the k nearest neighbors for the security sample, selects the nearest neighbor from the border samples and does nothing for latent noise.
2. Algorithmic Ensemble Techniques:
	- Modifying existing classification algorithms to make them appropriate for imbalanced data sets.
	- Bagging: The conventional bagging algorithm involves generating ‘n’ different bootstrap training samples with replacement. And training the algorithm on each bootstrapped algorithm separately and then aggregating the predictions at the end.
	- Boosting-Based: Boosting is an ensemble technique in which the predictors are not made independently, but sequentially. Boosting starts out with a base classifier / weak classifier that is prepared on the training data. In the next iteration, the new classifier focuses on or places more weight to those cases which were incorrectly classified in the last round.
	- Adaptive Boosting: Each classifier is serially trained with the goal of correctly classifying examples in every round that were incorrectly classified in the previous round. After each iteration, the weights of misclassified instances are increased and the weights of correctly classified instances are decreased.
	- Gradient Boosting: Decision Trees are used as weak learners in Gradient Boosting. In this technique, the residual of the loss function is calculated using the Gradient Descent Method and the new residual becomes a target variable for the subsequent iteration.
	- XG boost: Implements parrallel processing, thereby making it 10 times faster than the regular Gradient Boosting technique.

## Tradeoffs between different types of classification models. How to choose the best one?

There is no a well defined rule for such task. 
If your training set is small: 
- high bias/low variance classifiers (e.g., Naive Bayes) have an advantage over low bias/high variance classifiers (e.g., kNN), since the latter will overfit.
But low bias/high variance classifiers start to win out as your training set grows (they have lower asymptotic error)

Naive Bayes: 
	Pros: Super simple. Will converge quicker if the independence assumption holds(Computationally fast). Works well with high dimensions(words as features)
	Cons: Can't learn interaction between features.
Logistic Regression:
	Pros: Don't have to worry about feature being correlated. Can use it in a probabilistic framework. Has low variance.
	Cons: Often has high bias.
Decision Trees:
	Pros: Easy to interpret and visualize.
	Cons: Prone to overfitting
Bagged Trees:
	Pros: reduced variance compared to regular trees
	Cons: Interpretation is lost
Boosted Trees:
	Pros: More interpretable than bagged trees.
	Cons: Prone to overfitting
Random Forest:
	Pros: Decorrelates the trees and also has less variance
	Cons: Not visually interpretable
SVM:
	Pros: Performs well in non linear separation; Handles high dimensional data well
	Cons: Prone to overfitting, Can take lot of time


### Describe how Gradient Boosting works.

Gradient boosting is a machine learning technique for regression and classification problems, which produces a prediction model in the form of an ensemble of weak prediction models, typically decision trees.

So, the intuition behind gradient boosting algorithm is to repetitively leverage the patterns in residuals and strengthen a model with weak predictions and make it better. Once we reach a stage that residuals do not have any pattern that could be modeled, we can stop modeling residuals 


### Describe some of the different splitting rules used by different decision tree algorithms. 3, How would you build a decision tree model?

Entropy: A decision tree is built top-down from a root node and involves partitioning the data into subsets that contain instances with similar values (homogenous). ID3 algorithm uses entropy to calculate the homogeneity of a sample. If the sample is completely homogeneous the entropy is zero and if the sample is an equally divided it has entropy of one.

Gini Index: Gini index says, if we select two items from a population at random then they must be of same class and probability for this is 1 if population is pure.Higher the value of Gini higher the homogeneity.

https://www.analyticsvidhya.com/blog/2016/04/complete-tutorial-tree-based-modeling-scratch-in-python/

### How would you compare a decision tree to a logistic regression? Which is more suitable under different circumstances?

Decision Trees bisect the space into smaller and smaller regions, whereas Logistic Regression fits a single line to divide the space exactly into two. Of course for higher-dimensional data, these lines would generalize to planes and hyperplanes. A single linear boundary can sometimes be limiting for Logistic Regression.

Decision tree work well when the decision boundary is not linear, whereas logistic regression works well in case of linear decision boundaries.
Tree-based regression is a non-parametric model and does not handle 0-1 response variable very well. Logistic regression is a parametric model and fits data between 0-1. For fitting purpose, non-parametric model outperforms parametric model. For forecasting purpose, parametric model outperforms non-parametric model.
Parametric model is easy to explain compared with non-parametric one.


### What are some business reasons you might want to use a decision tree model?

Explain churn: The decision tree shows how the other data predicts whether or not customers churned.
https://www.displayr.com/decision-trees-are-usually-better-than-logistic-regression/

### What is pruning and why is it important?

Pruning reduces the size of decision trees by removing parts of the tree that do not provide power to classify instances. Decision trees are the most susceptible out of all the machine learning algorithms to overfitting and effective pruning can reduce this likelihood

Prune the tree by removing useless nodes based on:
– Additional test data (not used for training): Use cross validation on a test set to determine the performance
– Statistical significance tests- Chi square pruning to detect if the gain in information is statistically different from previous trees.


## Dimension Reduction

### Why standardize data? Difference between Normalizing and standardizing the data?

Standardisation replaces the values by their Z scores. This redistributes the features with their mean μ = 0 and standard deviation σ =1 .
It is not only important if we are comparing measurements that have different units, but it is also a general requirement for many machine learning algorithms. 

Min max scaling(Normalization): In this approach, the data is scaled to a fixed range - usually 0 to 1. A popular application is image processing, where pixel intensities have to be normalized to fit within a certain range.

- Scaling is not required while modelling trees, Linear Discriminant Analysis(LDA), Naive Bayes
- k-nearest neighbors with an Euclidean distance measure is sensitive to magnitudes and hence should be scaled for all features to weigh in equally.
- Scaling is critical, while performing Principal Component Analysis(PCA) and also while computing gradient descent.

### What are some dimensionality reduction techniques? Are all of them are (un)supervised?

Dimensional reduction can be broken down to Feature Selection and Feature Extraction.
Feature Selection:
Feature selection is for filtering irrelevant or redundant features from your dataset. The key difference between feature selection and extraction is that feature selection keeps a subset of the original features while feature extraction creates brand new ones.

- Some supervised algorithms already have built in feature selection. Regularized Regression and Random Forests.
- Variance thresholds can be used to remove features whose values don't change much from observation to observation.
- Correlation thresholds can also be used. If the correlation between a pair of features is above a given threshold, you'd remove the one that has larger mean absolute correlation with other features. 
- Stepwise search is a supervised feature selection method based on sequential search, and it has two flavors: forward and backward. 
	- For forward stepwise search, you start without any features. Then, you'd train a 1-feature model using each of your candidate features and keep the version with the best performance. You'd continue adding features, one at a time, until your performance improvements stall.
	- Backward stepwise search is the same process, just reversed: start with all features in your model and then remove one at a time until performance starts to drop substantially.


Feature Extraction:
Feature extraction is for creating a new, smaller set of features that stills captures most of the useful information
- Principal component analysis (PCA) is an unsupervised algorithm that creates linear combinations of the original features. The new features are orthogonal, which means that they are uncorrelated. Furthermore, they are ranked in order of their "explained variance." The first principal component (PC1) explains the most variance in your dataset, PC2 explains the second-most variance, and so on.
- Linear discriminant analysis (LDA) also creates linear combinations of your original features. However, unlike PCA, LDA doesn't maximize explained variance. Instead, it maximizes the separability between classes.

### Do we need to normalize data for PCA? Why?

PCA (Principal Component Analysis) finds new directions based on covariance matrix of original variables. We also knew that covariance matrix is sensitive to standardization of variables. Usually, we do standardization to assign equal weights to all the variables. It means that If we don't standardize the variables before applying PCA, we will get misleading directions
Mean centering does not affect the covariance matrix. However, scaling of variable affects the covariance matrix and so does standarization.

### What is the relationship between Principal Component Analysis (PCA) and Linear & Quadratic Discriminant Analysis (LDA & QDA)

Both LDA and PCA are linear transformation techniques: LDA is a supervised whereas PCA is unsupervised – PCA ignores class labels. In contrast to PCA, LDA attempts to find a feature subspace that maximizes class separability (note that LD 2 would be a very bad linear discriminant in the figure above). LDA makes assumptions about normally distributed classes and equal class covariances. 

## Hyperparameters

Model parameters are the properties of training data that will learn on its own during training by the classifier or other ML model. For example,Weights and Biases,Split points in Decision Tree

Model Hyperparameters are the properties that govern the entire training process. The below are the variables usually configure before training a model. For example: Learning Rate,Number of Epochs,Hidden Layers,Hidden Units,Activations Functions

- Hyperparameters are important because they directly control the behaviour of the training algorithm and have a significant impact on the performance of the model is being trained.

- Hyperparameters Optimisation Techniques:
	- Grid search is an approach to hyperparameter tuning that will methodically build and evaluate a model for each combination of algorithm parameters specified in a grid. 
	- Random search differs from a grid search. In that you longer provide a discrete set of values to explore for each hyperparameter; rather, you provide a statistical distribution for each hyperparameter from which values may be randomly sampled. So, faster.

## Regularization

This is a form of regression, that constrains/ regularizes or shrinks the coefficient estimates towards zero. In other words, this technique discourages learning a more complex or flexible model, so as to avoid the risk of overfitting.

Regularization, significantly reduces the variance of the model, without substantial increase in its bias. So the tuning parameter λ, used in the regularization techniques described above, controls the impact on bias and variance. As the value of λ rises, it reduces the value of coefficients and thus reducing the variance

### Ridge Regression:
RSS is modified by adding the shrinkage quantity. Now, the coefficients are estimated by minimizing this function. Here, λ is the tuning parameter that decides how much we want to penalize the flexibility of our model. 
However, as λ→∞, the impact of the shrinkage penalty grows, and the ridge regression coeﬃcient estimates will approach zero. As can be seen, selecting a good value of λ is critical. Cross validation comes in handy for this purpose. The coefficient estimates produced by this method are also known as the L2 norm.

## Lasso:
Lasso is another variation, in which the above function is minimized. Its clear that this variation differs from ridge regression only in penalizing the high coefficients. It uses |βj|(modulus)instead of squares of β, as its penalty. In statistics, this is known as the L1 norm.

Since ridge regression has a circular constraint with no sharp points, this intersection will not generally occur on an axis, and so the ridge regression coeﬃcient estimates will be exclusively non-zero. However, the lasso constraint has corners at each of the axes, and so the ellipse will often intersect the constraint region at an axis. When this occurs, one of the coefficients will equal zero.

This sheds light on the obvious disadvantage of ridge regression, which is model interpretability. It will shrink the coefficients for least important predictors, very close to zero. But it will never make them exactly zero. In other words, the final model will include all predictors. However, in the case of the lasso, the L1 penalty has the effect of forcing some of the coeﬃcient estimates to be exactly equal to zero when the tuning parameter λ is suﬃciently large. Therefore, the lasso method also performs variable selection and is said to yield sparse models.


## Model Evaluation Metrics

### Classification Accuracy
Classification Accuracy is what we usually mean, when we use the term accuracy. It is the ratio of number of correct predictions to the total number of input samples.
The real problem arises, when the cost of misclassification of the minor class samples are very high. If we deal with a rare but fatal disease, the cost of failing to diagnose the disease of a sick person is much higher than the cost of sending a healthy person to more tests.

### Logarithmic Loss
Logarithmic Loss or Log Loss, works by penalising the false classifications. It works well for multi-class classification. When working with Log Loss, the classifier must assign probability to each class for all the samples. 
Log Loss nearer to 0 indicates higher accuracy, whereas if the Log Loss is away from 0 then it indicates lower accuracy.

### Confusion Matrix
Confusion Matrix as the name suggests gives us a matrix as output and describes the complete performance of the model.
True Positives : The cases in which we predicted YES and the actual output was also YES.
True Negatives : The cases in which we predicted NO and the actual output was NO.
False Positives : The cases in which we predicted YES and the actual output was NO.
False Negatives : The cases in which we predicted NO and the actual output was YES.

### True Positive Rate (Sensitivity)
True Positive Rate is defined as TP/ (FN+TP). True Positive Rate corresponds to the proportion of positive data points that are correctly considered as positive, with respect to all positive data points.

### False Positive Rate (Specificity) : 
False Positive Rate is defined as FP / (FP+TN). False Positive Rate corresponds to the proportion of negative data points that are mistakenly considered as positive, with respect to all negative data points.

### AUC
AUC is the area under the curve of plot False Positive Rate vs True Positive Rate at different points in [0, 1].

### F1 Score
F1 Score is the Harmonic Mean between precision and recall. The range for F1 Score is [0, 1]. It tells you how precise your classifier is (how many instances it classifies correctly), as well as how robust it is (it does not miss a significant number of instances).
	- Precision : It is the number of correct positive results divided by the number of positive results predicted by the classifier. TP/TP+FP
	- Recall : It is the number of correct positive results divided by the number of all relevant samples (all samples that should have been identified as positive). TP/TP+FN

### Mean Absolute Error
Mean Absolute Error is the average of the difference between the Original Values and the Predicted Values. It gives us the measure of how far the predictions were from the actual output. 

### Mean Squared Error
Mean Squared Error(MSE) is quite similar to Mean Absolute Error, the only difference being that MSE takes the average of the square of the difference between the original values and the predicted values.



## Clustering

Clustering is the task of dividing the population or data points into a number of groups such that data points in the same groups are more similar to other data points in the same group than those in other groups. In simple words, the aim is to segregate groups with similar traits and assign them into clusters.

### K Means Clustering
K means is an iterative clustering algorithm that aims to find local maxima in each iteration. Steps involved include:
 - Specify the desired number of clusters K : Let us choose k=2 for these 5 data points in 2-D space.
 - Randomly assign each data point to a cluster
 - Compute cluster centroids
 - Re-assign each point to the closest cluster centroid
 - Re-compute cluster centroids
 - Repeat steps 4 and 5 until no improvements are possible

### Hierarchical Clustering
Hierarchical clustering, as the name suggests is an algorithm that builds hierarchy of clusters. This algorithm starts with all the data points assigned to a cluster of their own. Then two nearest clusters are merged into the same cluster. In the end, this algorithm terminates when there is only a single cluster left.

- The results of hierarchical clustering can be shown using dendrogram.
- Hierarchical clustering can’t handle big data well but K Means clustering can. This is because the time complexity of K Means is linear i.e. O(n) while that of hierarchical clustering is quadratic i.e. O(n2).


### How is KNN different from k-means clustering?

K-Nearest Neighbors is a supervised classification algorithm.  It requires labeled data to train.  Given the labeled points, KNN will classify new, unlabeled data by looking at the ‘k’ number of nearest data points.  The variable ‘k’ is a parameter that will be set by the machine learning engineer.


K-means clustering is an unsupervised clustering algorithm.  It requires unlabeled data to train.  Given the unlabeled points and some ‘k’ number of clusters, k-means clustering will gradually learn how to cluster the unlabeled points into groups by computing the mean distance between the points.

