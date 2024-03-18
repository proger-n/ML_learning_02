# Linear regression model

Summary: This project is about supervised learning and particularly about linear model, regularization, overfitting and underfitting and metrics for quality estimation.


## Task

We will continue our practice with a problem from Kaggle.com. In this chapter we will implement 
all models that were described upstairs. Measure quality metrics on train and test parts. 
Will detect overfitted models and regularize these. And dive more deeply with native model estimation and comparison.

1. Answer to the questions
   1. Derive an analytical solution for the regression task. Use a vector form of equation.
   2. What changes in solution when adding L1 and L2 regularization to the loss function.
   3. Explain why L1 regularization is commonly used to select features. Why are there a lot of weights that are equal to 0 after the model is fitted?
   4. Explain how you can use the same models (Linear Regression, Ridge, etc) but make it possible to fit nonlinear dependencies?
2. Introduction  - make all preprocess staff from the previous lesson
   1. Import libraries. 
   2. Read train and test parts.
   3. Preprocess “interest level” feature.
3. Intro data analysis part 2
   1. Lets generate additional features for better model quality. Consider a column named “features”. It consists of a list of highlights of the current flat. 
   2. Remove unused symbols ([,],’,” and space) from the column.
   3. Split values in every row with separator “,” and collect the result in one huge list for the whole dataset. You may use DataFrame.iterrows().
   4. How many unique values consist of a result list?
   5. Let's get to know the new library - Collections. With this package you could effectively get quantity statistics about your data. 
   6. Count the most popular features of our huge list and take the top 20 for this moment.
   7. If everything correct you should get next values:  'Elevator', 'HardwoodFloors', 'CatsAllowed', 'DogsAllowed', 'Doorman', 'Dishwasher', 'NoFee', 'LaundryinBuilding', 'FitnessCenter', 'Pre-War', 'LaundryinUnit', 'RoofDeck', 'OutdoorSpace', 'DiningRoom', 'HighSpeedInternet', 'Balcony', 'SwimmingPool', 'LaundryInBuilding', 'NewConstruction', 'Terrace'.
   8. Now generate 20 new features based on top 20 values: 1 if value in “feature” column else 0.
   9. Extend our features set with  'bathrooms',  'bedrooms', 'interest_level' and lets create a special variable feature_list with all feature names. Now there are 23 values. All models should be trained on these 23 features.
4. Models implementation - linear regression
   1. Implement python class for a linear regression algorithm with two basic methods - fit and predict. Use stochastic gradient descent to find optimal weights of the model. For better understanding we recommend additionally implementing separate versions of the algorithm with the analytical solution and non-stochastic gradient descent under the hood.
   2. Give definition for R squared (R2) coefficient and implement function for calculation.
   3. Make predictions with your algorithm and estimate model with MAE, RMSE and R2 metrics.
   4. Initialize LinearRegression() from sklearn.linear_model, fit model and predict train and test parts, same as previous lesson.
   5. Compare quality metrics and make sure that difference is small. (Between your implementations and sklearn).
   6. Save metrics as in the previous lesson in a table with columns model, train, test for MAE table, RMSE table and R2 coefficient.
5. Regularized models implementation - Ridge, Lasso, ElasticNet    
   1. Implement Ridge, Lasso, ElasticNet algorithms: extend loss function with L2, L1 and both regularization correspondingly.
   2. Make predictions with your algorithm and estimate the model with MAE, RMSE and R2 metrics.
   3. Initialize Ridge(), Lasso() and ElasticNet() from sklearn.linear_model, fit model and make prediction for train and test samples, same as previous lesson.
   4. Compare quality metrics and make sure that difference is small. (Between your implementations and sklearn).
   5. Save metrics as in the previous lesson in a table with columns model, train, test for MAE table, RMSE table and R2 coefficient.
6. Feature normalization
   1. First of all, please write several examples why and where features normalization is mandatory and vice versa.
   2. Here let's consider the first of classical normalization methods - MinMaxScaler. Please write a mathematical formula for the method.
   3. Implement your own function for MinMaxScaler feature normalization.
   4. Initialize MinMaxScaler() from sklearn.preprocessing.
   5. Compare feature normalization with your own method and from sklearn.
   6. Repeat steps from b to e for one more normalization method StandardScaler.
7. Fit models with normalization
   1. Fit all models - Linear regression, Ridge, Lasso and ElasticNet with MinMaxScaler.
   2. Fit all models - Linear regression, Ridge, Lasso and ElasticNet with StandardScaler.
   3. Add all results to our dataframe with metrics on samples.
8. Overfit models
   1. Let's consider an overfitted model in practice. After theory, you know that polynomial regression is easy to overfit. So let's create toy example and see how regularization works in real life.
   2. In the last lesson we created polynomial features with degree equals 10. Here repeat these steps from the previous lesson, remember, with only 3 basic features - 'bathrooms’, 'bedrooms’, ‘'interest_level'.
   3. And train and fit all our implemented algorithms - Linear regression, Ridge, Lasso and ElasticNet on a set of polynomial features.
   4. Store results of quality metrics in the result dataframe.
   5. Analyze results, and choose the best model for your opinion.
   6. Addition try different alpha parameters of regularization in algorithms, choose best and analyze results.
9. Native models
   1. Calculate metrics for mean and median from previous lesson and add results to final dataframe.
10. Compare results
    1. Print your final tables
    2. What is the best model?
    3. What is the most stable model?
11. Addition task
    1. There are some tricks with target variable for better model quality. If we have a distribution with a heavy tail, you can use some monotonic function to “improve” distribution. In practice you can use logarithmic functions. We recommend you do this exercise and compare results. But don’t forget to perform inverse transformation when you will compare metrics.
    2. Next trick is outliers. The angle of the linear regression line strongly depends on outlier points. And often you should remove these points from !allert! only train data. You should explain why it was removed only from the train sample.  We recommend you do this exercise and compare results.
    3. Also it will be a useful exercise to realize linear regression algorithm with batch training.

### Submission

Save your code in python JupyterNotebook. 
