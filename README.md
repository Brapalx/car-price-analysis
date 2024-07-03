# car-price-analysis

This is a data exploration exercise. The provided dataset contains information on 426K cars to ensure speed of processing. The goal is to understand what factors make a car more or less expensive. This information would be useful for a used car dealership.

# The Dataset

 The initial dataset contains information on 426k cars. These are all its columns:
![image description](images/data_initial.png)

# Data Preparation
To prepare the dataset for modeling, I chose to drop the following columns: ID, Region, Model, VIN and State. ID and VIN are unique values for each car. The Model column has thousands of unique values which would make training extremely slow. Region and State are location dependant, which isn't useful for the goal which is to provide information to all car sellers, regardless of where they operate.

The dataset had a huge amount of null values as well, so those were removed. Some unclear values like 'other' and 'missing' were removed from all columns, as well as values that barely appeared at all. The Cylinders and Transmission columns were converted to numeric types, and the latter was renamed to Automatic.

This is how the data looks after the preparation stage:

![image description](images/data_prepared.png)

# Building the Models
For this stage, I built two different pipelines. One uses Ridge Regression and the other one uses regular Linear Regression.
This is how the pipeline structure looks like:
![image description](images/pipeline.png)

An OrdinalEncoder is used for the Condition feature, and OneHotEncoder is used for every non-numeric feature. The rest are fed into a PolynomialFeatures object of degree 3. 

With this pipeline, I created a GridSearchCV object to look for the best value of alpha to be used in the Ridge regressor.

![image description](images/best_estimator.png)

The best value found was 0.001.

The pipeline with regular Linear Regression didn't require any hyper parameter tuning. The MSE on the test set was 0.917 on the Ridge model and 0.913 on the LinReg model.

# Findings
To answer the question of what features impact the price of a used car more, we'll look at permutation importance means and individual coefficients.

## Permutation Importance

![image description](images/perm_imp.png)

As we can see in this bar plot, the year in which the car was manufactured is the most important feature when it comes to determining its price. This is followed by the number of cylinders, the number of miles it has in its odometer, and the car's type.

## Model Coefficients

![image description](images/coefs.png)

In this case, the largest positive and negative factors for price prediction are a combination of the year value and the number of cylinders. Thus, these are the most important features to take into account when determining the price for a used car.

