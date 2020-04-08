
# Introduction to Cross-Validation - Lab

## Introduction

In this lab, you'll be able to practice your cross-validation skills!


## Objectives

You will be able to:

- Compare the results with normal holdout validation
- Apply 5-fold cross validation for regression

## Let's get started

This time, let's only include the variables that were previously selected using recursive feature elimination. We included the code to preprocess below.


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.datasets import load_boston

boston = load_boston()

boston_features = pd.DataFrame(boston.data, columns = boston.feature_names)
b = boston_features["B"]
logdis = np.log(boston_features["DIS"])
loglstat = np.log(boston_features["LSTAT"])

# minmax scaling
boston_features["B"] = (b-min(b))/(max(b)-min(b))
boston_features["DIS"] = (logdis-min(logdis))/(max(logdis)-min(logdis))

#standardization
boston_features["LSTAT"] = (loglstat-np.mean(loglstat))/np.sqrt(np.var(loglstat))
```


```python
X = boston_features[['B', 'DIS', "LSTAT", 'CHAS', 'RM']]
y = boston.target
```

## Train test split

Perform a train-test-split with a test set of 0.20.


```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
```


```python
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
```


```python

```

Fit the model and apply the model to the make test set predictions


```python
model = LinearRegression().fit(x_train, y_train)
y_test_hat = model.predict(x_test)
```

Calculate the residuals and the mean squared error


```python
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, y_test_hat)
mse
```




    23.678993120074722



## Cross-Validation: let's build it from scratch!

### Create a cross-validation function

Write a function k-folds that splits a dataset into k evenly sized pieces.
If the full dataset is not divisible by k, make the first few folds one larger then later ones.

We want the folds to be a list of subsets of data!


```python
def kfolds(data, k):
    # Force data as pandas dataframe
    data_df = pd.DataFrame(data)
    split_size = len(data_df) // k
    print(len(data_df))
    split_remainder = len(data_df) % k
    print(split_remainder)
    folds = []
    # add 1 to fold size to account for leftovers
    previous_index = 0
    for i in range(1,k+1):
        if i <= split_remainder:
            new_df_name = "data_df_" + str(i)
            last_index = (previous_index + split_size+1)
            print("First index: {}, Last Index: {}".format(previous_index, last_index))
            folds.append(data_df.iloc[previous_index:last_index]) 
            previous_index = last_index 
        else:
            new_df_name = "data_df_" + str(i)
            last_index = (previous_index + split_size)
            print("##First index: {}, Last Index: {}".format(previous_index, last_index))
            folds.append(data_df.iloc[previous_index:last_index]) 
            previous_index = last_index            
    return folds

```


```python
boston_features['MEDV'] = boston.target
```


```python
cross_val_list = kfolds(boston_features, 5)
```

    506
    1
    First index: 0, Last Index: 102
    ##First index: 102, Last Index: 203
    ##First index: 203, Last Index: 304
    ##First index: 304, Last Index: 405
    ##First index: 405, Last Index: 506


### Apply it to the Boston Housing Data


```python
# Make sure to concatenate the data again
```


```python

```

### Perform a linear regression for each fold, and calculate the training and test error

Perform linear regression on each and calculate the training and test error.


```python
test_errs = []
train_errs = []
k=5

for n in range(k):
    # Split in train and test for the fold
    train = pd.concat([fold for i, fold in enumerate(cross_val_list) if i != n])
    test = cross_val_list[n]
    train_features = train.drop(['MEDV'], axis=1)
    test_features = test.drop(['MEDV'], axis=1)
    # Fit a linear regression model
    model = LinearRegression().fit(train_features, train['MEDV'])
    #Evaluate Train and Test Errors
    train_hat = model.predict(train_features)
    test_hat = model.predict(test_features)
    train_errs.append(mean_squared_error(train['MEDV'], train_hat))
    test_errs.append(mean_squared_error(test['MEDV'], test_hat))

print(train_errs)
print(test_errs)
```

    [17.9185670542463, 17.3577081046629, 15.545678258525871, 11.03762238964458, 17.23404426556592]
    [13.016192102045745, 14.62832183142464, 24.81432997168215, 55.241077726377355, 19.022337999169658]


## Cross-Validation using Scikit-Learn

This was a bit of work! Now, let's perform 5-fold cross-validation to get the mean squared error through scikit-learn. Let's have a look at the five individual MSEs and explain what's going on.


```python
from sklearn.model_selection import cross_val_score
```

Next, calculate the mean of the MSE over the 5 cross-validations and compare and contrast with the result from the train-test-split case.


```python
linreg = LinearRegression()
cross_val_score(linreg, boston_features.drop(['MEDV'], axis=1), boston_features['MEDV'], cv=5, 
                 scoring="neg_mean_squared_error")
```




    array([-13.0161921 , -14.62832183, -24.81432997, -55.24107773,
           -19.022338  ])



##  Summary 

Congratulations! You now practiced your knowledge on k-fold crossvalidation!
