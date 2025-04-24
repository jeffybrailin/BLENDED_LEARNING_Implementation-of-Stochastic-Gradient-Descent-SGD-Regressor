# BLENDED_LEARNING
# Implementation-of-Stochastic-Gradient-Descent-SGD-Regressor

## AIM:
To write a program to implement Stochastic Gradient Descent (SGD) Regressor for linear regression and evaluate its performance.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
 1.Import Necessary Libraries: Import required libraries such as pandas, numpy, matplotlib, sklearn for the implementation.

2.Load the Dataset: Load the dataset (e.g., CarPrice_Assignment.csv) using pandas.

3.Data Preprocessing: Drop unnecessary columns (e.g., 'CarName', 'car_ID'). Handle categorical variables using pd.get_dummies().

4.Split the Data: Split the dataset into features (X) and target variable (Y). Split the data into training and testing sets using train_test_split().

5.Standardize the Data: Standardize the feature data (X) and target variable (Y) using StandardScaler() to ensure they have mean=0 and variance=1.

6.Create the SGD Regressor Model: Initialize the SGD Regressor model with max_iter=1000 and tol=1e-3.

7.Train the Model: Fit the model to the training data using the fit() method.

8.Make Predictions: Use the trained model to predict the target values for the test set.

9.Evaluate the Model: Calculate performance metrics like Mean Squared Error (MSE) and R-squared score using mean_squared_error() and r2_score().

10.Display Model Coefficients: Display the model's coefficients and intercept.

11Visualize the Results: Create a scatter plot comparing actual vs predicted prices.

12.End:The program finishes by displaying the evaluation metrics, model coefficients, and a visual representation of the predictions.

## Program:
```
/*
Program to implement SGD Regressor for linear regression.
Developed by:JEFFY BRAILIN T
RegisterNumber: 212223040076
*/
# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Load the dataset
data = pd.read_csv('CarPrice_Assignment.csv')
print(data.head())
print("\n\n")
print(data.info())

# Data preprocessing
# Dropping unnecessary columns and handling categorical variables
data = data.drop(['CarName', 'car_ID'], axis=1)
data = pd.get_dummies(data, drop_first=True)

# Splitting the data into features and target variable
x = data.drop('price', axis=1)
y = data['price']

# Standardizing the data
scaler = StandardScaler()
x = scaler.fit_transform(x)
y = scaler.fit_transform(np.array(y).reshape(-1, 1)).ravel()


# Splitting the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

# Creating the SGD Regressor model
sgd_model = SGDRegressor(max_iter=1000, tol=1e-3)

# Fitting the model on the training data
sgd_model.fit(x_train, y_train)

# Making predictions
y_pred = sgd_model.predict(x_test)

# Evaluating model performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print evaluation metrics
print()
print()
print("Mean Squared Error:", mse)
print("R-squared Score:", r2)

# Print model coefficients
print("\n\n")
print("Model Coefficients")
print("Coefficients:", sgd_model.coef_)
print("Intercept:", sgd_model.intercept_)

# Visualizing actual vs predicted prices
print("\n\n")
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices using SGD Regressor")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')  # Perfect prediction line
plt.show()
 
```

## Output:
![image](https://github.com/user-attachments/assets/9c3f05ef-4ddb-41e0-853e-8966e738d1ae)
![image](https://github.com/user-attachments/assets/bba23ee7-0567-4233-b189-1ee696b2c938)
![image](https://github.com/user-attachments/assets/0842549c-48ce-44a8-a4fc-55dfca995d2a)
![image](https://github.com/user-attachments/assets/ad0a12c8-d377-497d-9df9-9049c945973d)





## Result:
Thus, the implementation of Stochastic Gradient Descent (SGD) Regressor for linear regression has been successfully demonstrated and verified using Python programming.
