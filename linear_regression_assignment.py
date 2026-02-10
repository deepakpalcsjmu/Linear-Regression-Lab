# Linear Regression Lab Assignment

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score


data = pd.read_csv('linear_regression_dataset.csv')

print(data.head())
print(data.info())
print(data.describe())
print(data.isnull().sum())

sns.boxplot(data=data)
plt.show()

sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.show()

X = data[['hours_studied','practice_score','attendance_rate']]
y = data['final_score']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

slr = LinearRegression()
slr.fit(X_train[['hours_studied']], y_train)
y_slr = slr.predict(X_test[['hours_studied']])

mlr = LinearRegression()
mlr.fit(X_train, y_train)
y_pred = mlr.predict(X_test)

print("MSE:", mean_squared_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R2 Score:", r2_score(y_test, y_pred))

poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

Xp_train, Xp_test, yp_train, yp_test = train_test_split(
    X_poly, y, test_size=0.2, random_state=42
)

poly_model = LinearRegression()
poly_model.fit(Xp_train, yp_train)

ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)

lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)

residuals = y_test - y_pred
plt.scatter(y_pred, residuals)
plt.axhline(0, color='red')
plt.show()

print("Linear Regression Experiment Completed Successfully")
