#STEP 1 — Libraries import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

#STEP 2 — Dataset load
data = pd.read_csv("dataset.csv")
print(data.head())
print(data.info())

#STEP 3 — EDA (5 marks)
print(data.describe())
print(data.isnull().sum())

# correlation
print(data.corr())

# plots
data.hist(figsize=(10,8))
plt.show()

#STEP 4 — Simple Linear Regression
X = data[['Feature1']]
y = data['Target']

model = LinearRegression()
model.fit(X, y)

print("Slope:", model.coef_)
print("Intercept:", model.intercept_)

plt.scatter(X, y)
plt.plot(X, model.predict(X), color='red')
plt.show()

#STEP 5 — Multiple Linear Regression
X = data[['Feature1','Feature2','Feature3']]
y = data['Target']

model = LinearRegression()
model.fit(X, y)

pred = model.predict(X)

print("MSE:", mean_squared_error(y, pred))
print("RMSE:", np.sqrt(mean_squared_error(y, pred)))
print("R2:", r2_score(y, pred))

#STEP 6 — Polynomial Regression
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(data[['Feature1']])

model = LinearRegression()
model.fit(X_poly, y)

pred_poly = model.predict(X_poly)

print("Polynomial R2:", r2_score(y, pred_poly))

#STEP 7 — Ridge Regression
ridge = Ridge(alpha=1.0)
ridge.fit(X, y)

print("Ridge coefficients:", ridge.coef_)

#STEP 8 — Lasso Regression
lasso = Lasso(alpha=0.1)
lasso.fit(X, y)

print("Lasso coefficients:", lasso.coef_)

#STEP 9 — Residual Plot
residuals = y - pred

plt.scatter(pred, residuals)
plt.xlabel("Predicted")
plt.ylabel("Residuals")
plt.show()
