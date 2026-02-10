# STEP 1 — Libraries import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# STEP 2 — Dataset load
data = pd.read_csv("dataset.csv")
print(data.head())
print(data.info())

# STEP 3 — EDA
print(data.describe())
print(data.isnull().sum())
print(data.corr())

data.hist(figsize=(10,8))
plt.show()

# =====================================================
# SIMPLE LINEAR REGRESSION (manual formula)
# =====================================================
print("\n===== SIMPLE LINEAR REGRESSION =====")

x = data['Feature1'].values
y = data['Target'].values

x_mean = np.mean(x)
y_mean = np.mean(y)

b1 = np.sum((x-x_mean)*(y-y_mean)) / np.sum((x-x_mean)**2)
b0 = y_mean - b1*x_mean

print("Slope:", b1)
print("Intercept:", b0)

y_pred_simple = b0 + b1*x

plt.scatter(x,y)
plt.plot(x,y_pred_simple,color='red')
plt.title("Simple Linear Regression")
plt.show()

# =====================================================
# MULTIPLE LINEAR REGRESSION (Normal Equation)
# =====================================================
print("\n===== MULTIPLE LINEAR REGRESSION =====")

X = data[['Feature1','Feature2','Feature3']].values
y = data['Target'].values

# add bias column
X_b = np.c_[np.ones((len(X),1)), X]

# normal equation
theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

print("Coefficients:", theta)

y_pred = X_b.dot(theta)

# =====================================================
# METRICS (manual)
# =====================================================
def mse(y, pred):
    return np.mean((y-pred)**2)

def rmse(y, pred):
    return np.sqrt(mse(y,pred))

def r2(y,pred):
    ss_total = np.sum((y-np.mean(y))**2)
    ss_res = np.sum((y-pred)**2)
    return 1 - ss_res/ss_total

print("MSE:", mse(y,y_pred))
print("RMSE:", rmse(y,y_pred))
print("R2:", r2(y,y_pred))

# =====================================================
# POLYNOMIAL REGRESSION (degree 2 manual)
# =====================================================
print("\n===== POLYNOMIAL REGRESSION =====")

x_poly = data['Feature1'].values
X_poly = np.c_[np.ones(len(x_poly)), x_poly, x_poly**2]

theta_poly = np.linalg.inv(X_poly.T.dot(X_poly)).dot(X_poly.T).dot(y)
y_pred_poly = X_poly.dot(theta_poly)

print("Polynomial R2:", r2(y,y_pred_poly))

# =====================================================
# RIDGE REGRESSION (manual)
# =====================================================
print("\n===== RIDGE REGRESSION =====")

lam = 1
I = np.eye(X_b.shape[1])
theta_ridge = np.linalg.inv(X_b.T.dot(X_b)+lam*I).dot(X_b.T).dot(y)

print("Ridge coefficients:", theta_ridge)

# =====================================================
# LASSO REGRESSION (Gradient Descent)
# =====================================================
print("\n===== LASSO REGRESSION =====")

theta_lasso = np.zeros(X_b.shape[1])
lr = 0.0001
lam = 0.1

for _ in range(1000):
    pred = X_b.dot(theta_lasso)
    error = pred - y

    grad = X_b.T.dot(error)/len(y)

    theta_lasso -= lr*(grad + lam*np.sign(theta_lasso))

print("Lasso coefficients:", theta_lasso)

# =====================================================
# RESIDUAL PLOT
# =====================================================
residuals = y - y_pred

plt.scatter(y_pred, residuals)
plt.xlabel("Predicted")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.show()
