X1 = [2, 4, 6, 8, 10]
X2 = [60, 70, 80, 85, 90]
X3 = [40, 50, 65, 70, 80]
Y  = [45, 55, 70, 78, 88]

#Part A: Exploratory Data Analysis (EDA)
#Dataset Information

print("Number of records:", len(Y))
print("Number of features: 3")

# Summary Statistics
def mean(data):
    s = 0
    for i in data:
        s += i
    return s / len(data)

def variance(data):
    m = mean(data)
    s = 0
    for i in data:
        s += (i - m) ** 2
    return s / len(data)

print("Mean of Y:", mean(Y))
print("Variance of Y:", variance(Y))

#Missing Values & Outliers
for i in Y:
    if i is None:
        print("Missing value found")

# PART B: SIMPLE LINEAR REGRESSION
n = len(X1)

sum_x = sum(X1)
sum_y = sum(Y)
sum_xy = 0
sum_x2 = 0

for i in range(n):
    sum_xy += X1[i] * Y[i]
    sum_x2 += X1[i] * X1[i]

m = (n*sum_xy - sum_x*sum_y) / (n*sum_x2 - sum_x*sum_x)
c = (sum_y - m*sum_x) / n

print("Slope (m):", m)
print("Intercept (c):", c)

# Model 𝑦 = 𝑏0+𝑏1𝑥1+𝑏2𝑥2+𝑏3𝑥3yb0+b1x1+b2x2+b3x3	​
b0 = b1 = b2 = b3 = 0
lr = 0.0001

for epoch in range(1000):
    for i in range(len(Y)):
        y_pred = b0 + b1*X1[i] + b2*X2[i] + b3*X3[i]
        error = Y[i] - y_pred

        b0 = b0 + lr * error
        b1 = b1 + lr * error * X1[i]
        b2 = b2 + lr * error * X2[i]
        b3 = b3 + lr * error * X3[i]

print("Coefficients:", b0, b1, b2, b3)

# Evaluation Metrics

def mse(actual, predicted):
    s = 0
    for i in range(len(actual)):
        s += (actual[i] - predicted[i]) ** 2
    return s / len(actual)

# prediction
Y_pred = []
for i in range(len(Y)):
    Y_pred.append(b0 + b1*X1[i] + b2*X2[i] + b3*X3[i])

print("MSE:", mse(Y, Y_pred))

# Part D: Polynomial Regression
# Model  y=b0​+b1​x+b2​x2
b0 = b1 = b2 = 0
lr = 0.00001

for epoch in range(2000):
    for i in range(len(X1)):
        y_pred = b0 + b1*X1[i] + b2*(X1[i]**2)
        error = Y[i] - y_pred

        b0 = b0 + lr * error
        b1 = b1 + lr * error * X1[i]
        b2 = b2 + lr * error * (X1[i]**2)

print("Polynomial Coefficients:", b0, b1, b2)

# Part E: Regularization
lam = 0.1

# Ridge
b1_ridge = b1 - lam * b1
b2_ridge = b2 - lam * b2

# Lasso
if b1 > 0:
    b1_lasso = b1 - lam
else:
    b1_lasso = b1 + lam

#Part F: Model Diagnostics
#Residuals
residuals = []
for i in range(len(Y)):
    residuals.append(Y[i] - Y_pred[i])

print("Residuals:", residuals)    

