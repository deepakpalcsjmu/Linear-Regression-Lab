# 📊 Linear Regression Models – From Scratch Implementation (Assignment-1)

## 🔹 Objective

The objective of this assignment is to implement and compare different **Linear Regression models from scratch** using Python.
All models are implemented using **mathematical formulas and NumPy**, without using any machine-learning library like `sklearn`.

Models implemented:

* Simple Linear Regression
* Multiple Linear Regression
* Polynomial Regression
* Ridge Regression
* Lasso Regression

The goal is to understand how regression works internally, evaluate performance, and interpret results.

---

## 🔹 Dataset Description

A synthetic dataset (`dataset.csv`) is used.

Dataset contains:

* **Feature1**
* **Feature2**
* **Feature3**
* **Target**

The target variable depends on the input features, making it suitable for regression analysis.

---

## 🔹 Tools & Libraries Used

This project does **not use sklearn**.
Only basic libraries are used:

* Python
* NumPy → numerical calculations
* Pandas → dataset handling
* Matplotlib → visualization

All regression algorithms are implemented manually using formulas.

---

## 🔹 Project Workflow

### 1️⃣ Import Libraries

The following libraries are imported:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```

NumPy is used for matrix operations, Pandas for data loading, and Matplotlib for plotting.

---

### 2️⃣ Load Dataset

Dataset is loaded using Pandas:

```python
data = pd.read_csv("dataset.csv")
```

Basic information is displayed:

* First rows
* Data types
* Structure of dataset

---

### 3️⃣ Exploratory Data Analysis (EDA)

EDA is performed to understand the dataset.

Steps:

* Summary statistics using `describe()`
* Missing values check
* Correlation matrix
* Histogram plots

This helps identify relationships between features and target.

---

## 🔹 Model Implementations

### 4️⃣ Simple Linear Regression (Manual)

Simple linear regression is implemented using the formula:

[
y = b0 + b1x
]

Steps:

* Use `Feature1` to predict target
* Calculate slope and intercept manually
* Plot regression line

This shows how one feature affects the target.

---

### 5️⃣ Multiple Linear Regression (Normal Equation)

Multiple regression uses all features.

Formula used:
[
\theta = (X^T X)^{-1} X^T y
]

Steps:

* Add bias column
* Compute coefficients using matrix multiplication
* Predict target values

This model uses all three features.

---

### 6️⃣ Evaluation Metrics (Manual)

Metrics are calculated manually:

* Mean Squared Error (MSE)
* Root Mean Squared Error (RMSE)
* R² Score

These metrics evaluate model performance.

---

### 7️⃣ Polynomial Regression

Polynomial regression is implemented manually.

Steps:

* Create new feature: (x^2)
* Fit model using normal equation
* Compare R² score with linear model

This helps capture non-linear patterns.

---

### 8️⃣ Ridge Regression (Manual)

Ridge regression is implemented using formula:

[
(X^TX + \lambda I)^{-1} X^T y
]

It reduces overfitting by shrinking coefficients.

---

### 9️⃣ Lasso Regression (Manual Gradient Descent)

Lasso regression is implemented using gradient descent.

Steps:

* Initialize weights
* Update using gradient + L1 penalty
* Perform feature shrinkage

Lasso helps in feature selection.

---

### 🔟 Residual Analysis

Residuals are calculated:

[
Residual = Actual − Predicted
]

A residual plot is drawn to check:

* Model accuracy
* Random error distribution

If residuals are random → model is good.

---

## 🔹 Results & Observations

* Multiple regression performed better than simple regression.
* Polynomial regression captured non-linear trends.
* Ridge regression reduced coefficient magnitude.
* Lasso regression pushed some weights towards zero.

Overall, models predicted the target effectively.

---

## 🔹 Conclusion

This assignment demonstrates how regression algorithms work internally without using machine-learning libraries.

We learned:

* Mathematical implementation of regression
* Matrix-based solution
* Gradient descent for Lasso
* Model evaluation techniques

Implementing models from scratch improved understanding of machine-learning fundamentals.

---

## 🔹 Repository Structure

```
📁 Linear-Regression-Lab
 ├── dataset.csv
 ├── mlass.py
 ├── README.md
```

---

## 🔹 Author

M.Tech Student
Machine Learning Lab Assignment