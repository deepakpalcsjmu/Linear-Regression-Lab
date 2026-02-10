# 📊 Linear Regression Models – Complete Study (Assignment-1)

## 🔹 Objective

The objective of this assignment is to implement, analyze, and compare different types of **Linear Regression models** using Python.
This includes:

* Simple Linear Regression
* Multiple Linear Regression
* Polynomial Regression
* Ridge Regression
* Lasso Regression

The aim is to understand how regression models work, evaluate their performance, and interpret results using statistical metrics.

---

## 🔹 Dataset Description

A synthetic dataset (`dataset.csv`) is used for this assignment.

**Dataset contains:**

* 3 input features → `Feature1`, `Feature2`, `Feature3`
* 1 target variable → `Target`

The target variable is generated using a linear combination of input features with some noise, making it suitable for regression analysis.

---

## 🔹 Tools & Libraries Used

* Python
* NumPy
* Pandas
* Matplotlib
* Scikit-learn

These libraries are used for data handling, visualization, model building, and evaluation.

---

## 🔹 Project Workflow

### 1️⃣ Import Libraries

All required libraries are imported:

* NumPy for numerical operations
* Pandas for dataset handling
* Matplotlib for visualization
* Scikit-learn for regression models

---

### 2️⃣ Load Dataset

The CSV dataset is loaded using Pandas:

```python
data = pd.read_csv("dataset.csv")
```

Basic information like shape, columns, and preview of data is displayed.

---

### 3️⃣ Exploratory Data Analysis (EDA)

EDA is performed to understand the dataset.

Steps performed:

* Display first few rows
* Check missing values
* Summary statistics (`mean`, `std`, etc.)
* Correlation between features
* Histograms for feature distribution

This helps in understanding relationships between variables.

---

### 4️⃣ Simple Linear Regression

Simple Linear Regression is applied using **one feature** to predict the target.

Steps:

* Select a single feature
* Train model
* Plot regression line
* Interpret slope and intercept

This shows how one feature affects the target.

---

### 5️⃣ Multiple Linear Regression

Multiple Linear Regression uses **all features** to predict the target.

Steps:

* Select multiple features
* Train model
* Predict values
* Evaluate using:

  * Mean Squared Error (MSE)
  * Root Mean Squared Error (RMSE)
  * R² Score

This model usually performs better than simple regression.

---

### 6️⃣ Polynomial Regression

Polynomial Regression is applied to capture **non-linear relationships**.

Steps:

* Convert features into polynomial features
* Train regression model
* Compare with linear model

This helps when data is curved rather than straight line.

---

### 7️⃣ Ridge Regression

Ridge regression is used to reduce **overfitting**.

* Adds penalty to large coefficients
* Keeps all features but shrinks weights

Helps improve model stability.

---

### 8️⃣ Lasso Regression

Lasso regression performs **feature selection**.

* Some coefficients become zero
* Removes less important features

Useful for identifying important variables.

---

### 9️⃣ Model Evaluation Metrics

Models are evaluated using:

* **MSE** → Average squared error
* **RMSE** → Square root of MSE
* **R² Score** → How well model fits data

These metrics help compare different models.

---

### 🔟 Residual Analysis

Residuals (actual − predicted values) are plotted to check:

* Model accuracy
* Patterns in errors
* Assumptions of regression

If residuals are random → model is good.

---

## 🔹 Results & Observations

* Multiple regression performed better than simple regression.
* Polynomial regression captured non-linear patterns.
* Ridge reduced overfitting by shrinking coefficients.
* Lasso helped in identifying important features.

Overall, the models provided good prediction accuracy.

---

## 🔹 Conclusion

In this assignment, various regression models were implemented and compared.
We learned:

* How to build regression models
* How to evaluate model performance
* How regularization improves models
* How to interpret coefficients

This experiment provides a strong foundation in regression analysis and machine learning concepts.

---

## 🔹 Repository Structure

```
📁 Linear-Regression-Assignment
 ├── dataset.csv
 ├── mlass.py
 ├── README.md
```

---

## 🔹 Author

M.Tech Students
Machine Learning Assignment
