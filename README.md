# 📊 Assignment-1: Comprehensive Study of Linear Regression Models (Manual Implementation)

## Objective

The objective of this assignment is to perform a complete regression analysis using a synthetic dataset and implement different **linear regression models from scratch** using Python.

Unlike standard implementations, this project **does not use any machine learning library such as scikit-learn**.
All regression algorithms are implemented manually using mathematical formulas and NumPy operations to understand the internal working of models.

Models implemented:

* Simple Linear Regression
* Multiple Linear Regression
* Polynomial Regression
* Ridge Regression
* Lasso Regression

The goal is to understand how regression works mathematically and how model performance is evaluated.

---

## Dataset Description

* Dataset: `dataset.csv`
* Type: Synthetic numerical dataset
* Rows: Depends on generated dataset
* Columns:

  * `Feature1`
  * `Feature2`
  * `Feature3`
  * `Target`

The target variable is generated using a linear combination of the three features with some noise.
This makes the dataset suitable for testing different regression techniques.

---

## Tools & Technologies

* Python
* NumPy
* Pandas
* Matplotlib
* Jupyter Notebook

⚠️ **Important:**
No machine learning library (like scikit-learn) is used.
All models are implemented manually using formulas.

---

## Project Workflow

### 1️⃣ Data Loading

The dataset is loaded using pandas:

```python
data = pd.read_csv("dataset.csv")
```

Basic information such as head, data types, and structure of the dataset is displayed.

---

### 2️⃣ Exploratory Data Analysis (EDA)

EDA is performed to understand the dataset:

* Summary statistics (mean, std, etc.)
* Missing value check
* Correlation between variables
* Histogram plots for feature distribution

This helps in understanding relationships between features and target.

---

### 3️⃣ Simple Linear Regression (Manual)

Simple Linear Regression is implemented using mathematical formulas:

Slope formula:
[
b1 = \frac{\sum (x-x̄)(y-ȳ)}{\sum (x-x̄)^2}
]

Intercept:
[
b0 = ȳ - b1x̄
]

A regression line is plotted to visualize the relationship between Feature1 and Target.

---

### 4️⃣ Multiple Linear Regression (Normal Equation)

Multiple regression uses all three features:

[
\theta = (X^TX)^{-1}X^Ty
]

Steps:

* Add bias column
* Apply normal equation
* Compute coefficients
* Predict values

This allows prediction using multiple variables.

---

### 5️⃣ Model Evaluation Metrics (Manual)

The following metrics are implemented manually:

* Mean Squared Error (MSE)
* Root Mean Squared Error (RMSE)
* R² Score

These metrics evaluate how well the model fits the data.

---

### 6️⃣ Polynomial Regression

Polynomial regression of degree 2 is implemented manually:

* Create new feature (x^2)
* Apply normal equation
* Compare R² score with linear model

Used to capture non-linear patterns.

---

### 7️⃣ Ridge Regression (Manual)

Ridge regression adds regularization to reduce overfitting:

[
\theta = (X^TX + \lambda I)^{-1}X^Ty
]

This shrinks coefficients but keeps all features.

---

### 8️⃣ Lasso Regression (Manual Gradient Descent)

Lasso regression is implemented using gradient descent with L1 penalty.

Steps:

* Initialize weights
* Compute gradient
* Update weights iteratively
* Apply L1 regularization

This can reduce some coefficients toward zero.

---

### 9️⃣ Residual Analysis

Residuals are calculated:
[
Residual = Actual - Predicted
]

A residual plot is drawn to check:

* Prediction accuracy
* Error patterns
* Model assumptions

Random residual distribution indicates a good model.

---

## Results & Observations

* Simple regression shows basic linear relationship.
* Multiple regression improves accuracy using all features.
* Polynomial regression captures non-linear patterns.
* Ridge reduces overfitting by shrinking weights.
* Lasso performs coefficient regularization.

Overall, manually implemented models produce reasonable predictions.

---

## Conclusion

This assignment demonstrates how regression algorithms work internally without using machine learning libraries.
By implementing formulas manually, we gain deeper understanding of:

* Regression mathematics
* Model evaluation
* Regularization techniques
* Error analysis

This project builds a strong conceptual foundation for machine learning.

---

## Project Structure

```
📁 Regression-Assignment
 ├── dataset.csv
 ├── ass.ipynb
 ├── mlass.py
 └── README.md
```

---

## Author

M.Tech Student
Machine Learning Lab Assignment-1
