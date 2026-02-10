# 📘 Linear Regression Models Lab Assignment

## 📌 Assignment  
Comprehensive Study of Linear Regression Models

---

## 🎯 Objective  
The objective of this laboratory experiment is to implement, analyze, and compare different Linear Regression techniques using Python. The experiment focuses on understanding model behavior, performance evaluation, and the role of regularization techniques in improving prediction accuracy.

The following regression models are implemented:
- Simple Linear Regression  
- Multiple Linear Regression  
- Polynomial Regression  
- Ridge Regression  
- Lasso Regression  

---

## 📊 Dataset Used  

### 📁 Dataset Type  
Synthetic dataset generated using Python (NumPy)

### 🧪 Problem Type  
Regression (Prediction of a continuous target variable)

### 🎯 Target Variable  
`final_score` – Represents the final performance score of a student

### 📈 Input Features  
- `hours_studied` – Number of hours spent studying  
- `practice_score` – Practice and skill assessment score  
- `attendance_rate` – Attendance consistency rate  

The dataset contains 100 observations with three independent variables and one continuous dependent variable.

---

## 🛠️ Tools & Technologies  

### Programming Language  
- Python 3

### Libraries Used  
- NumPy  
- Pandas  
- Matplotlib  
- Seaborn  
- Scikit-learn  

---

## 🔬 Methodology  

### 1. Data Preparation  
- Generated a synthetic student performance dataset  
- Verified data quality and structure  
- Selected relevant features and target variable  

---

### 2. Exploratory Data Analysis (EDA)  
- Summary statistics analysis  
- Feature distribution visualization  
- Correlation analysis between variables  

---

### 3. Model Implementation  

#### 📍 Simple Linear Regression  
Implemented using `hours_studied` to analyze its impact on `final_score`.

#### 📍 Multiple Linear Regression  
Implemented using `hours_studied`, `practice_score`, and `attendance_rate` to improve prediction accuracy.

#### 📍 Polynomial Regression  
Applied to capture non-linear relationships between the input features and the target variable.

#### 📍 Ridge Regression  
Used to reduce overfitting by penalizing large coefficient values.

#### 📍 Lasso Regression  
Used to perform feature selection by reducing the influence of less important features.

---

## 📏 Evaluation Metrics  
- Mean Squared Error (MSE)  
- Root Mean Squared Error (RMSE)  
- R² Score  

---

## 📉 Model Diagnostics  
Residual analysis was performed to validate regression assumptions.

---

## 📌 Results & Observations  
- Multiple Linear Regression showed improved performance  
- Polynomial Regression handled non-linear patterns effectively  
- Ridge Regression improved generalization  
- Lasso Regression enhanced model interpretability  

---

## 🌟 Learning Outcomes  
- Practical understanding of linear regression techniques  
- Experience with data analysis and visualization  
- Ability to compare and evaluate regression models  
- Knowledge of regularization methods  

---

## 📂 Repository Structure  

Linear-Regression-Lab

│── linear_regression_assignment.py  
│── linear_regression_dataset.csv  
│── README.md  

---

## ✅ Conclusion  
This lab successfully demonstrates various linear regression models. Proper feature selection and regularization techniques play a key role in building effective predictive models.
