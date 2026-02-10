# 🌍 Air Quality Linear Regression Lab

## 📌 Assignment
Comprehensive Study of Linear Regression Models

---

## 🎯 Objective
The objective of this lab is to implement, analyze, and compare different Linear Regression techniques using a real-world dataset. The models implemented include:

- Simple Linear Regression
- Multiple Linear Regression
- Polynomial Regression
- Ridge Regression (Regularization)
- Lasso Regression (Regularization)

---

## 📊 Dataset Used
**Air Quality Dataset (UCI Repository)**

### 📁 File
`AirQualityUCI.csv`

### 🧪 Problem Type
Regression (Predict Continuous Value)

### 🎯 Target Variable
`CO(GT)` → Carbon Monoxide concentration

### 📈 Input Features Used
- `NOx(GT)` → Nitrogen Oxides concentration  
- `C6H6(GT)` → Benzene concentration  
- `T` → Temperature  
- `RH` → Relative Humidity  

---

## 🛠️ Technologies & Libraries Used

### Programming Language
- Python 3

### Libraries
- Pandas → Data Handling
- NumPy → Numerical Computations
- Matplotlib → Visualization
- Seaborn → Statistical Visualization
- Scikit-Learn → Machine Learning Models

---

## 🔬 Methodology

### ✅ 1. Data Preprocessing
- Removed missing values (`-200` replaced with NaN)
- Dropped empty columns
- Selected important features
- Removed rows with missing target values

---

### ✅ 2. Exploratory Data Analysis (EDA)
- Correlation Heatmap
- Feature Distribution Analysis
- Relationship Understanding between pollutants and environment factors

---

### ✅ 3. Model Implementation

#### 📍 Simple Linear Regression
Predict CO concentration using Temperature.

#### 📍 Multiple Linear Regression
Predict CO concentration using:
- NOx
- Benzene
- Temperature
- Humidity

#### 📍 Polynomial Regression
Capture non-linear relationship between Temperature and CO.

#### 📍 Ridge Regression
Reduce overfitting by shrinking coefficient values.

#### 📍 Lasso Regression
Perform feature selection and reduce less important feature influence.

---

## 📏 Evaluation Metrics Used

- **MSE (Mean Squared Error)** → Measures average squared prediction error  
- **RMSE (Root Mean Squared Error)** → Actual error magnitude  
- **R² Score** → Model accuracy (closer to 1 = better)

---

## 📉 Model Diagnostics
Residual plots were used to verify:
- Linearity
- Error distribution
- Model reliability

---

## 📌 Results & Observations

- Multiple Linear Regression performed better than Simple Linear Regression.
- Polynomial Regression captured non-linear relationships.
- Ridge Regression improved model stability.
- Lasso Regression helped identify important features.

---

## 🌟 Key Learning Outcomes
- Understanding real-world regression problems
- Data cleaning and preprocessing skills
- Model comparison techniques
- Importance of regularization
- Model evaluation using statistical metrics
