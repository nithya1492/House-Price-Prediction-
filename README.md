# 🏠 House Price Prediction Using Machine Learning

## 📘 Project Overview
This project predicts house prices based on features such as location, number of rooms, and population demographics.  
It uses **regression algorithms** and compares their performance to determine the most accurate model.

**Key Objectives:**
- Predict housing prices using real-world data.
- Apply **data preprocessing** (missing values, outlier handling, scaling).
- Compare **Linear Regression**, **Random Forest**, and **XGBoost** models.
- Evaluate model performance using **RMSE** and **R²**.

---

## 🧠 Skills & Concepts Covered
- Data Cleaning and Preprocessing  
- Feature Engineering and Encoding  
- Regression Modeling (Linear, Ridge, Random Forest, XGBoost)  
- Feature Scaling and Regularization  
- Model Evaluation (RMSE, R²)  
- Data Visualization (Matplotlib, Seaborn)

---

## 📊 Dataset Information
**Dataset:** [California Housing Dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html)  
Available via `scikit-learn`.

**Features include:**
| Feature | Description |
|----------|-------------|
| MedInc | Median income in the area |
| HouseAge | Median house age |
| AveRooms | Average number of rooms per household |
| AveOccup | Average number of household members |
| Latitude / Longitude | Geographical location |
| Population | Area population |
| Target | Median house value (in $) |

---

## ⚙️ Project Workflow

### 1️⃣ Data Preprocessing
- Handle missing values using median/mode imputation.
- Cap outliers at 1st and 99th percentiles.
- Encode categorical features using **One-Hot Encoding**.
- Scale numeric features using **StandardScaler**.
- Create new feature `Region` from Latitude and Longitude.

### 2️⃣ Model Training
Trained and compared the following models:
- **Linear Regression** (baseline)
- **Ridge Regression** (regularized)
- **Random Forest Regressor**
- **XGBoost Regressor**

### 3️⃣ Model Evaluation
Metrics used:
- **RMSE (Root Mean Squared Error)** – measures prediction error  
- **R² (R-squared)** – measures model accuracy  

### 4️⃣ Visualization
- Predicted vs Actual price scatter plots  
- Residual distribution plots  
- RMSE comparison bar chart  
- Feature importance visualization (for Random Forest & XGBoost)

---

## 🏁 Results Summary

| Model | RMSE ($) | R² |
|--------|----------|----|
| XGBoost | ~48,000 | 0.85 |
| Random Forest | ~50,000 | 0.83 |
| Ridge Regression | ~75,000 | 0.72 |
| Linear Regression | ~78,000 | 0.70 |

✅ **Best Model:** XGBoost Regressor (Highest accuracy, lowest RMSE)

---

## 🔍 Key Insights
- **Median Income** is the most influential factor in determining house prices.  
- Houses in **coastal regions** tend to be priced higher.  
- **Ensemble models (Random Forest, XGBoost)** outperform linear models significantly.  
- Proper preprocessing (scaling, encoding) is crucial for model accuracy.

---

## 🚀 Future Improvements
- Experiment with the **Ames Housing dataset** for richer categorical features.
- Perform **hyperparameter tuning** (GridSearchCV, RandomizedSearchCV).
- Try **Deep Learning models** for regression.
- Deploy the model using **Streamlit** or **Flask**.
- Add CI/CD integration for model retraining.

---

## 🧰 Tools and Technologies

| Category | Tools Used |
|-----------|-------------|
| Language | Python |
| Libraries | Pandas, NumPy, Scikit-learn, XGBoost, Matplotlib, Seaborn |
| Environment | Jupyter Notebook / VS Code |
| Version Control | Git, GitHub |

---

## 📁 Project Structure
