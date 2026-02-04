# 🔥🏃 Calories_Burnt_Prediction

## 📝 Project Overview

The **Calories Burnt Prediction System** is a machine learning solution designed to estimate the energy expenditure of individuals based on their exercise metrics and physiological data. By analyzing factors such as heart rate, duration of exercise, body temperature, and demographics, the system provides accurate calorie burn estimates.

This project covers the complete machine learning pipeline, from data ingestion, random sample imputation, and variable transformation to feature selection, model training, and deployment through a **Flask web application**.

## 🎯 Main Goal
The main goal of the **Calories Burnt Prediction System** is to **assist fitness enthusiasts and healthcare providers** by providing a reliable tool to track calorie expenditure, helping users manage their fitness goals and monitor workout intensity effectively.

## 📁 Dataset Description
The project utilizes data merging two sources: exercise data and calorie data, joined by a unique user identifier. It captures physical performance metrics and physiological stats.

| Feature | Description |
| :--- | :--- |
| **User_ID** | Unique identifier for each user |
| **Gender** | Gender of the user (Male / Female) |
| **Age** | Age of the user in years |
| **Height** | Height of the user in centimeters |
| **Weight** | Weight of the user in kilograms |
| **Duration** | Duration of the exercise in minutes |
| **Heart_Rate** | Heart rate in beats per minute (bpm) during exercise |
| **Body_Temp** | Body temperature in Celsius during exercise |
| **Calories** | (Target) Total calories burned during the session |

### 🏷️ Dataset Categories
- **Target Variable:** `Calories` (Numerical)
- **Demographics:** `Gender`, `Age`, `Height`, `Weight`
- **Physiological/Performance:** `Duration`, `Heart_Rate`, `Body_Temp`

---

## 🗂️ Project Structure

```text
Calories_Burnt_Prediction/
│
├─ data/
│   ├─ exercise.csv           # Feature data
│   └─ calories.csv           # Target data
│
├─ app.py                     # Flask web application entry point
├─ main.py                    # Main script to run the ML pipeline
├─ random_sample_imputataion.py # Handling missing values
├─ variable_transformation.py # Log transform & Quantile capping
├─ feature_selection.py       # Variance threshold & Correlation checks
├─ Scaling.py                 # Feature scaling (StandardScaler)
├─ model_training.py          # Linear Regression training script
├─ log_code.py                # Logging configuration
│
├─ models/
│   ├─ calories.pkl           # Trained Linear Regression model
│   ├─ scaling.pkl            # Saved StandardScaler object
│
├─ templates/
│   └─ index.html             # Frontend HTML for Flask app
│
├─ plot_path/                 # Saved KDE and Boxplots for analysis
│
├─ requirements.txt           # Required Python packages
└─ README.md                  # Project documentation


```
# 🔄 ML - Pipeline

The system uses a modular machine learning pipeline to ensure data quality and model accuracy. It handles missing values using random sampling, normalizes skewed data, encodes categorical variables, and scales features before training a regression model.

---

### 📊 Data Visualization
- **Library:** Matplotlib & Seaborn  
- **Techniques used:** KDE Plots (Kernel Density Estimation), Boxplots  
- **Purpose:** Visualize data distribution and detect outliers before and after variable transformation  

---

### 🛠️ Feature Engineering

#### 1️⃣ Handling Missing Values
- **Script:** `random_sample_imputation.py`  
- **Technique:** Random Sample Imputation  
- **Method:** Missing values in training and testing sets are filled by sampling random values from observed data  
- **Reason:** Preserves original variance and distribution better than mean/median imputation  

#### 2️⃣ Variable Transformation
- **Script:** `variable_transformation.py`  
- **Techniques:**  
  - Log Transformation: `np.log1p` to numeric features to reduce skewness  
  - Quantile Capping: Outliers capped between 1st (0.01) and 99th (0.99) percentiles based on training data  
- **Reason:** Normalizes data distribution for linear regression  

#### 3️⃣ Categorical Encoding
- **Script:** `main.py`  
- **Technique:** One-Hot Encoding  
- **Target:** `Gender` column  
- **Method:** Converts Gender into numeric representation (dropping the first category to avoid multicollinearity)  

#### 4️⃣ Feature Selection (Hypothesis Testing)
- **Script:** `feature_selection.py`  
- **Techniques:**  
  - Constant/Quasi-Constant Removal: Removes features with 0 variance  
  - Pearson Correlation: Checks correlation between features and target  
- **Outcome:** Only statistically significant features contribute to the model  

---

### ⚖️ Feature Scaling
- **Script:** `Scaling.py`  
- **Technique:** StandardScaler  
- **Method:** Standardizes features by removing the mean and scaling to unit variance  
- **Reason:** Ensures features with larger magnitudes do not dominate the Linear Regression objective  

---

### 🧠 Model Training
- **Model:** Linear Regression (`sklearn.linear_model.LinearRegression`)  
- **Workflow:**  
  1. Data split into Train/Test sets (80/20)  
  2. Model trained on scaled data  
  3. Model serialized (`calories.pkl`) for deployment  

---

### 🌐 Deployment (Flask Web App)
- **Script:** `app.py`  
- **Frontend:** HTML form taking inputs for Age, Height, Weight, Duration, Heart Rate, Body Temp, and Gender  
- **Backend:**  
  - Loads `calories.pkl` and `scaling.pkl`  
  - Preprocesses user input (Gender → numeric, scales data)  
  - Returns predicted calorie count  

---

### 📦 Install Required Packages
```bash
pip install -r requirements.txt
```

## **🚀 Run the Project**
> ```
> python main.py
> python app.py
> ```
---


## **👤 Author**
 ```
 Varadhana Varshini Kolipakula
 Machine Learning & Data Science Enthusiast
 ```

---
