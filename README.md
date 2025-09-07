# Autism Prediction using Machine Learning

This project aims to predict whether an individual is likely to have Autism Spectrum Disorder (ASD) using machine learning techniques. It leverages data preprocessing, feature engineering, and advanced classification algorithms to deliver an efficient predictive model that can aid healthcare professionals in early screening and diagnosis.

---

## 📌 Features

- Comprehensive data preprocessing including handling missing values and encoding categorical features.
- Exploratory Data Analysis (EDA) with insightful visualizations to understand feature distributions and correlations.
- Addressing class imbalance using SMOTE to improve model fairness.
- Implementation of robust machine learning models:
  - Decision Tree Classifier
  - Random Forest Classifier
  - XGBoost Classifier
- Hyperparameter tuning using `RandomizedSearchCV` to optimize model performance.
- Model evaluation with accuracy, confusion matrix, and classification reports.
- Saving and loading trained models using Pickle for future predictions.

---

## 📂 Dataset

The dataset includes behavioral and demographic features collected from individuals, along with labels indicating whether they are diagnosed with autism. It is used to train models that can distinguish between autistic and non-autistic cases.

---

## 🛠 Technologies Used

- Python
- Pandas, NumPy – for data manipulation
- Matplotlib, Seaborn – for data visualization
- Scikit-learn – for machine learning algorithms and evaluation
- Imbalanced-learn (SMOTE) – to handle class imbalance
- XGBoost – for gradient boosting classification
- Pickle – to save and reuse trained models

---

## 🚀 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/autism-prediction.git
   cd autism-prediction

## 📊 How to Use

Load the dataset and understand the features using visualizations.

Preprocess the data:

Handle missing values.

Encode categorical variables.

Use SMOTE to balance the dataset if necessary.

Split the data into training and testing sets.

Train models like Decision Tree, Random Forest, and XGBoost.

Optimize models using hyperparameter tuning.

Evaluate models based on accuracy, precision, recall, and F1-score.

Save the best-performing model using Pickle.

Load the saved model to make predictions on new data.

## 📖 Results

The notebook provides detailed insights into feature importance and data distribution.

Different models are compared using various evaluation metrics.

The final model can be deployed to assist in medical diagnosis processes.

