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

## 📊 Analysis & Observations

### 1. Age Distribution
<img width="575" height="457" alt="Age_Distribution" src="https://github.com/user-attachments/assets/6b1271ad-aa05-4d85-a7e5-2513d1aed264" />

- The distribution is right-skewed, indicating that the dataset contains more younger individuals than older ones.
- The mean age is higher than the median, pulled up by the smaller number of older participants.
- The most frequent age group in the data is concentrated in the early 20s.

---

### 2. Result Distribution
<img width="575" height="457" alt="Result_distribution" src="https://github.com/user-attachments/assets/242fb63f-22fd-447c-bd0b-0e9aa5dd8f40" />

- The distribution is left-skewed, showing that most participants achieved higher scores on the assessment.
- The mean score is lower than the median, influenced by a tail of individuals with lower results.
- The highest concentration of scores, or the peak of the distribution, is between **10 and 15**.

---

### 3. Correlation Heatmap
<img width="1241" height="1327" alt="Heatmap" src="https://github.com/user-attachments/assets/bc0d7df0-0a73-4fcc-af99-40bc919c8a18" />

- This heatmap visualizes the correlation strength between all numerical features in the dataset.
- Red cells indicate a strong positive correlation (e.g., A-scores with each other), while blue indicates a negative one.
- The A_Score features show a moderate positive correlation with the final result and the target Class/ASD.

---

### 4. Count Plot of Ethnicity
<img width="575" height="531" alt="Count_plot_ethinicity" src="https://github.com/user-attachments/assets/55a4aac7-f883-46ee-8a6b-a46442a69ba8" />

- 'White-European' and 'Middle Eastern' are the most represented ethnicities in the dataset.
- A large number of entries have missing or unspecified ethnicity, labeled with a '?'.
- The data shows a significant imbalance in ethnic representation, with many groups having very few samples.

---

<img width="437" height="186" alt="Screenshot 2025-09-07 230529" src="https://github.com/user-attachments/assets/15986ee2-cf54-4450-90ba-f43e55e48d0c" />

- This output shows the baseline cross-validation accuracies for three classifiers with default settings.
- Random Forest performed the best initially, achieving the highest accuracy of **0.92**.
- The Decision Tree had the lowest score (**0.85**), while XGBoost was the second-best performer (**0.89**).

---

### 6. Cross-Validation (CV) Scores
<img width="674" height="141" alt="Screenshot 2025-09-07 230539" src="https://github.com/user-attachments/assets/02eb5b0f-a85d-4377-b9fc-eb2b2fb3e2a6" />

- This array displays the individual accuracy scores from each of the 5 cross-validation folds for the models.
- Random Forest's scores are consistently high across all folds, indicating stable performance.
- These scores provide a detailed breakdown of model performance, which is averaged to get the single CV accuracy.

---

### 7. Tuned Decision Tree and Random Forest Scores
<img width="618" height="108" alt="Screenshot 2025-09-07 230550" src="https://github.com/user-attachments/assets/b74dcc1a-8c37-41f5-a37d-5e32ec1e4a14" />

- This shows the specific hyperparameters and resulting accuracy for the tuned Decision Tree and Random Forest models.
- The tuned Random Forest model (**0.912**) significantly outperformed the tuned Decision Tree (**0.848**).
- These results reflect the performance of the models after a hyperparameter optimization process.

---

### 8. Best Model Identification
<img width="778" height="66" alt="Screenshot 2025-09-07 230557" src="https://github.com/user-attachments/assets/cc923b32-a53e-44be-b0ba-77f2ab659d6f" />

- The Random Forest Classifier was selected as the best model based on its superior performance.
- Its final cross-validation accuracy after tuning was **0.91**.
- The optimal configuration for the best model was identified as `bootstrap=False` and `max_depth=10`.

---

### 9. Final Model Evaluation Report
<img width="554" height="303" alt="Screenshot 2025-09-07 230603" src="https://github.com/user-attachments/assets/f756f7d9-3c7d-4faf-ac18-a498009e4707" />

- The best model achieved a final accuracy score of **82.5%** on the unseen test data.
- It performs well on the majority class (0) but shows lower precision (**0.60**) and recall (**0.67**) for the minority class (1).
- The confusion matrix shows the model missed 12 positive cases (false negatives) and had 16 false alarms (false positives).

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

