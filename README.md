# 💡 SmartChurn: Predicting Customer Attrition in Banking

### 📘 Overview
This project uses machine learning to predict **which bank customers are likely to churn (leave)**.  
It showcases a complete data-science workflow — from **exploratory analysis** and **preprocessing** to **model comparison**, **hyperparameter tuning**, and **class imbalance handling** using SMOTE.

---

### 🎯 Objective
Identify at-risk customers so the bank can take proactive steps to retain them.  
The focus is not just accuracy — but **recall** (catching potential churners early).

---

### 📊 Dataset
**Source:** Provided bank churn dataset (`Churn (3).csv`)  
- **Rows:** 10,000  
- **Target Variable:** `Exited` (1 = churned, 0 = stayed)  
- **Key Features:**
  - `Age`, `Geography`, `Gender`, `CreditScore`, `Tenure`, `Balance`,  
    `NumOfProducts`, `HasCrCard`, `IsActiveMember`, `EstimatedSalary`

---

### 🧩 Project Workflow

1. **Exploratory Data Analysis (EDA)**
   - Checked class balance and feature distributions.
   - Visualized churn by country, gender, and customer behavior.
   - Found mild class imbalance (~20% churners).

2. **Data Preprocessing**
   - Dropped non-predictive columns (`RowNumber`, `CustomerId`, `Surname`).
   - Encoded categorical variables with **OneHotEncoder**.
   - Scaled numeric features with **StandardScaler**.
   - Built a unified **ColumnTransformer pipeline**.

3. **Model Development**
   - Compared four models:  
     Logistic Regression, Random Forest, Support Vector Machine (SVM), and K-Nearest Neighbors (KNN).
   - Used `accuracy`, `precision`, `recall`, and `F1-score` for fair comparison.

4. **Hyperparameter Tuning**
   - Applied **GridSearchCV** on Random Forest (best baseline model).  
   - Tuned:
     - `n_estimators` (100–150)
     - `max_depth` (None–10)
     - `min_samples_split` (2–4)
   - Improved test performance and model stability.

5. **Handling Class Imbalance with SMOTE**
   - Used **Synthetic Minority Oversampling (SMOTE)** on training data.
   - Improved recall (better at catching churners) with minor precision trade-off.
   - Visualized performance changes before vs after SMOTE.

6. **Feature Importance & Reduced Model**
   - Identified top predictors:  
     `Age`, `Balance`, `CreditScore`, `EstimatedSalary`, `NumOfProducts`.
   - Built a reduced Random Forest model using only the top 5 features to test performance trade-offs.

7. **Optional Extension: Ensemble Model**
   - Combined Random Forest + SVM using a **VotingClassifier** to test ensemble strength.
   - Compared ROC-AUC curves for all models (with and without SMOTE).

---

### ⚙️ Tech Stack

| Category | Tools & Libraries |
|-----------|------------------|
| Language | Python (Google Colab / Jupyter) |
| Data Handling | pandas, numpy |
| Visualization | matplotlib, seaborn |
| ML Algorithms | scikit-learn (LogisticRegression, RandomForest, SVM, KNN) |
| Tuning & Validation | GridSearchCV |
| Imbalance Handling | imblearn (SMOTE) |

---

### 📈 Results Summary

| Model | Accuracy | Precision | Recall | F1 | Key Insight |
|-------|-----------|-----------|--------|----|--------------|
| Logistic Regression | ~0.81 | 0.59 | 0.19 | 0.28 | Baseline |
| Random Forest | ~0.86 | 0.76 | 0.47 | 0.58 | Best overall |
| SVM (RBF) | ~0.86 | 0.85 | 0.40 | 0.54 | High precision |
| KNN | ~0.84 | 0.67 | 0.42 | 0.51 | Distance-sensitive |
| Random Forest + SMOTE | ~0.85 | 0.74 | 0.50 | 0.62 | Higher recall |
| Top 5 Features (RF) | ~0.83 | 0.62 | 0.40 | 0.48 | Simpler but less accurate |

---

### 💡 Key Takeaways
- Built an **end-to-end classification pipeline** with real preprocessing & tuning steps.  
- Learned to handle **class imbalance** properly (SMOTE).  
- Practiced **recall vs precision trade-offs** in a business context.  
- Understood **feature importance** and model interpretability.  

---

### 🚀 How to Run
1. Clone or download the repo.  
2. Install dependencies:
   ```bash
   pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn
