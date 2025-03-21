# 📊 IBM Telco Customer Churn Prediction – Data Science Project

This project aims to predict customer churn using various machine learning models and optimization techniques. By analyzing customer demographics, account information, and service usage, the goal is to identify patterns that indicate the likelihood of a customer leaving the company.

🔗 **Live Project Notebook on GitHub:**  
[IBM-Telco-Churn-Prediction](https://github.com/amharringthon/IBM-Telco-Churn-Prediction)

---

## 📁 Project Structure

```
🔹 dataset/                # Contains the IBM Telco dataset  
🔹 notebook.ipynb          # Main Jupyter notebook with full analysis  
🔹 model/                  # Folder containing the exported final model  
🔹 README.md               # Project overview and documentation  
```

---

## 🔍 Dataset

The dataset is based on IBM Telco's customer base and includes:

- Demographic information (e.g., gender, senior citizen)  
- Services subscribed (e.g., internet, tech support)  
- Payment and contract details  
- Customer Lifetime Value (CLTV)  
- Churn status  

ℹ️ The original dataset is publicly available on Kaggle. A copy is included in this repository for convenience.

---

## ✨ Project Workflow

1. **Data Cleaning**  
   - Handled missing values and removed redundant or non-informative features.

2. **Exploratory Data Analysis (EDA)**  
   - Visualized churn patterns and key relationships between features.

3. **Preprocessing**  
   - Applied label encoding, one-hot encoding, and MinMax scaling.

4. **Model Training**  
   - Trained multiple classifiers: Logistic Regression, Random Forest, XGBoost, and LightGBM.

5. **Class Imbalance Handling**  
   - Used SMOTE to address class imbalance and improve recall.

6. **Model Evaluation**  
   - Compared performance using Accuracy, Precision, Recall, F1-Score, and ROC-AUC.

7. **Hyperparameter Tuning**  
   - Performed optimization using Optuna to fine-tune LightGBM with SMOTE.

8. **Model Exporting**  
   - Saved the best model using `joblib` for production use.

---

## 🧠 Final Model – Optimized LightGBM with SMOTE

| Metric              | Value     |
|---------------------|-----------|
| Accuracy            | 85.22%    |
| F1-Score            | 85.73%    |
| ROC-AUC             | 0.93      |
| Cross-Validation    | 84.10%    |
| Recall (Churn)      | 87%       |

✅ The model showed **strong performance**, especially in **recall**, which is key for early churn detection.

---

## 🌟 Business Impact

Deploying this model in production would allow businesses to:

- 🌟 **Proactively identify at-risk customers**  
- 💰 **Reduce churn-related revenue loss**  
- 🏱️ **Create personalized retention strategies based on customer profiles and lifetime value**

---

## 🛠️ Tools & Technologies

- Python (Pandas, NumPy, Scikit-learn)  
- LightGBM, XGBoost, Random Forest  
- SMOTE (Imbalanced-learn)  
- Optuna (for hyperparameter tuning)  
- Seaborn & Matplotlib (visualization)  
- Jupyter Notebook

---

## 📌 How to Use

1. Clone the repository:
   ```bash
   git clone https://github.com/amharringthon/IBM-Telco-Churn-Prediction.git
   ```

2. Navigate to the project folder and run the notebook:
   ```bash
   jupyter notebook
   ```

3. To test the model:
   ```python
   import joblib
   model = joblib.load("model/final_lgbm_model.pkl")
   prediction = model.predict(new_data)
   ```
