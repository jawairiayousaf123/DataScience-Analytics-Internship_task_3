# Task 3: Customer Churn Prediction & Retention Analysis

## 🎯 Task Objective
Acquiring new customers is mathematically and operationally more expensive for financial institutions than keeping existing ones. The primary objective of this project is to build an predictive modeling pipeline that proactively flags bank customers who are likely to leave the bank (churn). By accurately pinpointing high-risk accounts, retention teams can execute targeted customer-care interventions to minimize capital attrition.

---

## 🛠️ Approach & Methodology

The project structure carries out an advanced end-to-end machine learning operational workflow:

1. **Dataset Auditing & Cleanse:** * Maintained data framework safety by checking for structural gaps via null diagnostics (`df.isnull().sum()`). The dataset proved completely clean with no missing items.
   * Dropped tracking identifiers (`RowNumber`, `CustomerId`, `Surname`) since row labels act as noise and cause model overfitting.

2. **Categorical Feature Engineering:**
   * **Label Encoding:** Converted binary textual elements (`Gender`) into standard integers (0 and 1).
   * **One-Hot Encoding:** Expanded multi-categorical geographical distributions (`Geography`) into discrete dummies via `pd.get_dummies(drop_first=True)` to prevent the dummy variable trap while preserving strict algorithmic readability.

3. **Exploratory Data Analysis (EDA):**
   * Plotted core demographic structures including baseline user churn balance, overall age clusters, and capital balances via Seaborn `histplot` curves to observe distributional patterns.

4. **Model Architecture & Validation:**
   * Split processed feature variables into an **80/20 train-test partition** configuration.
   * Formulated and deployed an ensemble tree model utilizing a **Random Forest Classifier** configured with 100 decision estimators.
   * Evaluated model generalization properties across class types via accuracy benchmarks, classification maps, and localized confusion matrix heatmaps.

5. **Feature Importance Analysis:**
   * Extracted internal tree weights (`model.feature_importances_`) to score and prioritize which metrics carry the highest predictive utility when a customer decides to sever their bank relationship.

---

## 📊 Results and Insights

### 1. Classification Metrics Summary
* **Global Evaluation Accuracy:** Achieved a powerful benchmark prediction capability of **86.6%**, proving that ensemble decision boundary lines perform exceptionally well on customer behavioral matrices.
* **Imbalance Dynamics (Confusion Matrix):** The classifier demonstrates exceptional performance tracking customers who stay with the bank (Class 0). However, similar to real-world churn scenarios, a lower recall for Class 1 highlights that some churning accounts are missed. This shows an opportunity for future tuning (e.g., threshold adjustments or SMOTE sampling).

### 2. Feature Importance Dominance
The Random Forest model identified the specific variables that strongly impact customer retention behaviors:

* **Age Profile:** Customer age emerged as a highly critical predictive driver. Churn patterns change significantly across generational groups.
* **Estimated Salary & Account Balance:** Financial footprint stability scales closely with account security; lower relative asset pools alongside specific pricing tiers serve as direct warning signals for account closure.
* **Product Saturation:** The number of products an active member consumes acts as a key indicator of customer retention; multi-product relationships correlate with significantly lower churn risk.

---

## 📁 Repository File Structure
* `Churn_Modelling.csv` - The source structural banking dataset used for feature mining.
* `task_3file` - Clean, comment-annotated Jupyter Notebook documenting data preparation scripts, model configurations, and output visualizations.
* `README.md` - Executive summary documentation.
