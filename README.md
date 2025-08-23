#  Predicting Loan Approval  using Logistic Regression and Decision Trees 
 Predicting Loan Approval  using Logistic Regression and Decision Trees with Regularization and Hyperparameter Tuning

 ## 📖 Project Overview
This project aims to build and compare machine learning models to predict loan eligibility. We implement and tune Logistic Regression (with L1, L2, and ElasticNet regularization) and Decision Tree classifiers, evaluating their performance with and without hyperparameter tuning via GridSearchCV.

## 🧪 Methodology
1.  **Data Preprocessing:** Handled missing values, encoded categorical variables, and scaled numerical features.
2.  **Baseline Model Training:** Trained initial models:
    - Logistic Regression (L1, L2, ElasticNet penalties)
    - Decision Tree
3.  **Hyperparameter Tuning:** Optimized models using `GridSearchCV` to find the best parameters.
4.  **Evaluation:** Models were evaluated based on Accuracy.

## 📈 Results and Comparison

### Model Performance Comparison
The following chart provides a clear comparison of model accuracy before and after hyperparameter tuning. It demonstrates the significant improvement achieved by using GridSearchC for the Decision Tree model,But it was the same accuracy for all the Logistic regression models.

![Model Accuracy Comparison](comparison.png)

**Key Findings:**
*   **No Gridsearch Models:** The Logistic Regression model had the highest baseline accuracy (0.805).
*   **Impact of Gridsearch:** Gridsearch provided a substantial boost to the Decision tree model's performance from(0.805) to (0.8235).
*   **Best Model:** The tuned Decision Tree model (`Decision Tree (CV)`) achieved the highest accuracy.

*For a complete analysis, including confusion matrices and a discussion of precision-recall trade-offs, please see the main Jupyter Notebook.*

## ✅ Conclusion
The analysis reveals that both Logistic Regression and Decision Trees are capable predictors for loan eligibility. The choice of the best model depends on the specific project goals:
- **For highest accuracy:** The **tuned Decision Tree** model is recommended.


Hyperparameter tuning was a critical step, significantly improving the performance and generalizability of the models.

## 👤 Author
[Mostafa Abdelrashid]
