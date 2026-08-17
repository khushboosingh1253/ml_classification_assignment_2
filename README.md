# Machine Learning Classification Assignment 2

## M.Tech AIML / DSE — Machine Learning Assignment 2

This project implements multiple machine learning classification algorithms on the **Breast Cancer Wisconsin (Diagnostic)** dataset. The trained models are evaluated using multiple classification metrics and deployed through an interactive Streamlit web application.

## a. Problem Statement

The objective of this project is to develop and evaluate machine learning classification models for predicting whether a breast tumor is **benign** or **malignant** based on numerical measurements computed from digitized images of breast mass samples.

The project implements multiple classification algorithms on the same dataset and compares their performance using:

* Accuracy
* AUC Score
* Precision
* Recall
* F1 Score
* Matthews Correlation Coefficient (MCC)

The best-performing model is then made available through an interactive Streamlit application.

---

## b. Dataset Description

### Dataset Name

**Breast Cancer Wisconsin (Diagnostic) Dataset**

### Dataset Source

The dataset was obtained from the **UCI Machine Learning Repository**.

### Dataset URL

https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data

### Dataset Characteristics

* 569 instances
* 30 numerical features
* 2 classes: Benign (B), Malignant (M)

The `id` column was removed and the target variable was encoded as:

* B → 0
* M → 1

---

## c. GitHub Repository Link

**GitHub Repository:**
https://github.com/khushboosingh1253/ml_classification_assignment_2

---

## d. Models Used

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbor (KNN)
4. Gaussian Naive Bayes
5. Random Forest Classifier

---

## Model Comparison

| Model               |   Accuracy |        AUC |  Precision |     Recall |         F1 |        MCC |
| ------------------- | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: |
| Logistic Regression |     0.9649 |     0.9960 |     0.9750 |     0.9286 |     0.9512 |     0.9245 |
| Decision Tree       |     0.9298 |     0.9246 |     0.9048 |     0.9048 |     0.9048 |     0.8492 |
| KNN                 |     0.9561 |     0.9823 |     0.9744 |     0.9048 |     0.9383 |     0.9058 |
| Naive Bayes         |     0.9211 |     0.9891 |     0.9231 |     0.8571 |     0.8889 |     0.8292 |
| **Random Forest**   | **0.9737** | **0.9929** | **1.0000** | **0.9286** | **0.9630** | **0.9442** |

---


## Overall Best Model

✅ **Random Forest** achieved the best performance with:

* Accuracy: 97.37%
* AUC: 99.29%
* Precision: 100%
* MCC: 0.9442

---

## Streamlit Application

The project includes an interactive Streamlit web application for model testing and evaluation.

### Features

* Upload dataset (CSV)
* Select ML model
* Generate predictions
* View metrics (Accuracy, AUC, Precision, Recall, F1, MCC)
* Confusion matrix
* Classification report

---

## Streamlit App Link

👉 https://mlclassificationassignment2.streamlit.app/

---

## Project Structure

```text
ML-Assignment-2/
│
├── app.py
├── requirements.txt
├── test_data.csv
│
└── model/
    ├── logistic_regression.pkl
    ├── decision_tree.pkl
    ├── knn.pkl
    ├── naive_bayes.pkl
    ├── random_forest.pkl
    └── scaler.pkl
```

---

## How to Run

```bash
git clone https://github.com/khushboosingh1253/ml_classification_assignment_2
cd ml_classification_assignment_2
pip install -r requirements.txt
streamlit run app.py
```

---

## Conclusion

Random Forest performed best among all models and was selected as the final model for deployment in the Streamlit application.

---
