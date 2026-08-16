import streamlit as st
import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Breast Cancer Classification",
    page_icon="🔬",
    layout="wide"
)


# ============================================================
# APPLICATION HEADER
# ============================================================

st.title("🔬 Breast Cancer Classification Dashboard")

st.markdown(
    """
    ### Machine Learning Assignment – 2

    This interactive application demonstrates five machine
    learning classification models using the **Breast Cancer
    Wisconsin (Diagnostic) dataset**.

    The application allows the user to upload test data,
    select a machine learning model, and view its evaluation
    performance.
    """
)

st.divider()


# ============================================================
# DATASET INFORMATION
# ============================================================

with st.expander("📊 Dataset Information", expanded=False):

    st.write(
        """
        **Dataset:** Breast Cancer Wisconsin (Diagnostic)

        **Problem Type:** Binary Classification

        **Target Variable:** `diagnosis`

        **Classes:**
        - B = Benign
        - M = Malignant

        **Number of Features:** 30

        **Machine Learning Models:**
        1. Logistic Regression
        2. Decision Tree
        3. K-Nearest Neighbors (KNN)
        4. Gaussian Naive Bayes
        5. Random Forest
        """
    )


# ============================================================
# LOAD SAVED MODELS
# ============================================================

@st.cache_resource
def load_models():

    models = {
        "Logistic Regression": joblib.load(
            "model/logistic_regression.pkl"
        ),

        "Decision Tree": joblib.load(
            "model/decision_tree.pkl"
        ),

        "KNN": joblib.load(
            "model/knn.pkl"
        ),

        "Naive Bayes": joblib.load(
            "model/naive_bayes.pkl"
        ),

        "Random Forest": joblib.load(
            "model/random_forest.pkl"
        )
    }

    scaler = joblib.load(
        "model/scaler.pkl"
    )

    return models, scaler


# ============================================================
# LOAD MODELS WITH ERROR HANDLING
# ============================================================

try:

    models, scaler = load_models()

except FileNotFoundError as error:

    st.error(
        "❌ Saved model files could not be found."
    )

    st.write(
        "Please make sure the following files exist:"
    )

    st.code(
        """
model/logistic_regression.pkl
model/decision_tree.pkl
model/knn.pkl
model/naive_bayes.pkl
model/random_forest.pkl
model/scaler.pkl
        """
    )

    st.stop()

except Exception as error:

    st.error(
        f"❌ Error loading the saved models: {error}"
    )

    st.stop()


# ============================================================
# SIDEBAR - MODEL SELECTION
# ============================================================

st.sidebar.header("⚙️ Model Configuration")

selected_model = st.sidebar.selectbox(
    "Select Classification Model",
    list(models.keys())
)

st.sidebar.success(
    f"Selected Model:\n{selected_model}"
)

st.sidebar.markdown("---")

st.sidebar.markdown(
    """
    **Available Models**

    • Logistic Regression  
    • Decision Tree  
    • KNN  
    • Naive Bayes  
    • Random Forest
    """
)


# ============================================================
# CSV FILE UPLOAD
# ============================================================

st.header("1️⃣ Upload Test Dataset")

uploaded_file = st.file_uploader(
    "Upload the test_data.csv file",
    type=["csv"],
    help="Upload the test dataset generated during model evaluation."
)


# ============================================================
# WAIT FOR FILE
# ============================================================

if uploaded_file is None:

    st.info(
        "👆 Please upload your test_data.csv file to begin evaluation."
    )

    st.stop()


# ============================================================
# READ TEST DATA
# ============================================================

try:

    test_data = pd.read_csv(
        uploaded_file
    )

except Exception as error:

    st.error(
        f"❌ Unable to read the uploaded CSV file: {error}"
    )

    st.stop()


# ============================================================
# DISPLAY UPLOAD SUCCESS
# ============================================================

st.success(
    "✅ Test dataset uploaded successfully."
)


# ============================================================
# DATASET PREVIEW
# ============================================================

st.subheader("📋 Uploaded Dataset Preview")

st.dataframe(
    test_data.head(10),
    use_container_width=True
)


# ============================================================
# DATASET SUMMARY
# ============================================================

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "Number of Samples",
        test_data.shape[0]
    )

with col2:

    st.metric(
        "Number of Columns",
        test_data.shape[1]
    )

with col3:

    st.metric(
        "Missing Values",
        int(test_data.isnull().sum().sum())
    )


# ============================================================
# VALIDATE TARGET COLUMN
# ============================================================

if "diagnosis" not in test_data.columns:

    st.error(
        """
        ❌ The uploaded dataset does not contain the required
        `diagnosis` column.

        Please upload the test_data.csv generated for this project.
        """
    )

    st.stop()


# ============================================================
# EXPECTED FEATURES
# ============================================================

expected_features = [

    "radius_mean",
    "texture_mean",
    "perimeter_mean",
    "area_mean",
    "smoothness_mean",
    "compactness_mean",
    "concavity_mean",
    "concave_points_mean",
    "symmetry_mean",
    "fractal_dimension_mean",

    "radius_se",
    "texture_se",
    "perimeter_se",
    "area_se",
    "smoothness_se",
    "compactness_se",
    "concavity_se",
    "concave_points_se",
    "symmetry_se",
    "fractal_dimension_se",

    "radius_worst",
    "texture_worst",
    "perimeter_worst",
    "area_worst",
    "smoothness_worst",
    "compactness_worst",
    "concavity_worst",
    "concave_points_worst",
    "symmetry_worst",
    "fractal_dimension_worst"
]


# ============================================================
# CHECK FOR MISSING FEATURES
# ============================================================

missing_features = [

    feature
    for feature in expected_features
    if feature not in test_data.columns

]


if missing_features:

    st.error(
        "❌ Required feature columns are missing."
    )

    st.write(
        "Missing features:"
    )

    st.write(
        missing_features
    )

    st.stop()


# ============================================================
# CHECK FOR EXTRA COLUMNS
# ============================================================

extra_columns = [

    column
    for column in test_data.columns
    if column not in expected_features
    and column != "diagnosis"

]


if extra_columns:

    st.warning(
        "The following extra columns will be ignored:"
    )

    st.write(
        extra_columns
    )


# ============================================================
# SEPARATE FEATURES AND TARGET
# ============================================================

X_test = test_data[
    expected_features
].copy()

y_test = test_data[
    "diagnosis"
].copy()


# ============================================================
# CONVERT TARGET LABELS
# ============================================================

if y_test.dtype == "object":

    y_test = y_test.map(
        {
            "B": 0,
            "M": 1
        }
    )


# ============================================================
# VALIDATE TARGET VALUES
# ============================================================

if y_test.isnull().any():

    st.error(
        """
        ❌ Invalid values were found in the diagnosis column.

        Expected values:
        B = Benign
        M = Malignant
        """
    )

    st.stop()


# ============================================================
# CHECK MISSING VALUES IN FEATURES
# ============================================================

missing_count = X_test.isnull().sum().sum()


if missing_count > 0:

    st.error(
        f"""
        ❌ The uploaded test data contains
        {missing_count} missing feature values.

        Please upload a cleaned test dataset.
        """
    )

    st.stop()


# ============================================================
# GET SELECTED MODEL
# ============================================================

model = models[
    selected_model
]


# ============================================================
# SCALE DATA WHEN REQUIRED
# ============================================================

scaled_models = [

    "Logistic Regression",
    "KNN",
    "Naive Bayes"

]


if selected_model in scaled_models:

    X_model = scaler.transform(
        X_test
    )

else:

    X_model = X_test


# ============================================================
# GENERATE PREDICTIONS
# ============================================================

try:

    y_pred = model.predict(
        X_model
    )

    y_prob = model.predict_proba(
        X_model
    )[:, 1]

except Exception as error:

    st.error(
        f"❌ Prediction failed: {error}"
    )

    st.stop()


# ============================================================
# CALCULATE EVALUATION METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

auc = roc_auc_score(
    y_test,
    y_prob
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

mcc = matthews_corrcoef(
    y_test,
    y_pred
)


# ============================================================
# DISPLAY MODEL PERFORMANCE
# ============================================================

st.header("2️⃣ Model Evaluation")

st.subheader(
    f"Performance of {selected_model}"
)


metric1, metric2, metric3 = st.columns(3)


with metric1:

    st.metric(
        "Accuracy",
        f"{accuracy:.4f}"
    )

    st.metric(
        "Precision",
        f"{precision:.4f}"
    )


with metric2:

    st.metric(
        "AUC",
        f"{auc:.4f}"
    )

    st.metric(
        "Recall",
        f"{recall:.4f}"
    )


with metric3:

    st.metric(
        "F1 Score",
        f"{f1:.4f}"
    )

    st.metric(
        "MCC",
        f"{mcc:.4f}"
    )


# ============================================================
# CONFUSION MATRIX
# ============================================================

st.header("3️⃣ Confusion Matrix")


cm = confusion_matrix(
    y_test,
    y_pred
)


fig, ax = plt.subplots(
    figsize=(6, 4)
)


sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=[
        "Benign",
        "Malignant"
    ],
    yticklabels=[
        "Benign",
        "Malignant"
    ],
    ax=ax
)


ax.set_xlabel(
    "Predicted Label"
)

ax.set_ylabel(
    "Actual Label"
)

ax.set_title(
    f"Confusion Matrix - {selected_model}"
)


st.pyplot(
    fig
)

plt.close(
    fig
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

st.header("4️⃣ Classification Report")


report = classification_report(
    y_test,
    y_pred,
    target_names=[
        "Benign",
        "Malignant"
    ],
    output_dict=True,
    zero_division=0
)


report_df = pd.DataFrame(
    report
).transpose()


st.dataframe(
    report_df.round(4),
    use_container_width=True
)


# ============================================================
# PREDICTION RESULTS
# ============================================================

st.header("5️⃣ Prediction Results")


prediction_results = pd.DataFrame(
    {
        "Actual": y_test.values,
        "Predicted": y_pred
    }
)


prediction_results[
    "Actual Label"
] = prediction_results[
    "Actual"
].map(
    {
        0: "Benign",
        1: "Malignant"
    }
)


prediction_results[
    "Predicted Label"
] = prediction_results[
    "Predicted"
].map(
    {
        0: "Benign",
        1: "Malignant"
    }
)


st.dataframe(
    prediction_results,
    use_container_width=True
)


# ============================================================
# DOWNLOAD PREDICTIONS
# ============================================================

csv_output = prediction_results.to_csv(
    index=False
)


st.download_button(
    label="⬇️ Download Prediction Results",
    data=csv_output,
    file_name="prediction_results.csv",
    mime="text/csv"
)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "M.Tech AIML/DSE | Machine Learning Assignment - 2"
)

st.caption(
    "Breast Cancer Wisconsin (Diagnostic) Dataset"
)