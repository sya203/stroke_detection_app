# ============================================================
# STROKE PREDICTION WEB APPLICATION
# XGBoost + DNN Stacking Hybrid
# ============================================================

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import shap
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Stroke Prediction System",
    page_icon="🩺",
    layout="centered"
)


# ============================================================
# MODEL PATHS
# ============================================================

MODEL_DIR = "models"

PREPROCESSOR_PATH = os.path.join(
    MODEL_DIR,
    "preprocessor.pkl"
)

XGB_PATH = os.path.join(
    MODEL_DIR,
    "final_xgb.pkl"
)

DNN_PATH = os.path.join(
    MODEL_DIR,
    "final_dnn.keras"
)

META_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "final_meta_model.pkl"
)

THRESHOLD_PATH = os.path.join(
    MODEL_DIR,
    "stack_threshold.pkl"
)


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_models():

    preprocessor = joblib.load(
        PREPROCESSOR_PATH
    )

    xgb_model = joblib.load(
        XGB_PATH
    )

    dnn_model = load_model(
        DNN_PATH
    )

    meta_model = joblib.load(
        META_MODEL_PATH
    )

    threshold = joblib.load(
        THRESHOLD_PATH
    )

    return (
        preprocessor,
        xgb_model,
        dnn_model,
        meta_model,
        float(threshold)
    )


# ============================================================
# LOAD DEPLOYMENT MODELS
# ============================================================

try:

    (
        preprocessor,
        xgb_model,
        dnn_model,
        meta_model,
        threshold

    ) = load_models()

except Exception as e:

    st.error(
        "Unable to load the deployment models."
    )

    st.exception(e)

    st.stop()

# ============================================================
# BMI GROUPING
# ============================================================

def create_bmi_group(bmi):

    if bmi < 18.5:

        return "Underweight"

    elif bmi < 25:

        return "Normal"

    elif bmi < 30:

        return "Overweight"

    else:

        return "Obese"


# ============================================================
# APPLICATION HEADER
# ============================================================

st.title(
    "Stroke Prediction System"
)

st.write(
    "Enter the patient's information below to "
    "generate a stroke-risk prediction using "
    "the proposed XGBoost–DNN stacking model."
)

st.divider()


# ============================================================
# PATIENT INFORMATION
# ============================================================

st.subheader(
    "Patient Information"
)


# ------------------------------------------------------------
# AGE
# ------------------------------------------------------------

age = st.number_input(

    "Age",

    min_value=0.0,

    max_value=120.0,

    value=50.0,

    step=1.0

)


# ------------------------------------------------------------
# GENDER
# ------------------------------------------------------------

gender = st.selectbox(

    "Gender",

    options=[
        "Male",
        "Female",
        "Other"
    ]

)


# ------------------------------------------------------------
# HYPERTENSION
# ------------------------------------------------------------

hypertension = st.selectbox(

    "Hypertension",

    options=[
        0,
        1
    ],

    format_func=lambda x:
        "No" if x == 0 else "Yes"

)


# ------------------------------------------------------------
# HEART DISEASE
# ------------------------------------------------------------

heart_disease = st.selectbox(

    "Heart Disease",

    options=[
        0,
        1
    ],

    format_func=lambda x:
        "No" if x == 0 else "Yes"

)


# ------------------------------------------------------------
# EVER MARRIED
# ------------------------------------------------------------

ever_married = st.selectbox(

    "Ever Married",

    options=[
        "Yes",
        "No"
    ]

)


# ------------------------------------------------------------
# WORK TYPE
# ------------------------------------------------------------

work_type = st.selectbox(

    "Work Type",

    options=[
        "Private",
        "Self-employed",
        "Govt_job",
        "children",
        "Never_worked"
    ]

)


# ------------------------------------------------------------
# RESIDENCE TYPE
# ------------------------------------------------------------

residence_type = st.selectbox(

    "Residence Type",

    options=[
        "Urban",
        "Rural"
    ]

)


# ------------------------------------------------------------
# SMOKING STATUS
# ------------------------------------------------------------

smoking_status = st.selectbox(

    "Smoking Status",

    options=[
        "formerly smoked",
        "never smoked",
        "smokes",
        "Unknown"
    ]

)


# ------------------------------------------------------------
# AVERAGE GLUCOSE LEVEL
# ------------------------------------------------------------

avg_glucose_level = st.number_input(

    "Average Glucose Level",

    min_value=0.0,

    max_value=500.0,

    value=100.0,

    step=0.1

)


# ------------------------------------------------------------
# BMI
# ------------------------------------------------------------

bmi = st.number_input(

    "BMI",

    min_value=0.0,

    max_value=100.0,

    value=25.0,

    step=0.1

)


# ============================================================
# PREDICTION BUTTON
# ============================================================

st.divider()

predict_button = st.button(

    "Predict Stroke Risk",

    type="primary",

    use_container_width=True

)


# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    # --------------------------------------------------------
    # BMI GROUP
    # --------------------------------------------------------

    bmi_group = create_bmi_group(
        bmi
    )


    # --------------------------------------------------------
    # CREATE INPUT DATAFRAME
    # --------------------------------------------------------

    input_data = pd.DataFrame({

        "gender": [
            gender
        ],

        "hypertension": [
            hypertension
        ],

        "heart_disease": [
            heart_disease
        ],

        "ever_married": [
            ever_married
        ],

        "work_type": [
            work_type
        ],

        "Residence_type": [
            residence_type
        ],

        "smoking_status": [
            smoking_status
        ],

        "bmi_group": [
            bmi_group
        ],

        "age": [
            age
        ],

        "avg_glucose_level": [
            avg_glucose_level
        ]

    })


    # --------------------------------------------------------
    # PREPROCESS INPUT
    # --------------------------------------------------------

    try:

        X_processed = (
            preprocessor.transform(
                input_data
            )
        )

    except Exception as e:

        st.error(
            "An error occurred during preprocessing."
        )

        st.exception(e)

        st.stop()


    # --------------------------------------------------------
    # VERIFY FEATURE COUNT
    # --------------------------------------------------------

    if X_processed.shape[1] != 26:

        st.error(

            "Unexpected number of processed features: "
            f"{X_processed.shape[1]}. "
            "Expected 26."

        )

        st.stop()


    # --------------------------------------------------------
    # XGBOOST PREDICTION
    # --------------------------------------------------------

    try:

        xgb_probability = (

            xgb_model
            .predict_proba(
                X_processed
            )[0, 1]

        )

    except Exception as e:

        st.error(
            "XGBoost prediction failed."
        )

        st.exception(e)

        st.stop()

    # --------------------------------------------------------
    # DNN PREDICTION
    # --------------------------------------------------------

    try:

        dnn_probability = (

            dnn_model
            .predict(
                X_processed,
                verbose=0
            )
            .reshape(-1)[0]

        )

    except Exception as e:

        st.error(
            "DNN prediction failed."
        )

        st.exception(e)

        st.stop()


    # --------------------------------------------------------
    # STACKING INPUT
    # --------------------------------------------------------

    stacking_input = pd.DataFrame({

        "XGB_Probability": [
            xgb_probability
        ],

        "DNN_Probability": [
            dnn_probability
        ]

    })

    # --------------------------------------------------------
    # STACKING PREDICTION
    # --------------------------------------------------------

    try:

        hybrid_probability = (

            meta_model
            .predict_proba(
                stacking_input
            )[0, 1]

        )
        
    except Exception as e:

        st.error(
            "Stacking prediction failed."
        )

        st.exception(e)

        st.stop()


    # --------------------------------------------------------
    # FINAL CLASSIFICATION
    # --------------------------------------------------------

    prediction = int(

        hybrid_probability
        >= threshold

    )


    # ========================================================
    # DISPLAY RESULTS
    # ========================================================

    st.divider()

    st.subheader(
        "Prediction Result"
    )


    # --------------------------------------------------------
    # BMI GROUP
    # --------------------------------------------------------

    st.info(
        f"Derived BMI Group: **{bmi_group}**"
    )


    # --------------------------------------------------------
    # FINAL RESULT
    # --------------------------------------------------------

    if prediction == 1:

        st.error(
            "### Stroke Prediction: Positive"
        )

        st.write(
            "The hybrid model predicts that the "
            "patient is classified as **Stroke** "
            "at the selected decision threshold."
        )

    else:

        st.success(
            "### Stroke Prediction: Negative"
        )

        st.write(
            "The hybrid model predicts that the "
            "patient is classified as **No Stroke** "
            "at the selected decision threshold."
        )


    # --------------------------------------------------------
    # HYBRID PROBABILITY
    # --------------------------------------------------------

    st.metric(

        "Hybrid Stroke Probability",

        f"{hybrid_probability * 100:.2f}%"

    )


    st.write(
        f"Decision threshold: "
        f"**{threshold * 100:.0f}%**"
    )


    # ========================================================
    # BASE MODEL PROBABILITIES
    # ========================================================

    st.subheader(
        "Base Model Predictions"
    )


    col1, col2 = st.columns(2)


    with col1:

        st.metric(

            "XGBoost",

            f"{xgb_probability * 100:.2f}%"

        )


    with col2:

        st.metric(

            "DNN",

            f"{dnn_probability * 100:.2f}%"

        )

    # ============================================================
    # XGBOOST LOCAL SHAP EXPLANATION
    # ============================================================

    def generate_xgb_local_shap(
        X_processed,
        feature_names,
        top_n=10
    ):
        """
        Generate a local SHAP explanation for one observation
        using the final XGBoost model.
        """

        # Create SHAP explainer
        explainer = shap.TreeExplainer(xgb_model)

        # Calculate SHAP values
        shap_explanation = explainer(X_processed)

        # Extract SHAP values for the single observation
        shap_values = shap_explanation.values[0]

        # Extract processed feature values
        feature_values = X_processed[0]

        # Build explanation dataframe
        explanation_df = pd.DataFrame({
            "Feature": feature_names,
            "Processed_Value": feature_values,
            "SHAP_Value": shap_values,
            "Absolute_SHAP": np.abs(shap_values)
        })

        # Sort by importance
        explanation_df = (
            explanation_df
            .sort_values(
                "Absolute_SHAP",
                ascending=False
            )
            .head(top_n)
            .reset_index(drop=True)
        )

        return explanation_df, shap_explanation

    feature_names = (
        preprocessor
        .get_feature_names_out()
    )

    print("Number of features:", len(feature_names))
    print(feature_names)

    # ========================================================
    # EXPLAINABILITY SECTION
    # ========================================================

    st.markdown("---")

    st.subheader("Why did the model make this prediction?")

    st.write(
        "The chart below shows the features that had the greatest "
        "influence on the XGBoost component of the hybrid model "
        "for this individual prediction."
    )

    xgb_explanation_df, xgb_shap_explanation = (
        generate_xgb_local_shap(
            X_processed,
            feature_names,
            top_n=10
        )
    )

    # ========================================================
    # VISUALIZE FEATURE CONTRIBUTIONS
    # ========================================================

    import plotly.express as px

    plot_df = xgb_explanation_df.copy()

    # Positive = pushes toward Stroke
    # Negative = pushes toward No Stroke

    plot_df["Direction"] = np.where(
        plot_df["SHAP_Value"] >= 0,
        "Pushes toward Stroke",
        "Pushes toward No Stroke"
    )

    plot_df["Impact"] = plot_df["SHAP_Value"].abs()

    plot_df = plot_df.sort_values(
        "SHAP_Value",
        ascending=True
    )

    fig = px.bar(
        plot_df,
        x="SHAP_Value",
        y="Feature",
        orientation="h",
        color="Direction",
        hover_data={
            "Processed_Value": ":.3f",
            "SHAP_Value": ":.3f",
            "Impact": ":.3f",
            "Feature": True,
            "Direction": True
        },
        labels={
            "SHAP_Value": "SHAP value",
            "Feature": "Feature"
        },
        title="Top Features Influencing the Prediction"
    )

    fig.add_vline(
        x=0,
        line_width=1
    )

    fig.update_layout(
        height=500,
        showlegend=True,
        legend_title_text="Contribution"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    display_df = xgb_explanation_df.copy()

    display_df["SHAP_Value"] = (
        display_df["SHAP_Value"]
        .round(4)
    )

    display_df["Processed_Value"] = (
        display_df["Processed_Value"]
        .round(4)
    )

    display_df = display_df.rename(
        columns={
            "Feature": "Feature",
            "Processed_Value": "Processed Value",
            "SHAP_Value": "SHAP Impact"
        }
    )

    st.dataframe(
        display_df[
            [
                "Feature",
                "Processed Value",
                "SHAP Impact"
            ]
        ],
        use_container_width=True,
        hide_index=True
    )
    
    # ========================================================
    # MODEL INFORMATION
    # ========================================================

    with st.expander(
        "Model Information"
    ):

        st.write(
            "The prediction is generated using "
            "an XGBoost and DNN stacking hybrid model."
        )

        st.write(
            f"Classification threshold: "
            f"{threshold:.2f}"
        )

        st.write(
            "The input data is transformed using "
            "the same preprocessing pipeline used "
            "during model development."
        )

        st.write(
            "The continuous BMI value is converted "
            "into the corresponding BMI group before "
            "prediction."
        )
