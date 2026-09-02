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

DNN_SHAP_BACKGROUND_PATH = os.path.join(
    MODEL_DIR,
    "dnn_shap_background.npy"
)


# ============================================================
# LOAD MODELS
# ============================================================

#@st.cache_resource
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

    # Load DNN SHAP background if available
    dnn_shap_background = None

    if os.path.exists(
        DNN_SHAP_BACKGROUND_PATH
    ):

        dnn_shap_background = np.load(
            DNN_SHAP_BACKGROUND_PATH
        ).astype(np.float32)

    return (
        preprocessor,
        xgb_model,
        dnn_model,
        meta_model,
        float(threshold),
        dnn_shap_background
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
        threshold,
        dnn_shap_background

    ) = load_models()

except Exception as e:

    st.error(
        "Unable to load the deployment models."
    )

    st.exception(e)

    st.stop()


# ============================================================
# SHAP EXPLAINERS
# ============================================================

#@st.cache_resource
def create_xgb_explainer(
    model
):

    return shap.TreeExplainer(
        model
    )


xgb_explainer = create_xgb_explainer(
    xgb_model
)


#@st.cache_resource
def create_dnn_explainer(
    model,
    background
):

    if background is None:

        return None

    return shap.DeepExplainer(
        model,
        background
    )


try:

    dnn_explainer = create_dnn_explainer(
        dnn_model,
        dnn_shap_background
    )

except Exception:

    dnn_explainer = None


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
# FEATURE NAME FORMATTING
# ============================================================

def format_feature_name(feature):

    feature = feature.replace(
        "num__",
        ""
    )

    feature = feature.replace(
        "cat__",
        ""
    )

    replacements = {

        "avg_glucose_level":
            "Average glucose level",

        "age":
            "Age",

        "gender_Female":
            "Gender: Female",

        "gender_Male":
            "Gender: Male",

        "gender_Other":
            "Gender: Other",

        "hypertension_0":
            "Hypertension: No",

        "hypertension_1":
            "Hypertension: Yes",

        "heart_disease_0":
            "Heart disease: No",

        "heart_disease_1":
            "Heart disease: Yes",

        "ever_married_No":
            "Ever married: No",

        "ever_married_Yes":
            "Ever married: Yes",

        "Residence_type_Rural":
            "Residence: Rural",

        "Residence_type_Urban":
            "Residence: Urban",

        "smoking_status_formerly smoked":
            "Smoking: Formerly smoked",

        "smoking_status_never smoked":
            "Smoking: Never smoked",

        "smoking_status_smokes":
            "Smoking: Smokes",

        "smoking_status_Unknown":
            "Smoking: Unknown",

        "work_type_Govt_job":
            "Work type: Government",

        "work_type_Never_worked":
            "Work type: Never worked",

        "work_type_Private":
            "Work type: Private",

        "work_type_Self-employed":
            "Work type: Self-employed",

        "work_type_children":
            "Work type: Children",

        "bmi_group_Underweight":
            "BMI group: Underweight",

        "bmi_group_Normal":
            "BMI group: Normal",

        "bmi_group_Overweight":
            "BMI group: Overweight",

        "bmi_group_Obese":
            "BMI group: Obese"
    }

    return replacements.get(
        feature,
        feature
    )


# ============================================================
# XGBOOST LOCAL SHAP FUNCTION
# ============================================================

def generate_xgb_local_shap(
    X_processed,
    feature_names,
    top_n=10
):

    shap_explanation = (
        xgb_explainer(
            X_processed
        )
    )

    shap_values = (
        shap_explanation
        .values[0]
    )

    feature_values = (
        X_processed[0]
    )

    explanation_df = pd.DataFrame({

        "Feature":
            feature_names,

        "Processed_Value":
            feature_values,

        "SHAP_Value":
            shap_values,

        "Absolute_SHAP":
            np.abs(
                shap_values
            )

    })

    # Keep ALL features for later hybrid analysis
    explanation_df = (
        explanation_df
        .sort_values(
            "Absolute_SHAP",
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )

    explanation_df[
        "Display Feature"
    ] = (
        explanation_df[
            "Feature"
        ]
        .apply(
            format_feature_name
        )
    )

    top_explanation_df = (
        explanation_df
        .head(top_n)
        .copy()
    )

    return (
        explanation_df,
        top_explanation_df,
        shap_explanation
    )


# ============================================================
# DNN LOCAL SHAP FUNCTION
# ============================================================

def generate_dnn_local_shap(
    X_processed,
    feature_names,
    top_n=10
):

    if dnn_explainer is None:

        return None, None

    X_processed_float = np.asarray(
        X_processed
    ).astype(
        np.float32
    )

    shap_values = (
        dnn_explainer
        .shap_values(
            X_processed_float
        )
    )

    # --------------------------------------------------------
    # HANDLE SHAP OUTPUT FORMAT
    # --------------------------------------------------------

    if isinstance(
        shap_values,
        list
    ):

        shap_values = (
            shap_values[0]
        )

    shap_values = np.asarray(
        shap_values
    )

    # Some SHAP versions return:
    # (samples, features, outputs)

    if shap_values.ndim == 3:

        shap_values = (
            shap_values[:, :, 0]
        )

    # Single observation
    if shap_values.ndim == 2:

        shap_values = (
            shap_values[0]
        )

    elif shap_values.ndim == 1:

        shap_values = (
            shap_values
        )

    else:

        raise ValueError(
            "Unexpected DNN SHAP output shape: "
            f"{shap_values.shape}"
        )

    # --------------------------------------------------------
    # CREATE EXPLANATION TABLE
    # --------------------------------------------------------

    explanation_df = pd.DataFrame({

        "Feature":
            feature_names,

        "Processed_Value":
            X_processed_float[0],

        "SHAP_Value":
            shap_values,

        "Absolute_SHAP":
            np.abs(
                shap_values
            )

    })

    # Keep ALL features for hybrid analysis
    explanation_df = (
        explanation_df
        .sort_values(
            "Absolute_SHAP",
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )

    explanation_df[
        "Display Feature"
    ] = (
        explanation_df[
            "Feature"
        ]
        .apply(
            format_feature_name
        )
    )

    top_explanation_df = (
        explanation_df
        .head(top_n)
        .copy()
    )

    return (
        explanation_df,
        top_explanation_df
    )


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


    # ========================================================
    # EXPLAINABILITY SECTION
    # ========================================================

    st.divider()

    st.subheader(
        "Why did the model make this prediction?"
    )

    st.write(
        "The following explanations show which "
        "features had the greatest influence on "
        "the predictions of the two base learners "
        "used in the hybrid model."
    )

    st.caption(
        "Positive SHAP values indicate movement "
        "toward Stroke, while negative SHAP values "
        "indicate movement toward No Stroke."
    )


    # ========================================================
    # FEATURE NAMES
    # ========================================================

    feature_names = (
        preprocessor
        .get_feature_names_out()
    )


    if len(feature_names) != 26:

        st.warning(
            f"The preprocessor returned "
            f"{len(feature_names)} feature names "
            "instead of the expected 26."
        )


    # ========================================================
    # XGBOOST LOCAL EXPLANATION
    # ========================================================

    st.markdown(
        "### XGBoost Local Explanation"
    )

    st.write(
        "The XGBoost explanation identifies the "
        "features that most strongly influenced "
        "the XGBoost component for this patient."
    )


    try:

        (
            xgb_all_df,
            xgb_top_df,
            xgb_shap_explanation

        ) = generate_xgb_local_shap(

            X_processed,
            feature_names,

            top_n=10

        )


        # ----------------------------------------------------
        # XGBOOST GRAPH
        # ----------------------------------------------------

        xgb_plot_df = (

            xgb_top_df

            .sort_values(
                "SHAP_Value",
                ascending=True
            )

        )


        fig, ax = plt.subplots(
            figsize=(8, 5)
        )


        ax.barh(

            xgb_plot_df[
                "Display Feature"
            ],

            xgb_plot_df[
                "SHAP_Value"
            ]

        )


        ax.axvline(
            0,
            linewidth=1
        )


        ax.set_xlabel(
            "SHAP value"
        )

        ax.set_ylabel(
            "Feature"
        )

        ax.set_title(
            "Top Features Influencing XGBoost"
        )


        plt.tight_layout()

        st.pyplot(
            fig
        )

        plt.close(fig)


        # ----------------------------------------------------
        # XGBOOST TABLE
        # ----------------------------------------------------

        xgb_display_df = (
            xgb_top_df
            .copy()
        )


        xgb_display_df[
            "SHAP_Value"
        ] = (

            xgb_display_df[
                "SHAP_Value"
            ]

            .round(4)

        )


        st.dataframe(

            xgb_display_df[
                [
                    "Display Feature",
                    "SHAP_Value"
                ]
            ]

            .rename(
                columns={
                    "Display Feature":
                        "Feature",

                    "SHAP_Value":
                        "SHAP Impact"
                }
            ),

            use_container_width=True,

            hide_index=True

        )


    except Exception as e:

        st.warning(
            "XGBoost explainability could not "
            "be generated."
        )

        st.exception(e)

        xgb_all_df = None


    # ========================================================
    # DNN LOCAL EXPLANATION
    # ========================================================

    st.markdown(
        "### DNN Local Explanation"
    )

    st.write(
        "The DNN explanation identifies the "
        "features that most strongly influenced "
        "the DNN component for this patient."
    )


    dnn_all_df = None
    dnn_top_df = None


    if dnn_explainer is None:

        st.warning(
            "DNN SHAP explainability is unavailable. "
            "Make sure dnn_shap_background.npy is "
            "present in the models folder."
        )

    else:

        try:

            (
                dnn_all_df,
                dnn_top_df

            ) = generate_dnn_local_shap(

                X_processed,
                feature_names,

                top_n=10

            )


            # ------------------------------------------------
            # DNN GRAPH
            # ------------------------------------------------

            dnn_plot_df = (

                dnn_top_df

                .sort_values(
                    "SHAP_Value",
                    ascending=True
                )

            )


            fig, ax = plt.subplots(
                figsize=(8, 5)
            )


            ax.barh(

                dnn_plot_df[
                    "Display Feature"
                ],

                dnn_plot_df[
                    "SHAP_Value"
                ]

            )


            ax.axvline(
                0,
                linewidth=1
            )


            ax.set_xlabel(
                "SHAP value"
            )

            ax.set_ylabel(
                "Feature"
            )

            ax.set_title(
                "Top Features Influencing DNN"
            )


            plt.tight_layout()

            st.pyplot(
                fig
            )

            plt.close(fig)


            # ------------------------------------------------
            # DNN TABLE
            # ------------------------------------------------

            dnn_display_df = (
                dnn_top_df
                .copy()
            )


            dnn_display_df[
                "SHAP_Value"
            ] = (

                dnn_display_df[
                    "SHAP_Value"
                ]

                .round(4)

            )


            st.dataframe(

                dnn_display_df[
                    [
                        "Display Feature",
                        "SHAP_Value"
                    ]
                ]

                .rename(
                    columns={
                        "Display Feature":
                            "Feature",

                        "SHAP_Value":
                            "SHAP Impact"
                    }
                ),

                use_container_width=True,

                hide_index=True

            )


        except Exception as e:

            st.warning(
                "DNN explainability could not "
                "be generated for this prediction."
            )

            st.exception(e)


    # ========================================================
    # HYBRID EXPLANATION
    # ========================================================

    st.divider()

    st.subheader(
        "Hybrid Explanation"
    )

    st.write(
        "The hybrid explanation summarizes feature "
        "importance across the XGBoost and DNN base "
        "learners. Features that are influential in "
        "both models provide evidence that multiple "
        "components of the hybrid system are responding "
        "to the same patient characteristics."
    )


    if (

        xgb_all_df is not None

        and dnn_all_df is not None

    ):

        try:

            # ------------------------------------------------
            # MERGE ALL 26 FEATURES
            # ------------------------------------------------

            hybrid_shap_df = pd.merge(

                xgb_all_df[
                    [
                        "Feature",
                        "Absolute_SHAP"
                    ]
                ],

                dnn_all_df[
                    [
                        "Feature",
                        "Absolute_SHAP"
                    ]
                ],

                on="Feature",

                how="outer",

                suffixes=(
                    "_XGBoost",
                    "_DNN"
                )

            )


            hybrid_shap_df = (
                hybrid_shap_df
                .fillna(0)
            )


            # ------------------------------------------------
            # NORMALIZE MODEL IMPORTANCE
            # ------------------------------------------------

            xgb_total = (
                hybrid_shap_df[
                    "Absolute_SHAP_XGBoost"
                ]
                .sum()
            )


            dnn_total = (
                hybrid_shap_df[
                    "Absolute_SHAP_DNN"
                ]
                .sum()
            )


            if xgb_total > 0:

                hybrid_shap_df[
                    "XGB_Normalized"
                ] = (

                    hybrid_shap_df[
                        "Absolute_SHAP_XGBoost"
                    ]

                    / xgb_total

                )

            else:

                hybrid_shap_df[
                    "XGB_Normalized"
                ] = 0


            if dnn_total > 0:

                hybrid_shap_df[
                    "DNN_Normalized"
                ] = (

                    hybrid_shap_df[
                        "Absolute_SHAP_DNN"
                    ]

                    / dnn_total

                )

            else:

                hybrid_shap_df[
                    "DNN_Normalized"
                ] = 0


            # ------------------------------------------------
            # COMBINED IMPORTANCE
            # ------------------------------------------------

            hybrid_shap_df[
                "Combined Importance"
            ] = (

                hybrid_shap_df[
                    "XGB_Normalized"
                ]

                +

                hybrid_shap_df[
                    "DNN_Normalized"
                ]

            ) / 2


            # ------------------------------------------------
            # SORT
            # ------------------------------------------------

            hybrid_shap_df = (

                hybrid_shap_df

                .sort_values(
                    "Combined Importance",
                    ascending=False
                )

                .reset_index(
                    drop=True
                )

            )


            hybrid_shap_df[
                "Display Feature"
            ] = (

                hybrid_shap_df[
                    "Feature"
                ]

                .apply(
                    format_feature_name
                )

            )


            # ------------------------------------------------
            # TOP 10 HYBRID FEATURES
            # ------------------------------------------------

            hybrid_top_df = (

                hybrid_shap_df

                .head(10)

                .copy()

            )


            # ------------------------------------------------
            # HYBRID GRAPH
            # ------------------------------------------------

            hybrid_plot_df = (

                hybrid_top_df

                .sort_values(
                    "Combined Importance",
                    ascending=True
                )

            )


            fig, ax = plt.subplots(
                figsize=(8, 5)
            )


            ax.barh(

                hybrid_plot_df[
                    "Display Feature"
                ],

                hybrid_plot_df[
                    "Combined Importance"
                ]

            )


            ax.set_xlabel(
                "Combined normalized importance"
            )

            ax.set_ylabel(
                "Feature"
            )

            ax.set_title(
                "Shared Feature Importance Across Base Learners"
            )


            plt.tight_layout()

            st.pyplot(
                fig
            )

            plt.close(fig)


            # ------------------------------------------------
            # HYBRID TABLE
            # ------------------------------------------------

            hybrid_display_df = (

                hybrid_top_df

                .copy()

            )


            hybrid_display_df[
                "XGBoost Importance"
            ] = (

                hybrid_display_df[
                    "XGB_Normalized"
                ]

                .round(4)

            )


            hybrid_display_df[
                "DNN Importance"
            ] = (

                hybrid_display_df[
                    "DNN_Normalized"
                ]

                .round(4)

            )


            hybrid_display_df[
                "Combined Importance"
            ] = (

                hybrid_display_df[
                    "Combined Importance"
                ]

                .round(4)

            )


            st.dataframe(

                hybrid_display_df[
                    [
                        "Display Feature",
                        "XGBoost Importance",
                        "DNN Importance",
                        "Combined Importance"
                    ]
                ]

                .rename(
                    columns={
                        "Display Feature":
                            "Feature"
                    }
                ),

                use_container_width=True,

                hide_index=True

            )


            # ------------------------------------------------
            # HYBRID INTERPRETATION
            # ------------------------------------------------

            st.info(
                """
                The shared feature importance summarizes
                the relative influence of features across
                the XGBoost and DNN base learners. Features
                with high combined importance were influential
                across both components of the hybrid model.

                These explanations describe model behaviour
                and should not be interpreted as evidence of
                causation or as a clinical diagnosis.
                """
            )


        except Exception as e:

            st.warning(
                "The combined hybrid explanation "
                "could not be generated."
            )

            st.exception(e)


    else:

        st.info(
            "The hybrid feature explanation requires "
            "both XGBoost and DNN SHAP explanations."
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

        st.write(
            "The feature explanations describe the "
            "behaviour of the individual base learners "
            "and the shared feature importance across "
            "those learners."
        )
