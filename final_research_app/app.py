# STROKE PREDICTION WEB APP
# XGBoost + DNN Stacking Model

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import shap
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model


# configure page
st.set_page_config(
    page_title="Stroke Prediction System",
    page_icon="🩺",
    layout="centered"
)

# define model paths
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


# load models
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


# load deployment models
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


# create SHAP explainers
def create_xgb_explainer(
    model
):

    return shap.TreeExplainer(
        model
    )


xgb_explainer = create_xgb_explainer(
    xgb_model
)

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


def create_bmi_group(bmi):

    if bmi < 18.5:

        return "Underweight"

    elif bmi < 25:

        return "Normal"

    elif bmi < 30:

        return "Overweight"

    else:

        return "Obese"


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


# XGBoost local SHAP explainer
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
        top_explanation_df,
        shap_explanation
    )


# DNN local SHAP explainer
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


    # create explanation table
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


# create app header
st.title(
    "Stroke Prediction System"
)

st.write(
    "Enter the patient's information below to "
    "generate a stroke-risk prediction using "
    "the proposed XGBoost–DNN stacking model."
)

st.divider()


# create input areas

st.subheader(
    "Patient Information"
)


age = st.number_input(

    "Age",

    min_value=0.0,

    max_value=120.0,

    value=50.0,

    step=1.0

)

gender = st.selectbox(

    "Gender",

    options=[
        "Male",
        "Female",
        "Other"
    ]

)

hypertension = st.selectbox(

    "Hypertension",

    options=[
        0,
        1
    ],

    format_func=lambda x:
        "No" if x == 0 else "Yes"

)

heart_disease = st.selectbox(

    "Heart Disease",

    options=[
        0,
        1
    ],

    format_func=lambda x:
        "No" if x == 0 else "Yes"

)

ever_married = st.selectbox(

    "Ever Married",

    options=[
        "Yes",
        "No"
    ]

)

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

residence_type = st.selectbox(

    "Residence Type",

    options=[
        "Urban",
        "Rural"
    ]

)

smoking_status = st.selectbox(

    "Smoking Status",

    options=[
        "formerly smoked",
        "never smoked",
        "smokes",
        "Unknown"
    ]

)

avg_glucose_level = st.number_input(

    "Average Glucose Level",

    min_value=0.0,

    max_value=500.0,

    value=100.0,

    step=0.1

)

bmi = st.number_input(

    "BMI",

    min_value=0.0,

    max_value=100.0,

    value=25.0,

    step=0.1

)

st.divider()

predict_button = st.button(

    "Predict Stroke Risk",

    type="primary",

    use_container_width=True

)


# prediction

if predict_button:

    bmi_group = create_bmi_group(
        bmi
    )

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


    # preprocess input

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


    # verify feature count

    if X_processed.shape[1] != 26:

        st.error(

            "Unexpected number of processed features: "
            f"{X_processed.shape[1]}. "
            "Expected 26."

        )

        st.stop()


    # XGBoost Prediction

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


    # DNN Prediction

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


    # Stacking input

    stacking_input = pd.DataFrame({

        "XGB_Probability": [
            xgb_probability
        ],

        "DNN_Probability": [
            dnn_probability
        ]

    })


     # Stacking prediction

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


    # Final classification

    prediction = int(

        hybrid_probability
        >= threshold

    )


    # Display the results

    st.divider()

    st.subheader(
        "Prediction Result"
    )


    st.info(
        f"Derived BMI Group: **{bmi_group}**"
    )


    # Display the final results

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


    # Hybrid probability

    st.metric(

        "Hybrid Stroke Probability",

        f"{hybrid_probability * 100:.2f}%"

    )


    st.write(
        f"Decision threshold: "
        f"**{threshold * 100:.0f}%**"
    )


    # Base model probabilities
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


    ##### Explainability Section #####

    st.divider()

    st.subheader(
        "Why did the model make this prediction?"
    )

    st.write(
        "These explanations show which "
        "features had the greatest influence on "
        "the predictions of the two base learners "
        "used in the hybrid model."
    )

    st.caption(
        "Positive SHAP values indicate movement "
        "toward Stroke, while negative SHAP values "
        "indicate movement toward No Stroke."
    )


    # Feature names

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


    # XGBoost local explanation

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


    # DNN local explanation
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


            # DNN Graph
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


            # DNN Table
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


    # Hybrid Explanation

    st.divider()

    st.subheader(
        "Hybrid Explanation"
    )


    if (

        xgb_all_df is not None

        and dnn_all_df is not None

    ):

        try:

            # merge all features
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

            # Normalize
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


            # Combined importance

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


            # List down top 10 features

            hybrid_top_df = (

                hybrid_shap_df

                .head(10)

                .copy()

            )


            # Hybrid Graph

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


            # Hybrid Table

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


            # Hybrid interpretation

            st.info(
                """
                The shared feature importance summarizes
                the relative influence of features across
                the XGBoost and DNN models. Features
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
