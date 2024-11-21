# app.py
import sklearn
import streamlit as st
import pandas as pd
import numpy as np
import numpy
import pickle
import os
import joblib
from joblib import load

# st.write("scikit-learn version:", sklearn.__version__)
# st.write("numpy version:", numpy.__version__)
# st.write("joblib version:", joblib.__version__)


# Construir la ruta absoluta del modelo
current_dir = os.path.dirname(os.path.abspath(__file__))
st.write("Archivos en el directorio actual:", os.listdir(current_dir))

# Intentar cargar el modelo
try:
    # Cargar el modelo con joblib
    model_path = os.path.join(current_dir, 'best_model_app.joblib')
    model = load(model_path)
    # st.write("Modelo cargado exitosamente.")
except FileNotFoundError:
    st.error(f"Error: No se encontró el archivo en la ruta: {model_path}")
    raise
except Exception as e:
    st.error(f"Error inesperado al cargar el modelo: {e}")
    raise

# # Load the trained model
# with open('best_model_app.pkl', 'rb') as f:
#     model = pickle.load(f)


# Streamlit app introduction with professional tone and disclaimers
st.title('Depression Chronicity Prediction')

# Hook line
st.write("**Empower your clinical assessments with data-driven insights.**")

# Detailed description with disclaimers
st.write("""
This tool is designed as an advanced aid to help medical professionals analyze the risk of depression chronicity in patients. While grounded in evidence-based algorithms and enriched by comprehensive data analysis, this resource is intended to supplement—not replace—clinical judgment. It serves as a guide to highlight potential red flags and deepen understanding, empowering practitioners to make well-rounded decisions with greater confidence.
""")

# Disclaimers with emojis
st.write("⚠️ **Disclaimer**: This is not a 100% precise predictive algorithm.")
st.write("🔄 **Note**: Depression and its progression to chronicity are highly variable and can be influenced by numerous complex factors.")
st.write("🧑‍⚕️ **Reminder**: This prediction should be considered as one of many factors to inform professional judgment, not as a substitute for it.")

# Invitation to use the tool
st.write("Explore its potential and see how it can enhance your diagnostic process today.")

# Create input forms for each variable
age = st.number_input('Enter your age', min_value=0, max_value=120, step=1)
smoking_status = st.checkbox('Are you a smoker?')
employment_status = st.checkbox('Are you currently employed?')
income = st.number_input('Enter your annual income', min_value=0.0, step=500.0)

# Dropdowns for categorical variables with options
marital_status = st.selectbox('Select your marital status', ['Single', 'Married', 'Divorced', 'Widowed'])
education_level = st.selectbox('Select your education level', [
    'High School', 'Associate Degree', "Bachelor's Degree", "Master's Degree", 'PhD'
])
physical_activity_level = st.selectbox('Select your physical activity level', ['Sedentary', 'Moderate', 'Active'])
alcohol_consumption_level = st.selectbox('Select your alcohol consumption level', ['Low', 'Moderate', 'High'])
diet_level = st.selectbox('Select your dietary habits', ['Unhealthy', 'Moderate', 'Healthy'])
sleep_level = st.selectbox('Select your sleep patterns', ['Poor', 'Fair', 'Good'])

# Checkboxes for historical conditions
mental_illness_history = st.checkbox('History of Mental Illness')
substance_abuse_history = st.checkbox('History of Substance Abuse')
family_depression_history = st.checkbox('Family History of Depression')

# Feature engineering for age groups (creating dummies)
age_group = {
    'Age_Childhood': 1 if 0 < age < 12 else 0,
    'Age_Adolescence': 1 if 12 <= age < 18 else 0,
    'Age_Young Adulthood': 1 if 18 <= age < 40 else 0,
    'Age_Adulthood': 1 if 40 <= age < 60 else 0,
    'Age_Eld': 1 if age >= 60 else 0
}

# Create a DataFrame that matches the model's expected format
input_data = pd.DataFrame({
    'Log_Income': [np.log1p(income)],
    'Number of Children': [0],  # If this column is no longer used, update accordingly.
    'Smoking Status': [1 if smoking_status else 0],
    'Employment Status': [1 if employment_status else 0],
    'Physical Activity Level': [0 if physical_activity_level == 'Sedentary' else 5 if physical_activity_level == 'Moderate' else 10],
    'Alcohol Consumption': [0 if alcohol_consumption_level == 'Low' else 5 if alcohol_consumption_level == 'Moderate' else 10],
    'Dietary Habits': [10 if diet_level == 'Unhealthy' else 5 if diet_level == 'Moderate' else 0],
    'Sleep Patterns': [10 if sleep_level == 'Poor' else 5 if sleep_level == 'Fair' else 0],
    'History of Mental Illness': [1 if mental_illness_history else 0],
    'History of Substance Abuse': [1 if substance_abuse_history else 0],
    'Family History of Depression': [1 if family_depression_history else 0],
    'Status_Single': [1 if marital_status == 'Single' else 0],
    'Status_Married': [1 if marital_status == 'Married' else 0],
    'Status_Divorced': [1 if marital_status == 'Divorced' else 0],
    'Status_Widowed': [1 if marital_status == 'Widowed' else 0],
    'Education_High School': [1 if education_level == 'High School' else 0],
    'Education_Associate Degree': [1 if education_level == 'Associate Degree' else 0],
    "Education_Bachelor's Degree": [1 if education_level == "Bachelor's Degree" else 0],
    "Education_Master's Degree": [1 if education_level == "Master's Degree" else 0],
    'Education_PhD': [1 if education_level == 'PhD' else 0],
    **age_group  # Add age group dummies
})

# Align the input DataFrame to match the training columns
input_data = input_data.reindex(columns=model.feature_names_in_, fill_value=0)

# Button to make prediction
if st.button('Predict'):
    # Make prediction
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0][1]  # Get chronicity probability

    # Custom result text based on the probability
    if probability > 0.5:
        result_text = "The patient has a high probability (>50%) of developing chronic depression. Please consider this case with extra care."
    else:
        result_text = "The patient has a low probability (<50%) of developing chronic depression. This does not rule out the possibility, so further assessment is advised."

    # Display the result
    st.write('Chronicity Prediction:', result_text)
    st.write("**Note**: This prediction is an additional tool and should not replace clinical judgment.")
