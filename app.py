import streamlit as st
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
df = pd.read_csv('HR_Data_Predict Employee Turnover.csv')

# Loading the RandomForest model
model = joblib.load('employee_turnover_model.pkl')

scaler = joblib.load('scaler.pkl')
st.set_page_config(page_title='HR Turnover Predictor', layout='centered')

# Sidebar Inputs
st.sidebar.header(" Employee Info")
satisfaction = st.sidebar.slider("Satisfaction Level", 0.0, 1.0, 0.5)
evaluation = st.sidebar.slider("Last Evaluation", 0.0, 1.0, 0.5)
projects = st.sidebar.slider("Number of Projects", 1, 10, 3)
hours = st.sidebar.slider("Average Monthly Hours", 80, 320, 160)
time_spent = st.sidebar.slider("Years at Company", 1, 10, 3)
accident = st.sidebar.selectbox("Had Work Accident?", ['No', 'Yes'])
promotion = st.sidebar.selectbox("Promoted in Last 5 Years?", ['No', 'Yes'])
salary = st.sidebar.selectbox("Salary Level", ['low', 'medium', 'high'])
department = st.sidebar.selectbox("Department", [
    'RandD', 'accounting', 'hr', 'management', 'marketing',
    'product_mng', 'sales', 'technical'])

# Create input vector matching model features
input_data = np.array([[
    satisfaction,
    evaluation,
    projects,
    hours,
    time_spent,
    1 if accident == 'Yes' else 0,
    1 if promotion == 'Yes' else 0,
    # Department one-hot
    1 if department == 'RandD' else 0,
    1 if department == 'accounting' else 0,
    1 if department == 'hr' else 0,
    1 if department == 'management' else 0,
    1 if department == 'marketing' else 0,
    1 if department == 'product_mng' else 0,
    1 if department == 'sales' else 0,
    1 if department == 'technical' else 0,
    # Salary one-hot
    1 if salary == 'high' else 0,
    1 if salary == 'low' else 0,
    1 if salary == 'medium' else 0
]])

input_scaled = scaler.transform(input_data)

st.markdown(
   """<h2 style='text-align:center; color:#4B8BBE;'>Employee Turnover Prediction</h2>
   """, unsafe_allow_html=True)

st.markdown(
    """
    <div style='text-align: center;'>
        <img src='https://factohr-1a56a.kxcdn.com/wp-content/uploads/2021/03/etrrgeqtj.png' 
             width='800' height='600' style='border-radius: 15px;' />
    </div>
    """,
    unsafe_allow_html=True
)



# Prediction
col1, col2, col3 = st.columns([1, 2, 1])  # 3 columns, center column is wider

with col2:
  if st.button(" Predict Turnover Risk"):
    prediction = model.predict(input_scaled)[0]
    if prediction == 1:
        st.error(" This employee is at HIGH risk of leaving the company.")
    else:
        st.success(" This employee is likely to STAY at the company.")
proba = model.predict_proba(input_scaled)[0][1]  # Probability of leaving

# Center using columns
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.metric(label=" Turnover Risk Probability", value=f"{proba:.2%}")





