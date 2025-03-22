import pandas as pd
import streamlit as st

import pickle
st.title("Chronic Kidney Disease Risk Factor Prediction")

st.sidebar.title("Queries or Issues")
st.sidebar.info(
    """
    For any issues with app usage, please contact: n1009755@my.ntu.ac.uk
    """
)

a, b, c = st.sidebar.columns([0.2, 1, 0.2])
with b:
    st.markdown(
        """
        <div align=center>
        <small>
        Helpful links:
        https://www.kidney.org/kidney-topics/chronic-kidney-disease-ckd
        </a>
        </small>
        </div>
        """,
        unsafe_allow_html=True,
    )

i, a, b, c, d = st.columns([0.2, 20, 0.01, 8, 0.5])
with a:
    with st.expander("ℹ️ General Instructions", expanded=False):
        st.markdown(
            """
            ### KD5 Predictor
            This web app is designed to predict the risk of Chronic Kidney Disease (CKD) based on various risk factors. This app uses a Logistic Regression (LR) algorithm to help predict CKD this algorithm is trained from the Chronic Kidney Disease dataset from the UCI Machine Learning Repository. Using the avalible training data, the model has a testing accuracy of 95%. Please consult a medical professional for a formal diagnosis.
            """
        )
    
        
        st.markdown("Patient Records can be uploaded by manual input. Once data has been inputed, the patient data will be displayed in comparison to the machine learning dataset. A prediction will be made based on the patient data using a Logisitc Regression Algorithm. The patient's Estimated Glomerular Filtration Rate (eGFR) will also be calculated using the 2021 CKD-EPI formula. Following the eGFR calculation, the patient's data will be displayed in comparison to the dataset in the Output page.")
    st.markdown(
        '<b style="font-family:serif; color:#6082B6; font-size: 28px;"></b>',
        unsafe_allow_html=True,
    )

with c:
    st.markdown(
        """
            ### Disclaimer
            Please read the general instructions before proceeding. 
   """
        )

def loaded_Prediction():
    with open('model.pkl', 'rb') as file:
        prediction = pickle.load(file)
    return prediction

prediction = loaded_Prediction()
important_features = prediction.feature_names_in_


# Input form
with st.form("my_form"):
    st.write("Manually input patient data:")
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Enter the patient's age:", min_value=0, max_value=120, value=30)
        bp = st.number_input("Enter the patient's blood pressure:", min_value=0, max_value=200, value=120)
        sg = st.number_input("Enter the patient's specific gravity:", min_value=1.0, max_value=1.1, value=1.02)
        al = st.number_input("Enter the patient's albumin:", min_value=0.0, max_value=5.0, value=0.0)
        su = st.number_input("Enter the patient's sugar:", min_value=0.0, max_value=20.0, value=0.0)
        bgr = st.number_input("Enter the patient's blood glucose random:", min_value=0.00, max_value=500.00, value=100.00)
        bu = st.number_input("Enter the patient's blood urea:", min_value=0.00, max_value=200.00, value=40.00)
        sc = st.number_input("Enter the patient's serum creatinine:", min_value=0.00, max_value=25.00, value=1.2)
    with col2:
        sod = st.number_input("Enter the patient's sodium:", min_value=0.00, max_value=200.00, value=140.00)
        pot = st.number_input("Enter the patient's potassium:", min_value=0.00, max_value=15.00, value=4.5)
        hemo = st.number_input("Enter the patient's hemoglobin:", min_value=0.00, max_value=20.00, value=15.00)
        pcv = st.number_input("Enter the patient's packed cell volume:", min_value=0.00, max_value=100.00, value=45.00)
        wc = st.number_input("Enter the patient's white blood cell count:", min_value=0.00, max_value=20000.00, value=8000.00)
        rc = st.number_input("Enter the patient's red blood cell count:", min_value=0.00, max_value=10.00, value=5.00)
        htn = st.selectbox("Does the patient have hypertension:", ("yes", "no"))
        dm = st.selectbox("Does the patient have diabetes mellitus:", ("yes", "no"))
    with col3:
        cad = st.selectbox("Does the patient have coronary artery disease:", ("yes", "no"))
        appet = st.selectbox("How is the patient's appetite:", ("good", "poor"))
        pe = st.selectbox("Does the patient have pedal edema:", ("yes", "no"))
        ane = st.selectbox("Does the patient have anemia:", ("yes", "no"))
        rbc = st.selectbox("How are the patient's red blood cells:", ("normal", "abnormal"))
        pc = st.selectbox("How are the patient's pus cells:", ("normal", "abnormal"))
        pcc = st.selectbox("Does the patient have pus cell clumps:", ("present", "not present"))
        ba = st.selectbox("Does the patient have bacteria:", ("present", "not present"))    

    checkbox_validation = st.checkbox("Confirm all patient data has been inputted correctly")

    submitted = st.form_submit_button("Submit")

if submitted:
    new_data = {
        'Age': age,
        'Blood Pressure': bp,
        'Specific Gravity': sg,
        'Albumin': al,
        'Sugar': su,
        'Blood Glucose Random': bgr,
        'Blood Urea': bu,
        'Serum Creatinine': sc,
        'Sodium': sod,
        'Potassium': pot,
        'Hemoglobin': hemo,
        'Packed Cell Volume': pcv,
        'White Blood Cell Count': wc,
        'Red Blood Cell Count': rc,
        'Hypertension': 1 if htn == 'yes' else 0,
        'Diabetes Mellitus': 1 if dm == 'yes' else 0,
        'Coranary Artery Disease': 1 if cad == 'yes' else 0,
        'Appetite': 1 if appet == 'good' else 0,
        'Pedal Edema': 1 if pe == 'yes' else 0,
        'Anemia': 1 if ane == 'yes' else 0,
        'Red Blood Cells': 1 if rbc == 'normal' else 0,
        'Pus Cells': 1 if pc == 'normal' else 0,
        'Pus Cell Clumps': 1 if pcc == 'present' else 0,
        'Bacteria': 1 if ba == 'present' else 0
    }

    new_df = pd.DataFrame([new_data])

    st.session_state['new_df'] = new_df.copy()
    # Extract important features for prediction
    important_features = ['sc', 'al', 'sg', 'hemo', 'htn', 'dm']
    column_mapping = {
    'Serum Creatinine': 'sc',
    'Albumin': 'al',
    'Specific Gravity': 'sg',
    'Hemoglobin': 'hemo',
    'Hypertension': 'htn',
    'Diabetes Mellitus': 'dm'
}
    print("new_df columns:", new_df.columns)

    new_df.rename(columns=column_mapping, inplace=True)
    new_df_prediction = new_df[important_features].reindex(columns=prediction.feature_names_in_)
# Extract the important features for prediction
    new_df_prediction = new_df[important_features]
    new_df_prediction = new_df.rename(columns=column_mapping)[important_features]

    # Align with model's trained feature order
    new_df_prediction = new_df_prediction.reindex(columns=prediction.feature_names_in_)

    # Prediction step
    model_prediction = prediction.predict(new_df_prediction)
    result = "Positive for CKD" if model_prediction[0] == 0 else "Negative for CKD"
    if 'patient_data' not in st.session_state:
        st.session_state['patient_data'] = []
    st.session_state['patient_data'].append({
    'data': new_df,
    'result': result
})
    # Display Results
    st.success(f"Patient data submitted! The Data Suggests the Patient is **{result}**.")




#EGFR calculation using 2021 ckd-epi formula
with st.form("EGFR Calculation"):
        st.write("Estimated Glomerular Filtration Rate (eGFR) Calculation:")
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Enter the patient's age:", min_value=0, max_value=120, value=30)
            gender = st.selectbox("Select the patient's gender:", ("Male","Female"))
            sc = st.number_input("Enter the patient's serum creatinine:", min_value=0.00, max_value=15.00, value=1.2)
            K = 0.7 if gender == "Female" else 0.9
        alpha = -0.329 if gender == "Female" else -0.411
        gender_multiplier = 1.012 if gender == "Female" else 1.0

        standardized_scr = sc / K
        eGFR = (142 *(min(standardized_scr, 1) ** alpha) *(max(standardized_scr, 1) ** -1.200) *(0.9938 ** age) *gender_multiplier)

        submitted = st.form_submit_button("Submit")
        if submitted:
            st.write(f"Estimated Glomerular Filtration Rate (eGFR): {eGFR:.2f} mL/min/1.73m²")
            if eGFR >= 90:
                st.write("Stage 1")
            elif 60 <= eGFR <= 89:
                st.write("Stage 2")
            elif 45 <= eGFR <= 59:
                st.write("Stage 3a")
            elif 30 <= eGFR <= 44:
                st.write("Stage 3b")
            elif 15 <= eGFR <= 29:
                st.write("Stage 4")
            elif eGFR < 15:
                st.write("Stage 5")
