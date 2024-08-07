import streamlit as st
import joblib
import json
import numpy as np
from azureml.core.model import Model

# Initialize the model
def init():
    global model_3
    model_3 = joblib.load('rf_model_500.pkl')  # Load from the local file

# Function to make predictions
def run(raw_data):
    try:
        data = json.loads(raw_data)['data']
        data = np.array(data)
        result_1 = model_3.predict(data)
        return {"prediction1": result_1.tolist()}
    except Exception as e:
        result = str(e)
        return result

# Streamlit interface
def main():
    st.title("Diabetes Prediction Model")

    # User inputs
    pregnancies = st.number_input('Pregnancies', min_value=0, max_value=20, step=1)
    glucose = st.number_input('Glucose', min_value=0, max_value=200)
    skin_thickness = st.number_input('Skin Thickness', min_value=0, max_value=100)
    bmi = st.number_input('BMI', min_value=0.0, max_value=70.0, step=0.1)
    age = st.number_input('Age', min_value=0, max_value=120, step=1)

    # Prepare input data for prediction
    input_data = [[pregnancies, glucose, skin_thickness, bmi, age]]

    if st.button('Predict'):
        # Format the input data as expected by the model
        raw_data = json.dumps({"data": input_data})
        result = run(raw_data)
        st.write('Prediction:', result["prediction1"])

if __name__ == '__main__':
    init()
    main()
