import gradio as gr
import pickle
import pandas as pd
import numpy as np
import os

# Get current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'loan_model.pkl')

print(f"Loading model from: {model_path}")

# Load model data
model_data = pickle.load(open(model_path, 'rb'))
model = model_data['model']
scaler = model_data['scaler']
feature_names = model_data['feature_names']

print(f"✅ Loaded model with features: {feature_names}")

def predict_loan(age, gender, education, income, emp_exp, home_ownership, 
                 loan_amount, loan_intent, interest_rate, loan_percent_income,
                 credit_history, credit_score, previous_defaults):
    
    # Encode gender
    gender_map = {"female": 0, "male": 1}
    gender_encoded = gender_map[gender]
    
    # Encode education
    education_map = {"High School": 0, "Associate": 1, "Bachelor": 2, "Master": 3}
    education_encoded = education_map[education]
    
    # Encode home ownership
    home_map = {"RENT": 0, "OWN": 1, "MORTGAGE": 2, "OTHER": 3}
    home_encoded = home_map[home_ownership]
    
    # Encode previous defaults
    defaults_map = {"No": 0, "Yes": 1}
    defaults_encoded = defaults_map[previous_defaults]
    
    # Encode loan intent (OneHot) - remember drop_first=True was used
    intent_education = 1 if loan_intent == "EDUCATION" else 0
    intent_homeimprovement = 1 if loan_intent == "HOMEIMPROVEMENT" else 0
    intent_medical = 1 if loan_intent == "MEDICAL" else 0
    intent_personal = 1 if loan_intent == "PERSONAL" else 0
    intent_venture = 1 if loan_intent == "VENTURE" else 0
    
    # Create input with all features in correct order
    input_dict = {
        'person_age': age,
        'person_gender': gender_encoded,
        'person_education': education_encoded,
        'person_income': income,
        'person_emp_exp': emp_exp,
        'person_home_ownership': home_encoded,
        'loan_amnt': loan_amount,
        'loan_int_rate': interest_rate,
        'loan_percent_income': loan_percent_income,
        'cb_person_cred_hist_length': credit_history,
        'credit_score': credit_score,
        'previous_loan_defaults_on_file': defaults_encoded,
        'loan_intent_EDUCATION': intent_education,
        'loan_intent_HOMEIMPROVEMENT': intent_homeimprovement,
        'loan_intent_MEDICAL': intent_medical,
        'loan_intent_PERSONAL': intent_personal,
        'loan_intent_VENTURE': intent_venture
    }
    
    # Create DataFrame with exact feature names
    input_data = pd.DataFrame([input_dict])[feature_names]
    
    try:
        # Scale input and convert back to DataFrame
        input_scaled = scaler.transform(input_data)
        input_scaled_df = pd.DataFrame(input_scaled, columns=feature_names)
        
        # Predict
        prediction = model.predict(input_scaled_df)
        probability = model.predict_proba(input_scaled_df)
        
        if prediction[0] == 0:
            result = "✅ Loan Approved"
            confidence = probability[0][0] * 100
        else:
            result = "❌ Loan Rejected (High Default Risk)"
            confidence = probability[0][1] * 100
        
        return f"{result}\n\nConfidence: {confidence:.2f}%"
    
    except Exception as e:
        return f"❌ Error: {str(e)}"

# Create Gradio interface
demo = gr.Interface(
    fn=predict_loan,
    inputs=[
        gr.Number(label="Age", value=30, minimum=18, maximum=100),
        gr.Dropdown(choices=["male", "female"], label="Gender", value="male"),
        gr.Dropdown(choices=["High School", "Associate", "Bachelor", "Master"], 
                   label="Education", value="Bachelor"),
        gr.Number(label="Annual Income ($)", value=50000, minimum=0),
        gr.Number(label="Employment Experience (years)", value=5, minimum=0, maximum=50),
        gr.Dropdown(choices=["RENT", "OWN", "MORTGAGE", "OTHER"], 
                   label="Home Ownership", value="RENT"),
        gr.Number(label="Loan Amount ($)", value=10000, minimum=500, maximum=50000),
        gr.Dropdown(choices=["EDUCATION", "HOMEIMPROVEMENT", "MEDICAL", "PERSONAL", "VENTURE"], 
                   label="Loan Intent", value="PERSONAL"),
        gr.Number(label="Interest Rate (%)", value=10.0, minimum=5, maximum=25),
        gr.Number(label="Loan as % of Income", value=0.3, minimum=0, maximum=1, step=0.01),
        gr.Number(label="Credit History Length (years)", value=5, minimum=0, maximum=30),
        gr.Number(label="Credit Score", value=650, minimum=300, maximum=850),
        gr.Dropdown(choices=["Yes", "No"], label="Previous Loan Defaults", value="No")
    ],
    outputs=gr.Textbox(label="Prediction Result"),
    title="🏦 Loan Default Prediction System",
    description="Enter loan applicant information to predict approval status and default risk",
    examples=[
        [30, "male", "Bachelor", 50000, 5, "RENT", 10000, "PERSONAL", 10.0, 0.3, 5, 650, "No"],
        [45, "female", "Master", 80000, 15, "OWN", 25000, "EDUCATION", 8.5, 0.25, 10, 720, "No"],
        [25, "male", "High School", 25000, 2, "RENT", 15000, "VENTURE", 18.0, 0.6, 2, 550, "Yes"],
    ]
)

if __name__ == "__main__":
    demo.launch()