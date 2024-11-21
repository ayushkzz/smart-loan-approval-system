from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load pre-trained model and scaler
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Mappings for categorical variables
gender_mapping = {'male': 0, 'female': 1}
education_mapping = {'high school': 0, "bachelor's": 1, "master's": 2, 'doctorate': 3}
region_mapping = {'rural': 0, 'urban': 1}
business_type_mapping = {'service': 0, 'retail': 1, 'wholesale': 2}
loan_purpose_mapping = {'working capital': 0, 'expansion': 1, 'purchase': 2, 'other': 3}

@app.route('/')
def home():
    return render_template('form.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect and preprocess input data
        form_data = request.form
        
        # Assuming all the input fields are correctly mapped and match the feature set
        features = [
            int(form_data['age']),
            gender_mapping[form_data['gender']],
            education_mapping[form_data['education']],
            region_mapping[form_data['region']],
            business_type_mapping[form_data['business_type']],
            float(form_data['monthly_income']),
            float(form_data['monthly_expenditure']),
            int(form_data['transaction_frequency']),
            int(form_data['network_size']),
            float(form_data['social_credibility_score']),
            float(form_data['monthly_sales']),
            float(form_data['inventory_turnover_rate']),
            int(form_data['vendor_invoices']),
            float(form_data['cash_deposits']),
            float(form_data['mobile_usage_hours']),
            int(form_data['geospatial_transactions']),
            int(form_data['utility_payment_regularity']),
            float(form_data['profit_margin']),
            int(form_data['previous_loans']),
            int(form_data['previous_loan_repayment']),
            int(form_data['dependents']),
            int(form_data['loan_amount_requested']),
            loan_purpose_mapping[form_data['loan_purpose']],
        ]

        # Scale the input features using the saved scaler
        scaled_features = scaler.transform([features])

        # Predict loan approval (1 = Approved, 0 = Not Approved)
        prediction = model.predict(scaled_features)[0]
        result = "Approved" if prediction == 1 else "Not Approved"

    except Exception as e:
        return f"Error: {e}"

    return render_template('result.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
