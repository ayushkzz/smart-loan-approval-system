import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
import pickle

# Load the dataset
data = pd.read_csv(r"cleaned_creditworthiness_data.csv")

# Encode categorical variables
label_encoders = {}
categorical_columns = ['gender', 'education', 'region', 'business_type', 'loan_purpose']

# Encode each categorical column using LabelEncoder
for column in categorical_columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column].astype(str))  # Ensure all values are strings before encoding
    label_encoders[column] = le  # Save the encoder for later use

# Feature selection
X = data[['age', 'gender', 'education', 'region', 'business_type', 'monthly_income', 
          'monthly_expenditure', 'transaction_frequency', 'network_size', 'social_credibility_score', 
          'monthly_sales', 'inventory_turnover_rate', 'vendor_invoices', 'cash_deposits', 
          'mobile_usage_hours', 'geospatial_transactions', 'utility_payment_regularity', 
          'profit_margin', 'previous_loans', 'previous_loan_repayment', 'dependents', 
          'loan_amount_requested', 'loan_purpose']]

y = data['loan_status']

# Handle missing values in features
X.fillna(X.mean(), inplace=True)  # Fill missing values with the column mean

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Scale the features using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fit and transform the training data
X_test_scaled = scaler.transform(X_test)  # Transform the test data based on the training set

# Train the Logistic Regression model
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train_scaled, y_train)

# Save the trained model, scaler, and label encoders for future use
with open('model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

with open('scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)

with open('label_encoders.pkl', 'wb') as le_file:
    pickle.dump(label_encoders, le_file)

print("Model, scaler, and label encoders saved successfully!")
