import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, confusion_matrix
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Load the dataset
data = pd.read_csv(r"cleaned_creditworthiness_data.csv")

# Encode categorical variables
label_encoders = {}
categorical_columns = ['gender', 'education', 'region', 'business_type', 'loan_purpose']

for column in categorical_columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column].astype(str))  # Convert to string and apply label encoding
    label_encoders[column] = le

# Feature and target selection
X = data[['age', 'gender', 'education', 'region', 'business_type', 'monthly_income', 
          'monthly_expenditure', 'transaction_frequency', 'network_size', 'social_credibility_score', 
          'monthly_sales', 'inventory_turnover_rate', 'vendor_invoices', 'cash_deposits', 
          'mobile_usage_hours', 'geospatial_transactions', 'utility_payment_regularity', 
          'profit_margin', 'previous_loans', 'previous_loan_repayment', 'dependents', 
          'loan_amount_requested', 'loan_purpose']]

y = data['loan_status']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)

# Train Logistic Regression model
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# print("Classification Report:\n", classification_report(y_test, y_pred))
# print("Accuracy Score:", accuracy_score(y_test, y_pred))
# print("ROC AUC Score:", roc_auc_score(y_test, y_prob))

# # Plot ROC curve
# fpr, tpr, _ = roc_curve(y_test, y_prob)
# roc_auc = auc(fpr, tpr)
# plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
# plt.plot([0, 1], [0, 1], 'k--')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC Curve')
# plt.legend(loc='lower right')
# plt.show()

# Optional: Save the model and label encoders for future use
with open("creditworthiness_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("label_encoders.pkl", "wb") as f:
    pickle.dump(label_encoders, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
