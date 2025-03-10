import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ✅ Load Saved Models
def load_models():
    clf = joblib.load("customer_satisfaction_model.pkl")
    reg = joblib.load("spending_prediction_model.pkl")
    label_encoders = joblib.load("label_encoders.pkl")
    scaler = joblib.load("scaler.pkl")
    return clf, reg, label_encoders, scaler

# ✅ Preprocess Data for Prediction
def preprocess_data(df, label_encoders, scaler):
    df.ffill(inplace=True)  # Fill missing values

    # Encode categorical columns
    categorical_cols = ['Gender', 'Membership Type', 'City']
    for col in categorical_cols:
        df[col] = label_encoders[col].transform(df[col].astype(str).str.strip())

    # Select Features
    X_classification = df[['Gender', 'Membership Type', 'Items Purchased', 'Discount Applied']]
    X_regression = df[['Age', 'Total Spend', 'Items Purchased', 'Days Since Last Purchase']]
    
    # Scale Regression Features
    X_regression = scaler.transform(X_regression)
    
    return X_classification, X_regression

# ✅ Streamlit App
def main():
    st.title("📊 E-Commerce Customer Prediction App")
    
    clf, reg, label_encoders, scaler = load_models()

    uploaded_file = st.file_uploader("📂 Upload Customer Data CSV", type=["csv"])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("### 📋 Uploaded Data Preview")
        st.dataframe(df.head())

        # Preprocess Data
        X_classification, X_regression = preprocess_data(df, label_encoders, scaler)

        # Make Predictions
        predictions_cls = clf.predict(X_classification)
        predictions_reg = reg.predict(X_regression)

        # Add Predictions to DataFrame
        df['Predicted_Satisfaction'] = predictions_cls
        df['Predicted_Spending'] = predictions_reg

        # Display Predictions
        st.write("### 🔍 Predictions")
        st.dataframe(df[['Customer ID', 'Predicted_Satisfaction', 'Predicted_Spending']])

        # Download Button
        st.download_button(
            label="📥 Download Predictions CSV",
            data=df.to_csv(index=False).encode('utf-8'),
            file_name="customer_predictions.csv",
            mime='text/csv'
        )

if __name__ == "__main__":
    main()


