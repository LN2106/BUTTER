import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier  # Example model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Title and Description
st.title("Customer Churn Prediction App")
st.write("This app predicts if a customer will churn based on input features.")

# Sidebar for navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox("Choose the page", ["Home", "Upload Data", "EDA", "Prediction", "Model Evaluation"])

if app_mode == "Home":
    st.subheader("Welcome to the Churn Prediction App!")
    st.write("Navigate to different sections using the sidebar.")


if app_mode == "Upload Data":
    st.subheader("Upload your dataset")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("Here's a preview of your data:", data.head(15))



import matplotlib.pyplot as plt
import seaborn as sns

if app_mode == "EDA":
    st.subheader("Exploratory Data Analysis")
    
    # Ensure data is uploaded
    if uploaded_file is not None:
        st.write("Data Overview:")
        st.write(data.describe())  # Show summary statistics of the data
        
        # Correlation heatmap
        st.write("Correlation Heatmap:")
        fig, ax = plt.subplots()
        sns.heatmap(data.corr(), ax=ax, annot=True, cmap="coolwarm")  # Heatmap of correlations
        st.pyplot(fig)

        # Histogram for feature distributions
        st.write("Distribution of Features:")
        feature = st.selectbox("Choose feature", data.columns)  # Dropdown to select a feature
        fig, ax = plt.subplots()
        sns.histplot(data[feature], kde=True, ax=ax)  # Plot histogram for the selected feature
        st.pyplot(fig)

    else:
        st.write("Please upload a CSV file to perform EDA.")

if app_mode == "Prediction":
    st.subheader("Customer Churn Prediction")
    
    # Input fields for features
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=72)
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, step=0.1)
    total_charges = st.number_input("Total Charges ($)", min_value=0.0, step=0.1)
    
    # Model inference
    model = RandomForestClassifier()  # Replace this with your trained model
    X = data[["tenure", "monthlycharges", "totalcharges"]]  # Sample features
    y = data["churn"]  # Target variable
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model.fit(X_train, y_train)
    
    # Prediction button
    if st.button("Predict"):
        input_data = np.array([[tenure, monthly_charges, total_charges]])
        prediction = model.predict(input_data)
        st.write("The prediction is:", "Churn" if prediction[0] == 1 else "Not Churn")


if app_mode == "Model Evaluation":
    st.subheader("Model Evaluation")
    
    if uploaded_file is not None:
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        
        st.write(f"Accuracy: {accuracy}")
        st.write("Classification Report:")
        st.text(report)
