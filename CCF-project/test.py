import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import streamlit as st

# Load data
data = pd.read_csv('creditcard.csv')

# Separate legitimate and fraudulent transactions
valid = data[data.Class == 0]
fraud = data[data.Class == 1]

# Undersample legitimate transactions to balance the classes
valid_sample = valid.sample(n=len(fraud), random_state=2)
data = pd.concat([valid_sample, fraud], axis=0)

# Split data into training and testing sets
X = data.drop(columns="Class", axis=1)
y = data["Class"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=2
)

# Train logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate model performance
train_acc = accuracy_score(model.predict(X_train), y_train)
test_acc = accuracy_score(model.predict(X_test), y_test)

# Create Streamlit app
st.title("Credit Card Fraud Detection Model")
st.write("Enter the feature values as a comma-separated list to check if the transaction is legitimate or fraudulent.")
st.write(f"The model expects {X.shape[1]} features.")

# Input field for user input (text area for all features in one box)
input_data = st.text_area("Input Features (comma-separated):", "")

# Button for submission
submit = st.button("Submit")

if submit:
    try:
        # Parse user input and clean up extra quotes
        input_list = [float(x.strip('"\'')) for x in input_data.split(',')]
        
        # Check if input matches the expected number of features
        if len(input_list) != X.shape[1]:
            st.error(f"Invalid input! Expected {X.shape[1]} features but got {len(input_list)}.")
        else:
            # Convert input to NumPy array and reshape for prediction
            features = np.array(input_list).reshape(1, -1)
            prediction = model.predict(features)

            # Display the prediction result in the same box
            if prediction[0] == 0:
                st.success("The transaction is Valid!")
            else:
                st.error("The transaction is Fraudulent!")
    except ValueError as e:
        # Handle ValueError when input cannot be converted to float
        st.error(f"Invalid input! Please enter numeric values separated by commas. Error: {e}")
