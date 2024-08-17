#!/usr/bin/env python
# coding: utf-8

from flask import Flask, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

# Load the trained model
with open('fraud_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Define the prediction function
@app.route('/predict', methods=['POST'])
def predict():
    transaction = request.get_json()  # Get the transaction data from the request
    transaction_df = pd.DataFrame([transaction])  # Convert to DataFrame

    # Ensure the columns match the model's expected features
    numerical_features = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 'hour',
                          'transactions_per_account', 'transactions_per_hour', 'avg_transaction_amount',
                          'is_new_account', 'transactions_per_destination', 'change_in_transaction_pattern']
    categorical_features = ['type']

    transaction_df = transaction_df[numerical_features + categorical_features]

    prediction = model.predict(transaction_df)  # Predict using the model

    # Return the result as a JSON response
    return jsonify({'fraud_prediction': bool(prediction[0])})

@app.route('/', methods=['GET'])
def home():
    return "Flask App is Running!"

if __name__ == '__main__':
    app.run(debug=True)


