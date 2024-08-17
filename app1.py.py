#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[2]:


from flask import Flask, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

# Load the trained model
with open('fraud_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Define the endpoint for prediction
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    transaction = pd.DataFrame([data])
    prediction = model.predict(transaction)
    return jsonify({'fraud': bool(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)


# In[ ]:




