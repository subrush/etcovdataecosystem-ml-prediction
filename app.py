# -*- Rush Alemu @ SIS -*-
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model  = pickle.load(open('covid_sev_pre_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """for rendering results on HTML GUI
    """
    int_features = [x for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    
    output = prediction 
    return render_template('index.html', prediction_text='Covid-19 severity prediction [1-recovered 0-deceased]: {}'. format(output))

@app.route('/predict_api',  methods=['POST'])
def predict_api():
    """ for direct API calls through request
    """
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])
    
    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)
    