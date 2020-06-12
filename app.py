# -*- Rush Alemu @ SIS -*-
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model  = pickle.load(open('covid_sev_pre_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('cov_sev_pre_frm.html')

@app.route('/predict', methods=['POST'])
def predict():
    """for rendering results on HTML GUI
    """
    int_features = [x for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    
    output = prediction 
    if prediction == 1:
        output = 'Recovered'
    elif prediction == 0:
        output = 'Deceased'
        
    return render_template('cov_sev_pre_frm.html', prediction_text='The result will be {}'.format(output))

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
    