from flask import Flask, render_template, request
import pickle
import numpy as np


myapp = Flask(__name__)
model = pickle.load(open('logreg.pkl', 'rb'))


@myapp.route('/')
def home():
    return render_template('home.html')


@myapp.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    features_arrary = [np.array(features)]
    pred = model.predict(features_arrary)
    return render_template('claim.html', data=pred)


myapp.run(debug=True)
