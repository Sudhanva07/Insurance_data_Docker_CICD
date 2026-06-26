import pickle
from flask import Flask, request, jsonify, url_for, redirect, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures


app = Flask(__name__)

# Load the model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json['data']
    print(data)
    polynomial_converter = PolynomialFeatures(degree=4, include_bias = False)
    print(np.array(list(data.values())).reshape(1, -1))
    data_array = np.array(list(data.values())).reshape(1, -1)
    data_array = polynomial_converter.fit_transform(data_array)
    output = model.predict(data_array)
    print(np.exp(output[0]))
    return jsonify(np.exp(output[0]))


@app.route('/predict', methods=['POST'])
def predict():
    data=[float(x) for x in request.form.values()]
    polynomial_converter = PolynomialFeatures(degree=4, include_bias = False)
    final_input = polynomial_converter.fit_transform(np.array(data).reshape(1, -1))
    print(final_input)
    output = np.exp(model.predict(final_input)[0])
    return render_template("index.html", prediction_text="The insurance charges are {}".format(output))

if __name__ == "__main__":
    app.run(debug=True)    