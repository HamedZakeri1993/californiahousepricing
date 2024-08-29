import pickle  # Need to load the regression model pickle file
from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np
import pandas as np

# Instance of Flask class
app = Flask(__name__)
# which is a special variable in Python that holds the name of the current module (the script being run)

# load the trained regression model
regmodel = pickle.load(open('regmodel.pkl', 'rb'))
scalar = pickle.load(open('scaling.pkl', 'rb'))
# decorator that tells Flask to execute the function home() whenever someone visits the root URL


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1, -1))
    new_data = scalar.transform(np.array(list(data.values())).reshape(1, -1))
    output = regmodel.predict(new_data)
    print(output[0])
    return jsonify(output[0])


if __name__ == "__main__":
    app.run(debug=True)
