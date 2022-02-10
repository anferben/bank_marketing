import pickle

from flask import Flask
from flask import request
from flask import jsonify


# Loading the models

output_file = 'rf_model.bin'

with open(output_file, 'rb') as f_in:
    (dv, model) = pickle.load(f_in)


# Creating the web service

app = Flask('mkt_success')

def mkt_campaign_success(customer):
    X = dv.transform([customer])
    proba = model.predict_proba(X)[0, 1]

    if model.predict(X) == 1:
        y_pred = 'success'
    else:
        y_pred = 'not success'

    return proba, y_pred


@app.route('/predict', methods=['POST'])
def predict():
    customer = request.get_json()

    proba, y_pred = mkt_campaign_success(customer=customer)

    results = {
        'Success probability': round(float(proba), 3),
        'Expected result': str(y_pred)
    }

    return jsonify(results)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)
    