from flask import Flask, jsonify, request
from marshmallow import ValidationError

from ml.ml import MachineLearningModel
from model import PredictReq
from model import PredictRes

app = Flask(__name__)


@app.route('/predict', methods=['POST'], endpoint='predict')
def predict():
    content = request.json
    req_model = PredictReq()
    try:
        result = req_model.load(content)
        value = ml_model.predict(result['radius_mean'], result['perimeter_mean'], result['area_mean'],
                                 result['concavity_mean'],
                                 result['concave_points_mean'], result['radius_worst'], result['perimeter_worst'],
                                 result['area_worst'], result['concavity_worst'], result['concave_points_worst'])
        response = PredictRes(value)
    except ValidationError as err:
        return jsonify(err.messages), 400
    return response.to_json(), 200


if __name__ == '__main__':
    print('init: ML Model...started')
    ml_model = MachineLearningModel()
    print('init:ML Model is completed')
    app.run(host='0.0.0.0', port=5000)
