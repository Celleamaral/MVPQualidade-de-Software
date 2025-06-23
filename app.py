from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load('modelo_review.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    input_data = np.array([list(data.values())])
    prediction = model.predict(input_data)[0]
    return jsonify({'prediction': int(prediction)})

if __name__ == '__main__':
    app.run(debug=True)
