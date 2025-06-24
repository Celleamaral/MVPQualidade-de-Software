import joblib
import numpy as np

def test_modelo():
    modelo = joblib.load('backend/modelo_review.pkl')
    exemplo = np.array([[4, 3, 5, 4, 4]])
    resultado = modelo.predict(exemplo)
    assert resultado[0] in [1, 2, 3, 4, 5]
