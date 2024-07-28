# app/api/v1/endpoints.py
from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse
from ml.model import NeuralNetwork
from core.models import PredictionRequest, PredictionResponse
import joblib
import numpy as np

router = APIRouter()

# Inicializar el modelo y cargar los parámetros entrenados
model = NeuralNetwork([8, 10, 5, 1])
model.load_model('model.npy')

# Cargar el escalador guardado
scaler = joblib.load('scaler.pkl')


@router.get("/", response_class=HTMLResponse)
async def main():
    with open("client/index.html") as f:
        return HTMLResponse(content=f.read(), status_code=200)

@router.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        features = np.array(request.features).reshape(1, -1)
        # Normalizar los datos de entrada
        features = scaler.transform(features)
        features = features.T  # Transponer para que tenga la forma adecuada para el modelo
        prediction = model.predict(features)
        return PredictionResponse(prediction=float(prediction[0, 0]))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/model-info")
async def model_info():
    return {"model": "Neural Network", "parameters": model.get_params()}
