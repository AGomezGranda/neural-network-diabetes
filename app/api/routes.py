from fastapi import APIRouter, HTTPException
from ml import model
from core.models import PredictionRequest, PredictionResponse

router = APIRouter()

# hello route
@router.get("/hello")
async def hello():
    return {"message": "Hello World"}

# preduction route
@router.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        # Lógica de predicción aquí
        prediction = model.predict(request.features)
        return PredictionResponse(prediction=prediction)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# model info route
@router.get("/model-info")
async def model_info():
    return {"model": "Neural Network", "parameters": model.get_params()}
