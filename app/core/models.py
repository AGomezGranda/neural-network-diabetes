from pydantic import BaseModel


class PredictionRequest(BaseModel):
    features: list


class PredictionResponse(BaseModel):
    prediction: float
