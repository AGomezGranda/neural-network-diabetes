from pydantic import BaseModel, Field
from typing import List, Optional


class PredictionRequest(BaseModel):
    features: list


class PredictionResponse(BaseModel):
    prediction: float
