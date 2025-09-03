from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, Any
from ...models.schemas import PredictionRequest, PredictionResponse
from ...services.prediction import PredictionService

router = APIRouter(prefix="/prediction", tags=["prediction"])

@router.post("/predict", response_model=PredictionResponse)
async def predict_compatibility(
    request: PredictionRequest,
    prediction_service: PredictionService = Depends()
) -> PredictionResponse:
    try:
        result = await prediction_service.predict(
            user_preferences=request.user_preferences,
            lab_features=request.lab_features
        )
        return PredictionResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))