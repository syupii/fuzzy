from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any

class UserPreferences(BaseModel):
    research_intensity: float = Field(..., ge=0, le=10)
    advisor_style: float = Field(..., ge=0, le=10)
    team_work: float = Field(..., ge=0, le=10)
    workload: float = Field(..., ge=0, le=10)
    theory_practice: float = Field(..., ge=0, le=10)

class LabFeatures(BaseModel):
    research_intensity: float = Field(..., ge=0, le=10)
    advisor_style: float = Field(..., ge=0, le=10)
    team_work: float = Field(..., ge=0, le=10)
    workload: float = Field(..., ge=0, le=10)
    theory_practice: float = Field(..., ge=0, le=10)
    lab_id: str
    lab_name: str

class PredictionRequest(BaseModel):
    user_preferences: UserPreferences
    lab_features: LabFeatures

class PredictionResponse(BaseModel):
    compatibility_score: float = Field(..., ge=0, le=100)
    confidence: float = Field(..., ge=0, le=100)
    explanation: str
    detailed_scores: Dict[str, float]
    model_version: str