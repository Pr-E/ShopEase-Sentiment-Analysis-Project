from pydantic import BaseModel
from typing import List


class TextRequest(BaseModel):
    text: str


class BatchRequest(BaseModel):
    texts: List[str]


class PredictionResponse(BaseModel):
    label: str
    class_id: int
    confidence: float


class BatchResponse(BaseModel):
    predictions: List[PredictionResponse]    