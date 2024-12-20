import cv2
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

class TestInput(BaseModel):
    body_img_url: str | None = None
    tattoo_img_url: str | None = None

@router.post('/test')
async def test(input: TestInput):
    return {"hello", 'world'}