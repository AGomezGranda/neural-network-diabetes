from fastapi import APIRouter
router = APIRouter()

# hello route
@router.get("/hello")
async def hello():
    return {"message": "Hello World"}

# preduction route

# model info route

