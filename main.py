from fastapi import FastAPI
from src.api  import router as api_router

app = FastAPI()
app.include_router(api_router)
