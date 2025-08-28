from fastapi import FastAPI
import numpy as np
from app.routers import prediction 

# Initialize the app
app = FastAPI()

# Add routers to the app
app.include_router(prediction.router) 

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to AIRO's TextAdapters API"}


@app.get("/health")
def health_check():
    return {"status": "ok"}