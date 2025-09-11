from fastapi import FastAPI, Security, HTTPException, Depends
from fastapi.security import APIKeyHeader
import numpy as np
from app.routers import classification 
from app.core.config import settings

# Initialize the app
app = FastAPI()

# Define the API key header
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)

async def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != settings.SECRET_KEY:
        raise HTTPException(
            status_code=411,
            detail="Invalid API key bleh bleh"
        )
    return api_key

# Add routers to the app
app.include_router(
    classification.router,
    dependencies=[Depends(verify_api_key)]
) 


# Root endpoint
@app.get("/", dependencies=[Depends(verify_api_key)])
def read_root():
    return {"message": "Welcome to AIRO's TextAdapters API"}


@app.get("/health")
def health_check():
    return {"status": "ok"}