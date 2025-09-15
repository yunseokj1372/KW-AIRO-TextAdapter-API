from fastapi import FastAPI, Security, HTTPException, Depends
from fastapi.security import APIKeyHeader
import numpy as np
from app.routers import classification 
from app.core.config import settings

# Initialize the app
app = FastAPI(
    title="KW AIRO TextAdapter API",
    description="""
    ## AIRO TextAdapter API for Symptom Classification

    This API provides classification of product symptom descriptions into standardized hierarchical symptom codes.

    ## Features
    - Multi-product Support: Supports 8 different product types (see below)
    - Multi-symptom Detection: Automatically detecs and handles descriptions with multiple symptoms
        - Ex: "The refrigerator is not cooling and ice maker is not working"
    - Ambiguous Description Detection: Automatically detects descriptions that are unclear or ambiguous
        - Ex: "I have a problem with my refrigerator"
    - Standardized Output: Returns consistent hierarchical symptom codes (S1, S2, S3)

    ### Authenitcation
    All endpoints require an API key to be provided in the `X-API-Key` header.  

    ### Supported Product Types
    - `REF_REF`: Refrigerator
    - `WSM_WSM`: Washing Machine
    - `WSM_DRY`: Dryer
    - `VDE_LED`: Television
    - `HKE_OSD`: Dishwasher
    - `HKE_EOV`: Oven
    - `HKE_GRA`: Grill
    - `HKE_MWO`: Microwave
    """,
    version="1.0.1",
    contact={
        "name": "AIRO Support",
        "email": "support@airo.com"
    }
)

# Define the API key header
api_key_header = APIKeyHeader(
    name="X-API-Key",
    auto_error=True,
    description="API key for authentication"
)

async def verify_api_key(api_key: str = Security(api_key_header)):
    """
    Verify the provided API key against the configured secret key

    Args:
        api_key (str): The API key provided in the X-API-Key header to verify

    Returns:
        str: The validated API key

    Raises:
        HTTPException: If the API key is invalid
    """
    if api_key != settings.SECRET_KEY:
        raise HTTPException(
            status_code=411,
            detail="Invalid API key"
        )
    return api_key

# Add routers to the app
app.include_router(
    classification.router,
    dependencies=[Depends(verify_api_key)]
) 


# Root endpoint
@app.get(
    "/", 
    dependencies=[Depends(verify_api_key)],
    summary="Root endpoint",
    description="Returns a welcome message to confirm the API is accessible and authentication is working.",
    response_description="Welcome message",
    responses={
        200: {
            "description": "Successful response",
            "content": {
                "application/json": {
                    "example": {"detail": "Welcome to AIRO's TextAdapters API"}
                }
            }
        },
        411: {
            "description": "Invalid API key",
            "content": {
                "application/json": {
                    "example": {"detail": "Invalid API key"}
                }
            }
        }
    }
)

def read_root():
    """Get API welcome message and confirm authentication"""
    return {"message": "Welcome to AIRO's TextAdapters API"}


@app.get(
    "/health",
    summary="Health Check Endpoint",
    description="""
    Health check endpoint to verify API availability.

    This endpoint does not require authentication and can be used for:
    - Service monitoring
    - Load balancer health checks
    - Deployment verification
    """,
    response_description="API health status",
    responses={
        200: {
            "description": "Successful response",
            "content": {
                "application/json": {
                    "example": {"status": "ok"}
                }
            }
        }
    }
)
def health_check():
    """Health check endpoint to verify API availability"""
    return {"status": "ok"}