from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import List
from pydantic import BaseModel, Field
from concurrent.futures import ThreadPoolExecutor
import logging
from app.models.TextAdapter import TextAdapter
from app.core.config import settings
from app.utils.process import preprocess_text, validate_product_type

# Set up file logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('prediction.log'),
        logging.StreamHandler()  # Also print to console
    ]
)
logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/classify", 
    tags=["Classification"],
    responses={
        411: {
            "description": "Invalid API key",
            "content": {
                "application/json": {
                    "example": {"detail": "Invalid API key"}
                }
            }
        },
        500: {
            "description": "Internal server error",
            "content": {
                "application/json": {
                    "example": {"detail": "Internal server error"}
                }
            }
        }
    }
)

text_adapter = TextAdapter(settings.CLASS_TOKENIZER, settings.AMB_CLASS_PATH, settings.MS_CLASS_PATH, settings.MS_SPLIT_PATH, settings.MODEL_TOKENIZER, settings.MODELS_DIR)

executor = ThreadPoolExecutor(max_workers=4)

class SingleClassificationRequest(BaseModel):
    """Request model for single symptom classification"""
    productType: str = Field(
        ...,
        description="Product type identifier",
        example="REF_REF",
        pattern="^(REF_REF|WSM_WSM|WSM_DRY|VDE_LED|HKE_OSD|HKE_EOV|HKE_GRA|HKE_MWO)$"
    )
    symptomDescription: str = Field(
        ...,
        description="Symptom description",
        example="The refrigerator is not cooling"
    )

    class Config:
        schema_extra = {
            "example": {
                "productType": "REF_REF",
                "symptomDescription": "The refrigerator is not cooling"
            }
        }

class Symptom(BaseModel):
    """
    Individual symptom classification with three-level hierarchy
    """
    S1: str = Field(
        ...,
        description="Primary symptom category (highest level classification)",
        example="Cooling/Temperature/Condensation"
    )
    S2: str = Field(
        ...,
        description="Secondary symptom category (middle level classification)",
        example="No cooling"
    )
    S3: str = Field(
        ...,
        description="Tertiary symptom category (lowest level classification)",
        example="All room"
    )

    class Config:
        schema_extra = {
            "example": {
                "S1": "Cooling/Temperature/Condensation",
                "S2": "No cooling",
                "S3": "All room"
            }
        }

class ClassificationResponse(BaseModel):
    """Response model for symptom classification results"""
    multiSymptom: int = Field(
        ...,
        description="Indicates if multiple distinct symptoms were detected in the description",
        example=0,
        ge=0,
        le=1
    )
    ambiguous: int = Field(
        ...,
        description="Indicates if the symptom description is too unclear or ambiguous to classify reliably",
        example=0,
        ge=0,
        le=1
    )
    symptoms: List[Symptom] = Field(
        ...,
        description="List of classified symptoms. Empty array if ambiguous=1. Contains multiple items if multiSymptom=1",
        example=[
            {
                "S1": "Cooling/Temperature/Condensation",
                "S2": "No cooling",
                "S3": "All room"
            },
            {
                "S1": "Design",
                "S2": "Exterior design",
                "S3": "Dented door"
            }
        ]
    )

    class Config:
        schema_extra = {
            "examples": {
                "single_symptom": {
                    "summary": "Single clear symptom",
                    "value": {
                        "multiSymptom": 0,
                        "ambiguous": 0,
                        "symptoms": [
                            {
                                "S1": "Cooling/Temperature/Condensation",
                                "S2": "No cooling",
                                "S3": "All room"
                            }
                        ]
                    }
                },
                "multiple_symptoms": {
                    "summary": "Multiple distinct symptoms",
                    "value": {
                        "multiSymptom": 1,
                        "ambiguous": 0,
                        "symptoms": [
                            {
                                "S1": "Cooling/Temperature/Condensation",
                                "S2": "No cooling",
                                "S3": "All room"
                            },
                            {
                                "S1": "Design",
                                "S2": "Exterior design",
                                "S3": "Dented door"
                            }
                        ]
                    }
                },
                "ambiguous_description": {
                    "summary": "Ambiguous description",
                    "value": {
                        "multiSymptom": 0,
                        "ambiguous": 1,
                        "symptoms": []
                    }
                }
            }
        }

@router.post(
    "/single",
    response_model=ClassificationResponse,
    summary="Classify a single symptom description",
    description="""
    ## Classifies a single product symptom description into standardized symptom codes

    This endpoint analyzes natural language descriptions of product issues and returns structured hierarchical symptom classifications.

    ### Process Flow
    1. Input Validation: Validates product type and description format
    2. Text Preprocessing: Cleans and normalizes the input description
    3. Multi-symptom Detection: Determined if the description contains multiple distinct symptoms
    4. Ambiguity Check: Flags descriptions that are unclear or ambiguous
    5. Symptom Classification: Generates S1, S2, S3 codes for each identified symptom

    ### Classification Logic
    - Single Symptom: Returns one symptom object with multiSymptom=0 and ambiguous=0
    - Multiple Symptoms: Returns multiple symptom objects with multiSymptom=1 and ambiguous=0
    - Ambiguous Input: Returns empty symptoms array with multiSymptom=0 and ambiguous=1

    ### Examples Description Cases
    - "Fridge not cooling" → Single symptom classification
    - "Washing machine won't start and makes noise" → Multiple symptom classification
    - "Something is wrong" → Ambiguous classification
    """,
    response_description="Structured symptom classification with metadata indicators for multi-symptom and ambiguous cases",
    responses={
        200: {
            "description": "Successful response",
            "content": {
                "application/json": {
                    "examples": {
                        "single_symptom": {
                            "summary": "Single clear symptom",
                            "value": {
                                "multiSymptom": 0,
                                "ambiguous": 0,
                                "symptoms": [
                                    {
                                        "S1": "Cooling/Temperature/Condensation",
                                        "S2": "No cooling",
                                        "S3": "All room"
                                    }
                                ]
                            }
                        },
                        "multiple_symptoms": {
                            "summary": "Multiple distinct symptoms",
                            "value": {
                                "multiSymptom": 1,
                                "ambiguous": 0,
                                "symptoms": [
                                    {
                                        "S1": "Cooling/Temperature/Condensation",
                                        "S2": "No cooling",
                                        "S3": "All room"
                                    },
                                    {
                                        "S1": "Design",
                                        "S2": "Exterior design",
                                        "S3": "Dented door"
                                    }
                                ]
                            }
                        },
                        "ambiguous_description": {
                            "summary": "Ambiguous description",
                            "value": {
                                "multiSymptom": 0,
                                "ambiguous": 1,
                                "symptoms": []
                            }
                        }
                    }
                }
            }
        },
        406: {
            "description": "Invalid Product Type",
            "content": {
                "application/json": {
                    "example": {"detail": "Invalid product type"}
                }
            }
        },
        422: {
            "description": "Invalid Input",
            "content": {
                "application/json": {
                    "example": {"detail": "Invalid input"}
                }
            }
        },
        500: {
            "description": "Internal Server Error",
            "content": {
                "application/json": {
                    "example": {"detail": "Internal server error"}
                }
            }
        }
    }
)
async def single_classification(request: SingleClassificationRequest):
    """
    Classifies a product symptom description into standardized hierarchical symptom codes.

    Args:
        request (SingleClassificationRequest): Classification request containing product type and symptom description

    Returns:
        ClassificationResponse: Structured symptom classification results

    Raises:
        HTTPException 406: Invalid product type
        HTTPException 422: Invalid input
        HTTPException 500: Internal server error
    """

    try:

        # Validate product type
        if not validate_product_type(request.productType):
            raise HTTPException(
                status_code=406,
                detail="Invalid product type"
            )

        # Preprocess the symptom description
        cleaned_description = preprocess_text(request.symptomDescription)
        
        # Check if description is multi-symptom or ambiguous
        is_multisymptom = text_adapter.is_multisymptom(cleaned_description)
        is_ambiguous = text_adapter.is_ambiguous(cleaned_description)

        # is_multisymptom = False
        # is_ambiguous = False

        symptoms = []

         # If ambiguous → return immediately, no predictions
        if is_ambiguous:
            return ClassificationResponse(
                multiSymptom=0,
                ambiguous=1,
                symptoms=[]
            )

        # If multi-symptom → split symptoms and predict each one
        if is_multisymptom:

            symptom_descriptions = text_adapter.split_symptoms(cleaned_description)
            for desc in symptom_descriptions:
                prediction = text_adapter.predict_one(desc, request.productType)
                symptoms.append(Symptom(
                    S1=prediction[0],
                    S2=prediction[1],
                    S3=prediction[2]
                ))
            
        else:
            print(f"Predicting symptoms for: {cleaned_description}")
            prediction = text_adapter.predict_one(cleaned_description, request.productType)
            symptoms = [Symptom(
                S1=prediction[0],
                S2=prediction[1],
                S3=prediction[2]
            )]

        return ClassificationResponse(
            multiSymptom=1 if is_multisymptom else 0,
            ambiguous=1 if is_ambiguous else 0,
            symptoms=symptoms if not is_ambiguous else []
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))