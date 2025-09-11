from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import List
from pydantic import BaseModel
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

router = APIRouter(prefix="/classify", tags=["Prediction"])

text_adapter = TextAdapter(settings.CLASS_TOKENIZER, settings.AMB_CLASS_PATH, settings.MS_CLASS_PATH, settings.MS_SPLIT_PATH, settings.MODEL_TOKENIZER, settings.MODELS_DIR)

executor = ThreadPoolExecutor(max_workers=4)

class SingleClassificationRequest(BaseModel):
    productType: str
    symptomDescription: str

class Symptom(BaseModel):
    S1: str
    S2: str
    S3: str

class ClassificationResponse(BaseModel):
    multiSymptom: int
    ambiguous: int
    symptoms: List[Symptom]

@router.post(
    "/single",
    response_model=ClassificationResponse,
    summary="Classify a single symptom description",
    description="Predicts S1, S2, S3 codes for a given symptom description."
    )
async def single_classification(request: SingleClassificationRequest):

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