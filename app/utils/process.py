import re
from typing import List
from fastapi import HTTPException

def preprocess_text(text):
    """
    Preprocess the input text by:
    1. Removing all characters except letters and digits
    2. Converting to lowercase
    3. Removing whitespace

    Args:
        text (str): The input symptom description

    Returns:
        str: Cleaned text
    """

    print(f"Cleaning text: {text}")

    if not text or not isinstance(text, str):
        raise HTTPException(status_code=406, detail="Invalid input text")

    # Convert to lowercase
    text = text.lower()

    # Remove all special characters except spaces
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)

    # Replace multiple spaces with single space and trim
    text = ' '.join(text.split())

    return text

def validate_product_type(product_type):
    """
    Make sure that the product type is valid

    Args:
        product_type (str): The input product type

    Returns:
        bool: True if the product type is valid, False otherwise
    """

    print(f"Validating product type: {product_type}")

    # List of valid product types
    VALID_PRODUCT_TYPES = [
        "REF_REF", # Refrigerator
        "WSM_WSM", # Washing Machine
        "WSM_DRY", # Dryer
        "VDE_LED", # Television
        "HKE_OSD", # Dishwasher
        "HKE_EOV", # Oven
        "HKE_GRA", # Grill
        "HKE_MWO"  # Microwave
    ]

    if not product_type or not isinstance(product_type, str):
        return False

    return product_type.upper() in VALID_PRODUCT_TYPES