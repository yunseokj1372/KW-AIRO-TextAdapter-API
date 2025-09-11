import pandas as pd
from datasets import Dataset, Features, ClassLabel, Value
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, TrainingArguments, Trainer
import torch
import os
from huggingface_hub import login
from dotenv import load_dotenv
from app.core.config import settings

class TextAdapter:
    def __init__(self, classifier_tokenizer, amb_classifier, ms_classifier, ms_split, models_tokenizer, models_path):
        """
        Initialize TextAdapter

        Keyword arguments:
        classifier_tokenizer -- (str) pretrained tokenizer for classifiers
        amb_classifier -- (str) path to ambiguous classifier
        ms_classifier -- (str) path to multi-symptom classifier
        ms_split -- (str) path to multi-symptom identifier
        models_tokenizer -- (str) pretrained tokenizer for product type classifiers
        models_path -- (str) path to directory containing models for each product type
        """
        load_dotenv()
        login(token=settings.TOKEN)

        self.classifier_tokenizer_path = classifier_tokenizer
        self.models_tokenizer_path = models_tokenizer

        self.amb_classifier_path = amb_classifier
        self.ms_classifier_path = ms_classifier
        self.ms_split_path = ms_split
        self.models_path = models_path

        self.classifier_tokenizer = self.load_tokenizer(tokenizer=self.classifier_tokenizer_path)
        self.amb_classifier = self.load_model(model_path=self.amb_classifier_path)
        self.ms_classifier = self.load_model(model_path=self.ms_classifier_path)

        #self.load_ms_split_model(model_path=ms_split)

        self.models_tokenizer = self.load_tokenizer(tokenizer=self.models_tokenizer_path)
        self.all_model = self.load_model(model_path=os.path.join(self.models_path, 'all'))

        #self.ref_ref_model = self.load_model(model_path=os.path.join(self.models_path, 'ref_ref'))
        #self.wsm_wsm_model = self.load_model(model_path=os.path.join(self.models_path, 'wsm_wsm'))
        #self.wsm_dry_model = self.load_model(model_path=os.path.join(self.models_path, 'wsm_dry'))
        #self.vde_led_model = self.load_model(model_path=os.path.join(self.models_path, 'vde_led'))
        #self.hke_osd_model = self.load_model(model_path=os.path.join(self.models_path, 'hke_osd'))
        #self.hke_eov_model = self.load_model(model_path=os.path.join(self.models_path, 'hke_eov'))
        #self.hke_gra_model = self.load_model(model_path=os.path.join(self.models_path, 'hke_gra'))
        #self.hke_mwo_model = self.load_model(model_path=os.path.join(self.models_path, 'hke_mwo'))

    
    def load_model(self, model_path):
        """
        Loads saved fine-tuned model

        Keyword arguments:
        model_path -- (str) path to finetuned model
        """
        try:
            print(f"Loading model from path: {model_path}")
            model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
            print("Model successfully loaded")
            return model
        except Exception as e:
            print(f"Error in load_model: {str(e)}")
            raise

    
    def load_tokenizer(self, tokenizer):
        """
        Loads tokenizer

        Keyword arguments:
        tokenizer -- (str) pretrained tokenizer
        """
        try:
            print(f"Loading tokenizer: {tokenizer}")
            return AutoTokenizer.from_pretrained(tokenizer)
        except Exception as e:
            print(f"Error in load_tokenizer: {str(e)}")
            raise
    
    def load_ms_split_model(self, model_path):
        """
        Loads model for identifying multiple symptoms

        Keyword arguments:
        model_path -- (str) path to pretrianed model
        """
        try:
            print(f"Loading multi-symptom split model: {model_path}")
            self.ms_split_model = AutoModelForCausalLM.from_pretrained(model_path)
            self.ms_split_tokenizer = AutoTokenizer.from_pretrained(model_path)
            print("Model successfully loaded")
            return 
        except Exception as e:
            print(f"Error in load_ms_split_model: {str(e)}")
            raise

    def is_multisymptom(self, description):
        """
        Determines whether description has multiple symptoms

        Keyword arguments:
        description -- (str) description of issue with product

        Output:
        (boolean) indicates whether a description has multiple symptoms
        """
        try:
            print("Running is_multisymptom")
            input_str = "Determine whether the description contains multiple symptoms.\nDescription: " + str(description)
            inputs = self.classifier_tokenizer(input_str, return_tensors="pt", truncation=True, max_length=256)

            with torch.no_grad():
                output = self.ms_classifier.generate(
                    **inputs,
                    max_new_tokens=64,
                    #repitition_penalty=1.2,
                    do_sample=True,
                    top_p=0.9,
                    temperature=0.7,
                    eos_token_id=self.classifier_tokenizer.eos_token_id,
                    pad_token_id=self.classifier_tokenizer.pad_token_id
                )

                prediction = self.classifier_tokenizer.decode(output[0], skip_special_tokens=True)
                print(f"Predicted Multisymptom? {prediction}")
                if prediction == "Yes":
                    return True
                elif prediction == "No":
                    return False
                else:
                    print(f"Error: erroneous prediction: {prediction}")
                    raise
        except Exception as e:
            print(f"Error in is_multisymptom: {e}")

    
    def is_ambiguous(self, description):
        """
        Determines whether description is ambiguous

        Keyword arguments:
        description -- (str) description of issue with product

        Output:
        (boolean) indicates whether a description is ambiguous
        """
        try:
            print("Running is_ambiguous")
            input_str = "Determine whether the description is unclear.\nDescription: " + str(description)
            inputs = self.classifier_tokenizer(input_str, return_tensors="pt", truncation=True, max_length=256)

            with torch.no_grad():
                output = self.amb_classifier.generate(
                    **inputs,
                    max_new_tokens=64,
                    #repitition_penalty=1.2,
                    do_sample=True,
                    top_p=0.9,
                    temperature=0.7,
                    eos_token_id=self.classifier_tokenizer.eos_token_id,
                    pad_token_id=self.classifier_tokenizer.pad_token_id
                )

                prediction = self.classifier_tokenizer.decode(output[0], skip_special_tokens=True)
                print(f"Predicted Ambiguous? {prediction}")
                if prediction == "Yes":
                    return True
                elif prediction == "No":
                    return False
                else:
                    print(f"Error: erroneous prediction: {prediction}")
                    raise
        except Exception as e:
            print(f"Error in is_ambiguous: {e}")
    
    def split_symptoms(self, description):
        """
        Returns the multiple symptoms found in description

        Keyword arguments:
        description -- (str) description of issue with product

        Output:
        (list) list of (str) symptoms
        """
        try:
            print("Running split_symptoms")
            input_str = "Identify the machinery symptoms from the following input.\nInput: " + str(description)
            inputs = self.ms_split_tokenizer(input_str, return_tensors="pt")

            outputs = self.ms_split_model.generate(
                **inputs,
                max_length=100,
                temperature=0.1,
                top_p=0.95,
                do_sample=True
            )

            prediction = self.ms_split_tokenizer.decode(output[0], skip_special_tokens=True)
            print(f"Split symptoms into {len(prediction)}: {prediction}")
            return prediction

        except Exception as e:
            print(f"Error in split_symptoms:{e}")
            raise
    
    # TODO: add more functions for other classes
    def predict_one(self, description, product_type):
        """
        Returns predicted symptoms for REF_REF

        Keyword arguments:
        description -- (str) description of issue with product
        product_type -- (str) product type

        Output:
        Returns a list containing 3 symptoms
        """
        try:
            print(f"Description: {description}")
            input_str = "Predict the failure symptoms.\nDescription:" + str(description)
            input = self.models_tokenizer(input_str, return_tensors="pt", truncation=True, max_length=256)

            with torch.no_grad():
                output = self.all_model.generate(
                    **input,
                    max_new_tokens=64,
                    repetition_penalty=1.2,
                    do_sample=True,
                    top_p=0.9,
                    temperature=0.7,
                    eos_token_id=self.models_tokenizer.eos_token_id,
                    pad_token_id=self.models_tokenizer.pad_token_id
                )

                prediction = self.models_tokenizer.decode(output[0], skip_special_tokens=True)
                symptoms = prediction.split(' > ')
            print(f"Predicted symptoms: {symptoms}")
            return symptoms
        except Exception as e:
            print(f"Error in predict_one: {e}")

#if __name__ == "__main__":
#    test = TextAdapter(
#        classifier_tokenizer="google/flan-t5-base", 
#        amb_classifier="/home/ubuntu/KW-AIRO-TextAdapter-API/app/model/amb_classifier", 
#        ms_classifier="/home/ubuntu/KW-AIRO-TextAdapter-API/app/model/ms_classifier", 
#        ms_split="mistralai/Mistral-7B-Instruct-v0.3", 
#        models_tokenizer="google/flan-t5-large", 
#        models_path="/home/ubuntu/KW-AIRO-TextAdapter-API/app/model/")