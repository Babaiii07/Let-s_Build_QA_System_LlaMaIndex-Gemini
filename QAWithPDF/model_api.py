import os
from dotenv import load_dotenv
import sys

from llama_index.llms.gemini import Gemini
from exception import customexception
from logger import logging

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

def load_model():
    try:
        logging.info("Loading Gemini model...")
        model = Gemini(models='gemini-pro', api_key=GOOGLE_API_KEY)
        return model
    except Exception as e:
        logging.error("Error while loading Gemini model.")
        raise customexception(e, sys)
