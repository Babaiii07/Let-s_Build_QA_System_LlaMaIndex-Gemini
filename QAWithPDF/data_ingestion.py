from llama_index.core import SimpleDirectoryReader
import sys
from exception import customexception
from logger import logging

def load_data(data):
    try:
        logging.info("data loading started...")
        loader = SimpleDirectoryReader("Data")
        documents=loader.load_data()
        logging.info("data loading completed...")
        return documents
    except Exception as e:
        logging.info("exception in loading data...")
        raise customexception(e,sys)



    