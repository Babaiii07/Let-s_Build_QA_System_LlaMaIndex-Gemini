from llama_index.core import VectorStoreIndex
from llama_index.embeddings.gemini import GeminiEmbedding

from QAWithPDF.data_ingestion import load_data
from QAWithPDF.model_api import load_model
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Settings

import sys
from exception import customexception
from logger import logging

def download_gemini_embedding(model,document):
    try:
        logging.info("")
        gemini_embed_model = GeminiEmbedding(model_name="models/embedding-001")

        Settings.llm = model
        Settings.embed_model = gemini_embed_model
        Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=20)
        Settings.num_output = 512
        Settings.context_window = 3900
        
        logging.info("")
        index = VectorStoreIndex.from_documents(document,settings=Settings)
        index.storage_context.persist()
        
        logging.info("")
        query_engine = index.as_query_engine()
        return query_engine
    except Exception as e:
        raise customexception(e,sys)