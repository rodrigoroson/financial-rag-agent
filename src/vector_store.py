import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from src.utils.logger import get_logger

# Load GEMINI_API_KEY
load_dotenv()

logger = get_logger(__file__)

class VectorStoreManager:
    """
    Manage the projection of text fragments into the R^768 vector space using Gemini
    and their persistence using ChromaDB. Maintain the topology of the original document
    """
    
    def __init__(self, persist_directory: str | Path = "data/chroma_db"):
        self.persist_directory = str(Path(persist_directory).resolve())
        self.collection_name = "financial_reports_gemini"
        self.vector_store = None
        
        try:
            # Use text-embedding-004
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model="models/text-embedding-004",
                task_type="RETRIEVAL_DOCUMENT" 
            )
        except Exception as e:
            logger.error("Google Embeddings failed to initialize. Please verify your GEMINI_API_KEY in .env.")
            raise e

    def get_store(self) -> Chroma:
        """
        Initializes or recovers the existing vector database on disk.
        """
        if self.vector_store is None:
            logger.info(f"Connecting to ChromaDB at: {self.persist_directory}")
            self.vector_store = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory
            )
        return self.vector_store

    def ingest_documents(self, documents: list[Document]):
        """
        Projects and stores the tensors generated from the discretized documents.

        Args:
            documents(list[Document]): List of fragments processed with spatial metadata.
        """
        if not documents:
            logger.warning("The document input vector is empty. Operation aborted.")
            return

        store = self.get_store()
        
        try:
            logger.info(f"Starting projection in R^768 of {len(documents)} fragments...")
            store.add_documents(documents=documents)
            
            logger.info("Vector ingestion completed successfully.")
            
        except Exception as e:
            logger.error(f"Error during vector ingestion: {str(e)}")
            raise