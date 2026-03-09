import os
from pathlib import Path
from typing import List, Union
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from src.utils.logger import get_logger

# logging setup
logger = get_logger(Path(__file__).stem)

class FinancialDocumentProcessor:
    """
    Process financial reports (10-K, 10-Q) preparing them for vector injection.
    Preserve the spatial continuity of the text.
    """
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 150):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ".", " ", ""]
        )

    def process_pdf(self, file_path: Union[str, Path]) -> List[Document]:
        """
        Load and split.
        
        Args:
            file_path (Union[str, Path]): PDF Path.
            
        Returns:
            List[Document]: List of text fragments.
        """
        path_obj = Path(file_path)
        
        if not path_obj.exists() or not path_obj.is_file():
            raise FileNotFoundError(f"File not found or invalid in: {path_obj.resolve()}")

        try:
            logger.info(f"Loading document from: {path_obj.resolve()}")
            
            # Cast to str
            loader = PyPDFLoader(str(path_obj))
            pages = loader.load()
            
            logger.info("Applying text chunking...")
            chunks = self.splitter.split_documents(pages)
            
            total_chunks = len(chunks)
            if total_chunks == 0:
                logger.warning("The document did not generate any valid fragments.")
                return []

            # Metadata update to preserve spatial context
            for i, chunk in enumerate(chunks):
                chunk.metadata.update({
                    "chunk_index": i,
                    "total_chunks": total_chunks,
                    "relative_position": i / (total_chunks - 1) if total_chunks > 1 else 0.0
                })
            
            logger.info(f"Processing successful. {total_chunks} fragments generated with spatial metadata.")
            return chunks
        
        except Exception as e:
            logger.error(f"Unexpected error processing {path_obj.name}: {str(e)}")
            raise