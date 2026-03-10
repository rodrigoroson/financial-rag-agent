from typing import List, Tuple, Dict
from langchain_core.documents import Document
from src.vector_store import VectorStoreManager
from src.utils.logger import get_logger

logger = get_logger(__name__)

class SpectralRetriever:
    """
    Motor de recuperación semántica que extrae tanto el contexto para el LLM
    como la serie temporal de relevancias S(t) para el análisis espectral 1D.
    """
    def __init__(self, vector_store_manager: VectorStoreManager):
        # Inyectamos la dependencia del gestor de la base de datos
        self.vector_store = vector_store_manager.get_store()

    def get_context_and_signal(self, query: str, top_k: int = 5) -> Tuple[List[Document], List[Dict]]:
        """
        Calculates cosine similarity and returns the best fragments and topological signal.
        
        Args:
            query (str): The user's question.
            top_k (int): Number of fragments to retrieve for generation (standard RAG).
            
        Returns:
            Tuple: 
            - List of the top 'top_k' Documents to inject into the LLM.
            - List of dictionaries with the S(t) signal from all evaluated fragments.
        """
        logger.info(f"Projecting a query in R^768 and calculating the internal product: '{query}'")
        
        try:
            # In very large documents, we limit it to 100 to avoid overloading the memory,
            # but it's enough to visualize the spectral density.
            results = self.vector_store.similarity_search_with_relevance_scores(
                query=query, 
                k=100 
            )
            
            if not results:
                logger.warning("The metric space returned no results.")
                return [], []

            # Extract the strict top_k values ​​for the LLM prompt
            # results is a list of tuples (Document, score)
            top_documents = [doc for doc, _ in results[:top_k]]
            
            # Reconstruct the signal S(t) for scientific visualization
            spectral_signal = []
            for doc, score in results:
                spectral_signal.append({
                    "relative_position": doc.metadata.get("relative_position", 0.0),
                    "chunk_index": doc.metadata.get("chunk_index", 0),
                    "relevance_score": score
                })
                
            # We order the signal topologically
            spectral_signal.sort(key=lambda x: x["relative_position"])
            
            logger.info(f"Retrieved {len(top_documents)} fragments for the LLM and {len(spectral_signal)} points for the S(t) signal.")
            
            return top_documents, spectral_signal
            
        except Exception as e:
            logger.error(f"Error in vector recovery operation: {e}")
            raise