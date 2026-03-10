from typing import List
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from src.utils.logger import get_logger

logger = get_logger(__name__)

class FinancialResponseGenerator:
    """
    A generative operator that synthesizes responses based strictly on the retrieved metric context.
    Uses LCEL to orchestrate the tensor flow to the LLM.
    """
    def __init__(self, temperature: float = 0.0):
        # Initialize the Gemini language model.
        # temperature=0.0 minimizes stochastic variance in inference,
        # forcing the model to adhere to the boundary conditions (the context).
        try:
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                temperature=temperature,
                max_retries=2
            )
            logger.info(f"LLM gemini-1.5-flash inicialized with temperatura {temperature}")
        except Exception as e:
            logger.error(f"Error instantiating Gemini's generative model: {e}")
            raise

        # Define the rules of the inference space
        self.prompt = PromptTemplate.from_template(
            """Eres un analista financiero cuantitativo experto. Tu objetivo es responder a la pregunta del usuario utilizando ÚNICAMENTE el contexto extraído de los informes 10-K/10-Q proporcionados.

            REGLAS DE ORO:
            1. Si la respuesta no está en el contexto, indica explícitamente: "No hay suficiente evidencia en los fragmentos recuperados para responder a esta métrica".
            2. Cita cifras numéricas exactas si aparecen en el contexto.
            3. Mantén un tono técnico, riguroso y objetivo. No uses jerga vacía.

            CONTEXTO EXTRAÍDO:
            {context}

            PREGUNTA DEL USUARIO:
            {query}

            ANÁLISIS FINANCIERO:"""
        )

    def _format_context(self, documents: List[Document]) -> str:
        """
        Collapse the document array into a continuous string.
        """
        return "\n\n---\n\n".join([doc.page_content for doc in documents])

    def generate(self, query: str, context_documents: List[Document]) -> str:
        """
        Perform function composition (Prompt -> LLM -> Parser) via LCEL.

        Args:
            query(str): Original query.
            context_documents(List[Document]): The top_k fragments returned by the Retriever.
        
        Returns:
            str: The final synthesized response.
        """
        if not context_documents:
            logger.warning("Aborted generation: Empty context matrix.")
            return "No spatial context has been recovered to evaluate this query."

        logger.info("Starting generative inference phase (LCEL)...")
        
        try:
            formatted_context = self._format_context(context_documents)
            
            # LCEL functional composition
            chain = self.prompt | self.llm | StrOutputParser()
            
            # Chain evaluation
            response = chain.invoke({
                "context": formatted_context,
                "query": query
            })
            
            logger.info("Generative inference completed successfully.")
            return response
            
        except Exception as e:
            logger.error(f"Error in the inference chain: {e}")
            raise