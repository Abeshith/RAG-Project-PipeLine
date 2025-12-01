from typing import List, Dict, Any
from langchain.schema import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from project.source.data_preparation import DataPreparation
from project.model.retriever import DocumentRetriever
from project.model.reranking import DocumentReranker
from project.utils.model_loader import ModelLoader
from project.prompts.prompt_template import RAG_PROMPT
from project.logger.logging import get_logger

logger = get_logger(__name__)


class RAGPipeline:
    
    def __init__(self, config_path: str = None):
        self.config_path = config_path
        self.model_loader = ModelLoader(config_path)
        self.llm = self.model_loader.load_llm()
        self.data_prep = DataPreparation()
        self.retriever_module = DocumentRetriever(config_path)
        self.reranker = DocumentReranker(config_path)
        self.chain = None
        self.retriever = None
        logger.info("RAGPipeline initialized")
    
    def setup(self, pdf_path: str = None, use_attention_paper: bool = True):
        chunks = self.data_prep.prepare_documents(
            pdf_path=pdf_path,
            use_attention_paper=use_attention_paper
        )
        
        self.retriever_module.create_vectorstore(chunks)
        self.retriever = self.retriever_module.get_base_retriever()
        
        self._build_chain()
        logger.info("RAG pipeline setup complete")
    
    def _retrieve_and_rerank(self, query: str) -> List[Document]:
        retrieved_docs = self.retriever.invoke(query)
        reranked_docs = self.reranker.rerank(query, retrieved_docs)
        return reranked_docs
    
    def _format_docs(self, docs: List[Document]) -> str:
        return "\n\n".join([
            f"Document {i+1}:\n{doc.page_content}" 
            for i, doc in enumerate(docs)
        ])
    
    def _build_chain(self):
        self.chain = (
            {
                "context": lambda x: self._format_docs(
                    self._retrieve_and_rerank(x["question"])
                ),
                "question": lambda x: x["question"]
            }
            | RAG_PROMPT
            | self.llm
            | StrOutputParser()
        )
        logger.info("RAG chain built successfully")
    
    def invoke(self, query: str) -> str:
        if self.chain is None:
            raise ValueError("Pipeline not setup. Call setup() first.")
        
        response = self.chain.invoke({"question": query})
        logger.info(f"Query processed successfully")
        return response
    
    def get_retrieved_documents(self, query: str) -> List[Document]:
        if self.retriever is None:
            raise ValueError("Pipeline not setup. Call setup() first.")
        
        return self._retrieve_and_rerank(query)
