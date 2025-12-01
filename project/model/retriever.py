from typing import List, Optional
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from project.utils.model_loader import ModelLoader
from project.utils.config_loader import load_config
from project.logger.logging import get_logger

logger = get_logger(__name__)

class DocumentRetriever:
    def __init__(self, config_path: str = None):
        self.config = load_config(config_path)
        self.model_loader = ModelLoader(config_path)
        self.embeddings = self.model_loader.load_embeddings()
        self.llm = self.model_loader.load_llm()
        self.vectorstore = None
        self.retriever = None
        logger.info("DocumentRetriever initialized")
    
    def create_vectorstore(self, documents: List[Document]) -> FAISS:
        self.vectorstore = FAISS.from_documents(documents, self.embeddings)
        logger.info(f"Vector store created with {len(documents)} documents")
        return self.vectorstore
    
    def setup_self_query_retriever(
        self,
        document_content_description: str = "Research papers and technical documents",
        metadata_field_info: Optional[List[AttributeInfo]] = None
    ):
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized. Call create_vectorstore first.")
        
        if metadata_field_info is None:
            metadata_field_info = [
                AttributeInfo(
                    name="source",
                    description="The source file or document name",
                    type="string"
                ),
                AttributeInfo(
                    name="page",
                    description="The page number in the document",
                    type="integer"
                )
            ]
        
        retriever_config = self.config.get('retriever', {})
        
        self.retriever = SelfQueryRetriever.from_llm(
            llm=self.llm,
            vectorstore=self.vectorstore,
            document_contents=document_content_description,
            metadata_field_info=metadata_field_info,
            search_kwargs={
                'k': retriever_config.get('top_k', 3)
            },
            enable_limit=True
        )
        
        logger.info("Self-query retriever configured")
        return self.retriever
    
    def retrieve(self, query: str) -> List[Document]:
        if self.retriever is None:
            raise ValueError("Retriever not initialized. Call setup_self_query_retriever first.")
        
        documents = self.retriever.invoke(query)
        logger.info(f"Retrieved {len(documents)} documents for query")
        return documents
    
    def get_base_retriever(self):
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized.")
        
        retriever_config = self.config.get('retriever', {})
        search_type = retriever_config.get('search_type', 'similarity')
        top_k = retriever_config.get('top_k', 3)
        
        if search_type == 'mmr':
            self.retriever = self.vectorstore.as_retriever(
                search_type='mmr',
                search_kwargs={'k': top_k, 'fetch_k': top_k * 2}
            )
        else:
            self.retriever = self.vectorstore.as_retriever(
                search_type='similarity',
                search_kwargs={'k': top_k}
            )
        
        logger.info(f"Base retriever configured with {search_type} search")
        return self.retriever
