from typing import List
from langchain.schema import Document
from flashrank.Ranker import Ranker, RerankRequest
from project.utils.config_loader import load_config
from project.logger.logging import get_logger

logger = get_logger(__name__)


class DocumentReranker:
    
    def __init__(self, config_path: str = None):
        self.config = load_config(config_path)
        reranker_config = self.config.get('reranker', {})
        model_name = reranker_config.get('model_name', 'rank-T5-flan')
        cache_dir = reranker_config.get('cache_dir')
        self.top_k = reranker_config.get('top_k', 3)
        
        if cache_dir:
            self.ranker = Ranker(model_name=model_name, cache_dir=cache_dir)
        else:
            self.ranker = Ranker(model_name=model_name)
        
        logger.info(f"FlashRank reranker initialized with model: {model_name}")
    
    def rerank(
        self, 
        query: str, 
        documents: List[Document], 
        top_k: int = None
    ) -> List[Document]:
        
        if top_k is None:
            top_k = self.top_k
        
        if not documents:
            logger.warning("No documents to rerank")
            return []
        
        passages = [
            {
                "id": i,
                "text": doc.page_content,
                "meta": doc.metadata
            }
            for i, doc in enumerate(documents)
        ]
        
        rerank_request = RerankRequest(query=query, passages=passages)
        
        results = self.ranker.rerank(rerank_request)
        
        reranked_docs = []
        for result in results[:top_k]:
            doc_idx = result["id"]
            original_doc = documents[doc_idx]
            original_doc.metadata["rerank_score"] = result["score"]
            reranked_docs.append(original_doc)
        
        logger.info(f"Reranked {len(documents)} documents, returning top {len(reranked_docs)}")
        return reranked_docs
