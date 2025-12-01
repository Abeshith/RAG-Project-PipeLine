import os
from pathlib import Path
from typing import List, Optional
from langchain_community.document_loaders import PyPDFLoader, ArxivLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
from project.logger.logging import get_logger

logger = get_logger(__name__)

class DataPreparation:
    def __init__(
        self,
        data_dir: str = "data",
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        logger.info(f"DataPreparation initialized with chunk_size={chunk_size}")
    
    def load_attention_paper(self, arxiv_id: str = "1706.03762") -> List[Document]:
        pdf_path = self.data_dir / "attention-is-all-you-need.pdf"

        if pdf_path.exists():
            logger.info(f"Loading PDF from local file: {pdf_path}")
            return self._load_pdf(str(pdf_path))

        logger.info(f"PDF not found locally. Downloading from ArXiv: {arxiv_id}")
        try:
            loader = ArxivLoader(query=arxiv_id, load_max_docs=1)
            documents = loader.load()
            if documents:
                logger.info(f"Successfully downloaded paper from ArXiv")
                return documents
            else:
                raise ValueError("No documents returned from ArXiv")
                
        except Exception as e:
            logger.error(f"Failed to download from ArXiv: {str(e)}")
            raise
    
    def _load_pdf(self, pdf_path: str) -> List[Document]:
        try:
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            logger.info(f"Loaded {len(documents)} pages from PDF")
            return documents
        except Exception as e:
            logger.error(f"Failed to load PDF: {str(e)}")
            raise
    
    def load_custom_pdf(self, pdf_path: str) -> List[Document]:
        if not Path(pdf_path).exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        return self._load_pdf(pdf_path)
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        try:
            chunks = self.text_splitter.split_documents(documents)
            logger.info(f"Split documents into {len(chunks)} chunks")
            return chunks
        except Exception as e:
            logger.error(f"Failed to split documents: {str(e)}")
            raise
    
    def prepare_documents(
        self, 
        pdf_path: Optional[str] = None,
        use_attention_paper: bool = True
    ) -> List[Document]:
        try:
            if pdf_path:
                documents = self.load_custom_pdf(pdf_path)
            elif use_attention_paper:
                documents = self.load_attention_paper()
            else:
                raise ValueError("Either provide pdf_path or set use_attention_paper=True")
            
            chunks = self.split_documents(documents)
            
            logger.info(f"Document preparation complete: {len(chunks)} chunks ready")
            return chunks
            
        except Exception as e:
            logger.error(f"Document preparation failed: {str(e)}")
            raise
