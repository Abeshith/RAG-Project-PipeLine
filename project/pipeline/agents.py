import os
from typing import List, Literal
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langchain.schema import Document
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import END, StateGraph, START
from project.pipeline.rag import RAGPipeline
from project.utils.model_loader import ModelLoader
from project.prompts.prompt_template import ROUTER_PROMPT, WEB_SEARCH_PROMPT
from project.logger.logging import get_logger

logger = get_logger(__name__)


class GradeDocuments(BaseModel):
    binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")


class GraphState(TypedDict):
    question: str
    generation: str
    web_search: str
    documents: List[str]


class AgentWorkflow:
    
    def __init__(self, config_path: str = None):
        self.config_path = config_path
        self.model_loader = ModelLoader(config_path)
        self.llm = self.model_loader.load_llm()
        self.rag_pipeline = RAGPipeline(config_path)
        self.web_search_tool = None
        self._setup_web_search()
        self.workflow = None
        self.app = None
        self._setup_graders()
        logger.info("AgentWorkflow initialized")
    
    def _setup_web_search(self):
        tavily_key = os.getenv("TAVILY_API_KEY")
        if tavily_key:
            try:
                from langchain_community.tools.tavily_search import TavilySearchResults
                self.web_search_tool = TavilySearchResults(k=3)
                logger.info("Web search tool initialized")
            except Exception as e:
                logger.warning(f"Could not initialize web search: {str(e)}")
                self.web_search_tool = None
        else:
            logger.warning("TAVILY_API_KEY not found, web search disabled")
    
    def _setup_graders(self):
        grade_prompt = """You are a grader assessing relevance of a retrieved document to a user question.
If the document contains keywords or semantic meaning related to the question, grade it as relevant.
Give ONLY a binary score 'yes' or 'no' to indicate whether the document is relevant to the question.

Retrieved document: {document}
User question: {question}

Answer (yes or no):"""
        
        self.grade_prompt_text = grade_prompt
        self.retrieval_grader = self.llm | StrOutputParser()
        
        rewrite_prompt = """You are a question re-writer that converts an input question to a better version optimized for web search.
Look at the input and try to reason about the underlying semantic intent/meaning.
Provide only the improved question without any explanation.

Initial question: {question}

Improved question:"""
        
        self.rewrite_prompt_text = rewrite_prompt
        self.question_rewriter = self.llm | StrOutputParser()
    
    def setup(self, pdf_path: str = None, use_attention_paper: bool = True):
        self.rag_pipeline.setup(pdf_path=pdf_path, use_attention_paper=use_attention_paper)
        self._build_graph()
        logger.info("Agent workflow setup complete")
    
    def retrieve(self, state: GraphState):
        logger.info("---RETRIEVE---")
        question = state["question"]
        documents = self.rag_pipeline.retriever.invoke(question)
        return {"documents": documents, "question": question}
    
    def grade_documents(self, state: GraphState):
        logger.info("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
        question = state["question"]
        documents = state["documents"]
        
        filtered_docs = []
        web_search = "No"
        
        for d in documents:
            prompt_filled = self.grade_prompt_text.format(
                document=d.page_content[:500],
                question=question
            )
            score = self.retrieval_grader.invoke(prompt_filled)
            grade = score.strip().lower()
            
            if "yes" in grade:
                logger.info("---GRADE: DOCUMENT RELEVANT---")
                filtered_docs.append(d)
            else:
                logger.info("---GRADE: DOCUMENT NOT RELEVANT---")
                web_search = "Yes"
        
        return {"documents": filtered_docs, "question": question, "web_search": web_search}
    
    def generate(self, state: GraphState):
        logger.info("---GENERATE---")
        question = state["question"]
        documents = state["documents"]
        
        generation = self.rag_pipeline.chain.invoke({"question": question})
        return {"documents": documents, "question": question, "generation": generation}
    
    def transform_query(self, state: GraphState):
        logger.info("---TRANSFORM QUERY---")
        question = state["question"]
        documents = state["documents"]
        
        prompt_filled = self.rewrite_prompt_text.format(question=question)
        better_question = self.question_rewriter.invoke(prompt_filled)
        
        return {"documents": documents, "question": better_question}
    
    def web_search(self, state: GraphState):
        logger.info("---WEB SEARCH---")
        question = state["question"]
        documents = state["documents"]
        
        if self.web_search_tool is None:
            logger.warning("Web search tool not available, skipping")
            return {"documents": documents, "question": question}
        
        try:
            response = self.web_search_tool.invoke({"query": question})
            
            if not response:
                logger.warning("No results from web search")
                return {"documents": documents, "question": question}
            
            web_results = "\n".join([d["content"] for d in response if "content" in d])
            web_doc = Document(page_content=web_results)
            documents.append(web_doc)
        except Exception as e:
            logger.error(f"Web search failed: {str(e)}")
        
        return {"documents": documents, "question": question}
    
    def decide_to_generate(self, state: GraphState) -> Literal["transform_query", "generate"]:
        logger.info("---ASSESS GRADED DOCUMENTS---")
        documents = state.get("documents", [])
        
        if len(documents) == 0:
            logger.info("---DECISION: NO RELEVANT DOCUMENTS, TRANSFORM QUERY---")
            return "transform_query"
        else:
            logger.info("---DECISION: RELEVANT DOCUMENTS FOUND, GENERATE---")
            return "generate"
    
    def _build_graph(self):
        workflow = StateGraph(GraphState)
        
        workflow.add_node("retrieve", self.retrieve)
        workflow.add_node("grade_documents", self.grade_documents)
        workflow.add_node("generate", self.generate)
        workflow.add_node("transform_query", self.transform_query)
        workflow.add_node("web_search", self.web_search)
        
        workflow.add_edge(START, "retrieve")
        workflow.add_edge("retrieve", "grade_documents")
        workflow.add_conditional_edges(
            "grade_documents",
            self.decide_to_generate,
            {
                "transform_query": "transform_query",
                "generate": "generate",
            },
        )
        workflow.add_edge("transform_query", "web_search")
        workflow.add_edge("web_search", "generate")
        workflow.add_edge("generate", END)
        
        self.app = workflow.compile()
        logger.info("LangGraph workflow compiled")
    
    def save_graph(self, output_path: str = "workflow.png"):
        try:
            from IPython.display import Image
            graph_image = self.app.get_graph().draw_mermaid_png()
            with open(output_path, "wb") as f:
                f.write(graph_image)
            logger.info(f"Workflow graph saved to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save graph: {str(e)}")
    
    def run(self, question: str) -> str:
        if self.app is None:
            raise ValueError("Workflow not setup. Call setup() first.")
        
        inputs = {"question": question}
        
        for output in self.app.stream(inputs):
            for key, value in output.items():
                logger.info(f"Node '{key}' completed")
        
        final_generation = value.get("generation", "No answer generated")
        return final_generation
