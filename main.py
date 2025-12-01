import os
from dotenv import load_dotenv
from project.pipeline.agents import AgentWorkflow
from project.logger.logging import get_logger

load_dotenv()

logger = get_logger(__name__)


def setup_langsmith():
    langsmith_api_key = os.getenv("LANGSMITH_API_KEY")
    if langsmith_api_key:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
        os.environ["LANGCHAIN_API_KEY"] = langsmith_api_key
        os.environ["LANGCHAIN_PROJECT"] = "rag-corrective-pipeline"
        logger.info("LangSmith tracing enabled")
    else:
        logger.warning("LANGSMITH_API_KEY not found, tracing disabled")


def main():
    setup_langsmith()
    logger.info("Starting RAG application...")
    
    agent = AgentWorkflow()
    
    logger.info("Setting up pipeline with Attention Is All You Need paper...")
    agent.setup(use_attention_paper=True)
    
    agent.save_graph("workflow.png")
    logger.info("Workflow graph saved")
    
    questions = [
        "What is the attention mechanism in transformers?",
        "Explain the multi-head attention.",
        "What are the advantages of the transformer architecture?"
    ]
    
    print("\n" + "="*80)
    print("RAG PIPELINE WITH CORRECTIVE RAG (CRAG)")
    print("="*80 + "\n")
    
    for i, question in enumerate(questions, 1):
        print(f"\n{'='*80}")
        print(f"Question {i}: {question}")
        print(f"{'='*80}\n")
        
        answer = agent.run(question)
        
        print(f"\nAnswer:\n{answer}\n")
        print(f"{'='*80}\n")
    
    logger.info("RAG application completed successfully")


if __name__ == "__main__":
    main()
