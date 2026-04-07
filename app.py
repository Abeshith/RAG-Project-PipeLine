from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from project.pipeline.agents import AgentWorkflow
from project.logger.logging import get_logger
import uvicorn
import asyncio
from contextlib import asynccontextmanager

logger = get_logger(__name__)

# Global initialization state
agent = None
initialization_complete = False
initialization_error = None


async def initialize_rag_pipeline():
    """Background task to initialize RAG pipeline"""
    global agent, initialization_complete, initialization_error
    try:
        logger.info("Initializing RAG pipeline in background...")
        agent = AgentWorkflow()
        agent.setup(use_attention_paper=True)
        initialization_complete = True
        logger.info("RAG pipeline ready")
    except Exception as e:
        initialization_error = str(e)
        logger.error(f"Initialization failed: {e}")
        initialization_complete = True


@asynccontextmanager
async def lifespan(app: FastAPI):
    """App lifecycle manager - returns immediately, initializes in background"""
    # Startup
    logger.info("App starting up - launching background initialization")
    asyncio.create_task(initialize_rag_pipeline())
    yield
    # Shutdown
    logger.info("App shutting down")


app = FastAPI(title="Learn with Transformers", lifespan=lifespan)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.get("/health")
async def health():
    """Liveness probe - returns immediately (HF Spaces requirement)"""
    return JSONResponse({"status": "alive"}, status_code=200)


@app.get("/ready")
async def readiness():
    """Readiness probe - checks if initialization complete"""
    if initialization_error:
        return JSONResponse(
            {"status": "failed", "error": initialization_error}, 
            status_code=503
        )
    if not initialization_complete:
        return JSONResponse(
            {"status": "initializing"}, 
            status_code=503
        )
    return JSONResponse({"status": "ready"}, status_code=200)


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    if not initialization_complete:
        return templates.TemplateResponse(
            "index.html", 
            {"request": request, "message": "System initializing... Please wait"}
        )
    if initialization_error:
        return templates.TemplateResponse(
            "index.html", 
            {"request": request, "error": f"Initialization failed: {initialization_error}"}
        )
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/search", response_class=HTMLResponse)
async def search(request: Request, query: str = Form(...)):
    if initialization_error:
        return templates.TemplateResponse(
            "index.html", 
            {"request": request, "error": f"System error: {initialization_error}"}
        )
    
    if not initialization_complete:
        return templates.TemplateResponse(
            "index.html", 
            {"request": request, "error": "System still initializing. Please try again in a moment"}
        )
    
    if not query.strip():
        return templates.TemplateResponse(
            "index.html", 
            {"request": request, "error": "Please enter a question"}
        )
    
    try:
        answer = agent.run(query)
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "query": query, "answer": answer}
        )
    except Exception as e:
        logger.error(f"Search failed: {str(e)}")
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "error": f"Error: {str(e)}"}
        )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
