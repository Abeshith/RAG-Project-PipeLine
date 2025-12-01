from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from project.pipeline.agents import AgentWorkflow
from project.logger.logging import get_logger
import uvicorn

logger = get_logger(__name__)

app = FastAPI(title="Learn with Transformers")

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

agent = None


@app.on_event("startup")
async def startup_event():
    global agent
    logger.info("Initializing RAG pipeline...")
    agent = AgentWorkflow()
    agent.setup(use_attention_paper=True)
    logger.info("RAG pipeline ready")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/search", response_class=HTMLResponse)
async def search(request: Request, query: str = Form(...)):
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
    import os
    port = int(os.getenv("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
