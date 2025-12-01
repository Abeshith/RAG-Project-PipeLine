# Docker commands for RAG Project

## Build the Docker image
docker build -t rag-project .

## Run the container
docker run -d -p 8000:8000 --name rag-app \
  -e GROQ_API_KEY=your_groq_api_key \
  -e GOOGLE_API_KEY=your_google_api_key \
  -e LANGSMITH_API_KEY=your_langsmith_api_key \
  -e TAVILY_API_KEY=your_tavily_api_key \
  rag-project

## Run with .env file
docker run -d -p 8000:8000 --name rag-app --env-file .env rag-project


