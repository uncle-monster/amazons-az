from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from backend.api import router as api_router

app = FastAPI(title="Amazons")

app.include_router(api_router)

# Serve the web UI at /
app.mount("/", StaticFiles(directory="web", html=True), name="web")