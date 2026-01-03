from fastapi import FastAPI
from app.api.endpoints import router as api_router
from dotenv import load_dotenv
load_dotenv()

app = FastAPI()

@app.on_event("startup")
async def startup_event():
    # Initialize resources or connections if needed
    pass

@app.on_event("shutdown")
async def shutdown_event():
    # Clean up resources or connections if needed
    pass

app.include_router(api_router, prefix="/api")