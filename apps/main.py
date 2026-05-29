from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from  apps.routers import ClientRouter
from apps.routers import CareerRouter
from dotenv import load_dotenv
import os
from pathlib import Path
app = FastAPI()

model = SentenceTransformer('BAAI/bge-m3')

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # cho tất cả origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

env_path = (
    Path(__file__)
    .resolve()
    .parent / '.env'
)

load_dotenv(env_path)

app.include_router(ClientRouter.router)
app.include_router(CareerRouter.router)

print(os.getenv('MODEL_NAME'))

class RequestData(BaseModel):
    text: str

@app.get('/')
def home ():
    return {
        "message" : "Hello chào bạn đã đến đây hehehehehehehehehe"
    }

@app.post('/embedding')
def embedding(data: RequestData):
    vector = model.encode(data.text).tolist()
    return { "vector" : vector }

