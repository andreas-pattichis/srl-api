from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.db.config import register_db
from app.essays.models import Essay
from tortoise.contrib.fastapi import HTTPNotFoundError
from tortoise.contrib.pydantic import pydantic_model_creator
from app.essays.routes import router as essays_router

origins = [
    "http://localhost",
    "https://floralearn.org",
]


def get_application() -> FastAPI:
    _app = FastAPI(
        title="SRL API",
        description=""
    )
    _app.include_router(essays_router)
    _app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    register_db(_app)

    return _app


app = get_application()
essay_pydantic = pydantic_model_creator(Essay)


@app.get("/")
def root():
    message = "World"
    return {"Hello": message}