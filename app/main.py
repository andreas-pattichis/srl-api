from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.db.config import register_db
from app.trace_data.routes import router as tracedata_router
from app.essays.routes import router as essays_router

origins = [
    '*'
]

def get_application() -> FastAPI:
    _app = FastAPI(
        title="SRL API",
        description="",
        docs_url="/docs",
    )
    _app.include_router(tracedata_router)
    _app.include_router(essays_router)  # Include the essays router
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


@app.get("/")
def root():
    message = "World"
    return {"Hello": message}
