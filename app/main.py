from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

origins = [
    "http://localhost",
    "https://floralearn.org",
]


def get_application() -> FastAPI:
    _app = FastAPI(
        title="SRL API",
        description=""
    )
    _app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    return _app


app = get_application()

@app.get("/")
def root():
    message = "World"
    return {"Hello": message}


@app.get("/result/{username}")
def get_result(username: str):
    print(username)

    return {"username": username}
