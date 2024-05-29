# From https://medium.com/@talhakhalid101/python-tortoise-orm-integration-with-fastapi-c3751d248ce1
from tortoise import Tortoise
from tortoise.contrib.fastapi import register_tortoise
from fastapi import FastAPI
from app.core.config import settings


TORTOISE_ORM = {
    "connections": {
        "flora_annotation": settings.FLORA_ANNOTATION_DATABASE_URI,
        "moodle": settings.MOODLE_DATABASE_URI
    },
    "apps": {
        "models": {
            "models": [
                'app.trace_data.models',
                'app.essays.models',
            ],
            "default_connection": "flora_annotation",
        },
    },
}


async def connect_to_db():
    await Tortoise.init(
        config=TORTOISE_ORM
    )


def register_db(app: FastAPI) -> None:
    register_tortoise(
        app,
        config=TORTOISE_ORM,
        generate_schemas=False,
        add_exception_handlers=True,
    )
