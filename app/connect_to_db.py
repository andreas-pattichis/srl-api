# From https://medium.com/@talhakhalid101/python-tortoise-orm-integration-with-fastapi-c3751d248ce1
from tortoise import Tortoise


DATABASE_URL = "mysql://{}:{}@{}:{}/{}".format(
    'user',
    'password',
    'localhost',
    '3306',
    'flora_annotation',
)


async def connect_to_db():
    await Tortoise.init(
        db_url=DATABASE_URL,
        modules={'models': [
            'app.users.models',
            'app.essays.models'
        ]},
    )
