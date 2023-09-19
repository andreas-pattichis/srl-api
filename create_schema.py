from tortoise import Tortoise, run_async
from app.db.config import connect_to_db
import argparse
from app.essays.models import query


class CreateDB:
    def __init__(self):
        parser = argparse.ArgumentParser(description="Database generator")
        parser.add_argument("-s", "--seed", help="Seed database entries", required=False, default="True")
        parser.add_argument("-g", "--generate", help="Generate schemas", required=False, default="True")

        self.argument = parser.parse_args()

        run_async(self.main())

    async def main(self):
        await connect_to_db()

        if self.argument.generate:
            await Tortoise.generate_schemas()

        if self.argument.seed:
            conn = Tortoise.get_connection("default")
            await conn.execute_query(query())

        await Tortoise.close_connections()


if __name__ == '__main__':
    app = CreateDB()
