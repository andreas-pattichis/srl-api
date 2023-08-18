from __future__ import annotations

from pathlib import Path

from pydantic import (
    AnyHttpUrl,
    HttpUrl,
    MySQLDsn,
    field_validator,
)
from pydantic_settings import BaseSettings

try:
    from enum import StrEnum
except ImportError:
    from enum import Enum


    class StrEnum(str, Enum):
        pass


class Environment(StrEnum):
    dev = "dev"
    prod = "prod"


class Paths:
    # srl-backend
    ROOT_DIR: Path = Path(__file__).parent.parent.parent
    BASE_DIR: Path = ROOT_DIR / "app"
    ASSETS_DIR: Path = BASE_DIR / "assets"
    LABEL_NAMES_CSV: Path = ASSETS_DIR / "label_names.csv"
    PROCESS_LABEL_DIR: Path = ASSETS_DIR / "processlabel"


class Settings(BaseSettings):
    @property
    def PATHS(self) -> Paths:
        return Paths()

    ENVIRONMENT: Environment = "dev"
    DEBUG: bool = False
    SERVER_HOST: AnyHttpUrl = "http://localhost:80"  # type:ignore
    PAGINATION_PER_PAGE: int = 20
    MAX_TIME: int = 2700000 # max time = 45 mins

    BACKEND_CORS_ORIGINS: list[AnyHttpUrl] = []
    BACKEND_CORS_ORIGINS: list[AnyHttpUrl] = []

    def assemble_cors_origins(cls, v: str | list[str]) -> list[str] | str:
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, (list, str)):
            return v
        raise ValueError(v)

    DATABASE_URI: str = "mysql://{}:{}@{}:{}/{}".format(
        'user',
        'password',
        'host.docker.internal',
        '3306',
        'flora_annotation',
    )

    class Config:
        env_file = ".env"


settings = Settings()
