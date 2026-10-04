import os

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL
from sqlalchemy.orm import sessionmaker


load_dotenv()


DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_SERVICE = os.getenv("DB_SERVICE")


required_variables = {
    "DB_USER": DB_USER,
    "DB_PASSWORD": DB_PASSWORD,
    "DB_HOST": DB_HOST,
    "DB_PORT": DB_PORT,
    "DB_SERVICE": DB_SERVICE,
}


missing_variables = [
    name
    for name, value in required_variables.items()
    if not value
]


if missing_variables:
    raise RuntimeError(
        "Missing environment variables: "
        + ", ".join(missing_variables)
    )


database_url = URL.create(
    drivername="oracle+oracledb",
    username=DB_USER,
    password=DB_PASSWORD,
    host=DB_HOST,
    port=int(DB_PORT),
    query={
        "service_name": DB_SERVICE,
    },
)


engine = create_engine(
    database_url,
    pool_pre_ping=True,
)


SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
)


def check_database_connection() -> dict[str, str]:
    query = text(
        """
        SELECT
            SYS_CONTEXT(
                'USERENV',
                'CURRENT_USER'
            ) AS DATABASE_USER,
            SYS_CONTEXT(
                'USERENV',
                'CON_NAME'
            ) AS CONTAINER_NAME
        FROM DUAL
        """
    )

    with engine.connect() as connection:
        result = connection.execute(query).mappings().one()

        return {
            "database_user": result["database_user"],
            "container_name": result["container_name"],
        }


def get_db():
    database = SessionLocal()

    try:
        yield database
    finally:
        database.close()
