import yaml
import pandas as pd
from sqlalchemy import create_engine, text

# Reading the config.yaml file
try:
    with open("config.yaml", "r") as file:
        DB_CONFIG = yaml.safe_load(file)
except FileNotFoundError:
    print("config.yaml not found. Database connection will fail.")
    DB_CONFIG = {}

def get_engine():
    """Creates a SQLAlchemy connection to the PostgreSQL database."""
    try:
        # Construct the database URL using credentials from config.yaml
        db_url = f"postgresql+psycopg2://{DB_CONFIG.get('DB_USER')}:{DB_CONFIG.get('DB_PASSWORD')}@" \
                 f"{DB_CONFIG.get('DB_HOST')}:{DB_CONFIG.get('DB_PORT')}/{DB_CONFIG.get('DB_NAME')}"
        
        # Create the engine object
        engine = create_engine(db_url)
        return engine
    except Exception as e:
        print(f"Database connection failed - {e}")
        # Re-raise the exception to be handled by the calling function
        raise

def run_query_data(sql_query: str, params: dict) -> pd.DataFrame:
    """
    Executes a SELECT query with parameters and returns a DataFrame.
    """
    try:
        engine = get_engine()
        with engine.connect() as connection:
            # Use text() to secure the raw query against SQL injection
            # read_sql_query handles the execution and DataFrame creation
            df = pd.read_sql_query(text(sql_query), connection, params=params)
        return df

    except Exception as e:
        # Return an empty DataFrame on error to prevent the main application from crashing
        print(f"SQLAlchemy Error (SELECT): {e}")
        return pd.DataFrame()

def execute_sql_command(sql_command: str):
    """
    Executes an SQL command (INSERT, UPDATE, DELETE) that does not return data.
    """
    try:
        engine = get_engine()
        with engine.connect() as connection:
            # Execute the command and commit the transaction to the database
            connection.execute(text(sql_command))
            connection.commit()
        return True
    except Exception as e:
        print(f" ERROR: SQL command failed - {e}")
        return False