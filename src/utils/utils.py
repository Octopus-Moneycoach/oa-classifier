import atexit
import importlib
import logging
import os
from pathlib import Path
from typing import Any, cast

import pandas as pd
import pandera as pa

# import snowflake related packages
import snowflake.connector
import yaml
from cryptography.hazmat.primitives import serialization
from dotenv import dotenv_values
from snowflake.connector.pandas_tools import write_pandas

import src.utils.schemas as schemas_module

# Reload schemas in case schemas.py has changed
schemas_module = importlib.reload(schemas_module)
schemas = schemas_module.schemas

_ENV_PATH = Path(".env")
_created_env = False

if not _ENV_PATH.exists():
    _ENV_PATH.write_text("")  # or put minimal keys if needed
    _created_env = True


def _cleanup_env() -> None:
    """Cleanup the dummy .env file if it was created by this module."""
    if _created_env:
        try:
            _ENV_PATH.unlink()
        except FileNotFoundError:
            pass


atexit.register(_cleanup_env)

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)
logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(
    logging.WARNING
)


def read_config() -> dict[str, Any]:
    """Read the YAML configuration file and return its contents.

    Returns:
        dict[str, Any]: Configuration settings loaded from ``cfg/config.yaml``.
    """
    config_file_path = os.path.join("cfg", "config.yaml")
    with open(config_file_path, "r", encoding="utf-8") as file:
        config_data = yaml.safe_load(file)
    return cast(dict[str, Any], config_data)


def read_file_config(file_name: str) -> dict[str, Any]:
    """Read configuration for a specific file from ``cfg/files.yaml``.

    Args:
        file_name: Key that identifies the file in ``files.yaml``.

    Returns:
        dict[str, Any]: The configuration for the specified file.

    Raises:
        ValueError: If the file key is not present in the configuration.
    """
    config_file_path = os.path.join("cfg", "files.yaml")
    with open(config_file_path, "r", encoding="utf-8") as file:
        config_files = yaml.safe_load(file)

    config_files = cast(dict[str, Any], config_files)

    if "files" not in config_files or file_name not in config_files["files"]:
        raise ValueError(
            f"File name '{file_name}' not found in files.yaml configuration."
        )

    # Default to CSV if file_type is not specified
    config_files["files"]["file_type"] = config_files["files"].get("file_type", "csv")

    return cast(dict[str, Any], config_files["files"][file_name])


def _read_project_env() -> dict[str, str]:
    """Read the project .env file and return key/value pairs.

    Returns:
        dict[str, str]: Key/value pairs from the .env file.

    Raises:
        EnvironmentError: If the .env file is not found.

    """
    env_path = os.path.join(".", ".env")
    if not os.path.exists(env_path):
        raise EnvironmentError(f".env file not found at {env_path}")
    env_values = dotenv_values(env_path)
    return {key: value for key, value in env_values.items() if value is not None}


def _load_private_key_der() -> bytes:
    """Load Snowflake private key from environment variable.

    Assumes user has a private key and has set relevant env variables for key path
    and passphrase.

    Returns:
        bytes: The private key in DER format.

    Raises:
        EnvironmentError: If the private key path environment variable is not set.
    """
    env_values = _read_project_env()
    private_key_path = env_values.get("SNOWFLAKE_PRIVATE_KEY_PATH")
    if not private_key_path:
        raise EnvironmentError(
            "Environment variable SNOWFLAKE_PRIVATE_KEY_PATH is not set."
        )
    passphrase = os.environ.get("SNOWFLAKE_PRIVATE_KEY_PASSPHRASE") or None

    with open(private_key_path, "rb") as f:
        p_key = serialization.load_pem_private_key(
            f.read(),
            password=passphrase.encode() if passphrase else None,
        )

    return p_key.private_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )


# may be unnecessary if read_data and write_data always use user/account params
def _connect_to_snowflake() -> snowflake.connector.SnowflakeConnection:
    """Establish a connection to Snowflake using environment variables.

    Returns:
        snowflake.connector.SnowflakeConnection: The Snowflake connection object.

    Raises:
        EnvironmentError: If required environment variables are not set.
        Exception: For connection errors.
    """
    try:
        private_key_der = _load_private_key_der()
        env_values = _read_project_env()
        conn = snowflake.connector.connect(
            # Username and account name stored in users' personal env variables
            user=env_values.get("SNOWFLAKE_USER"),
            account=env_values.get("SNOWFLAKE_ACCOUNT"),
            # Check project env variables for warehouse, role, database, schema
            warehouse=env_values.get("SNOWFLAKE_WAREHOUSE"),
            role=env_values.get("SNOWFLAKE_ROLE"),
            database=env_values.get("SNOWFLAKE_DATABASE"),
            schema=env_values.get("SNOWFLAKE_SCHEMA"),
            private_key=private_key_der,
        )
        if not conn:
            raise EnvironmentError(
                "Failed to establish Snowflake connection. Environment variables may be missing or incorrect."
            )
        return conn

    except Exception as exc:
        logger.error(f"Error connecting to Snowflake: {exc}")
        raise


# may be unnecessary if read_data and write_data always use user/account params
# if kept, update to check conn directly from read_data and write_data functions
def check_snowflake_connection() -> None:
    """Check Snowflake connection by executing a simple query.

    Raises:
        Exception: For connection errors.
    """
    try:
        conn = _connect_to_snowflake()
        cur = conn.cursor()
        cur.execute("SELECT CURRENT_TIMESTAMP()")
        result = cur.fetchone()
        logger.info(
            f"Snowflake connection test successful. Current timestamp: {result[0]}"
        )
        logger.info(
            f"Current User: {cur.execute('SELECT CURRENT_USER()').fetchone()[0]}"
        )
        logger.info(
            f"Current Account: {cur.execute('SELECT CURRENT_ACCOUNT()').fetchone()[0]}"
        )
        logger.info(
            f"Current Warehouse: {cur.execute('SELECT CURRENT_WAREHOUSE()').fetchone()[0]}"
        )
        logger.info(
            f"Current Database: {cur.execute('SELECT CURRENT_DATABASE()').fetchone()[0]}"
        )
        logger.info(
            f"Current Schema: {cur.execute('SELECT CURRENT_SCHEMA()').fetchone()[0]}"
        )
        logger.info(
            f"Current Role: {cur.execute('SELECT CURRENT_ROLE()').fetchone()[0]}"
        )
    except Exception as exc:
        logger.error(f"Snowflake connection test failed: {exc}")
        raise


if __name__ == "__main__":
    check_snowflake_connection()


def read_data(
    sql_query: str | None = None,
    table_name: str | None = None,
    schema_obj: str | None = None,
    user: str | None = _read_project_env().get("SNOWFLAKE_USER"),
    account: str | None = _read_project_env().get("SNOWFLAKE_ACCOUNT"),
    warehouse: str | None = _read_project_env().get("SNOWFLAKE_WAREHOUSE"),
    database: str | None = _read_project_env().get("SNOWFLAKE_DATABASE"),
    schema: str | None = _read_project_env().get("SNOWFLAKE_SCHEMA"),
) -> pd.DataFrame:
    """Read data from Snowflake using general purpose function.

    Args:
        sql_query: SQL query to execute.
        table_name: The target table name in Snowflake.
        schema_obj: The schema object for validation. Optional; if not provided, schema validation will be skipped.
        user: The Snowflake username. Optional; if not provided, the default from the project environment will be used.
        account: The Snowflake account name. Optional; if not provided, the default from the project environment will be used.
        warehouse: The Snowflake warehouse name. Optional; if not provided, the default from the project environment will be used.
        database: The Snowflake database name. Optional; if not provided, the default from the project environment will be used.
        schema: The Snowflake schema name. Optional; if not provided, the default from the project environment will be used.

    Returns:
        pandas.DataFrame: The loaded DataFrame.

    Raises:
        Exception: For IO or validation errors.
        ValueError: If stage is not one of 'raw', 'prepared', or 'output'.
    """
    if schema_obj is None:
        logger.warning(
            f"No schema found for '{schema_obj}'. Skipping schema validation."
        )

    # Connect to Snowflake using user defined parameters if provided
    try:
        conn = snowflake.connector.connect(
            user=user,
            account=account,
            warehouse=warehouse,
            database=database,
            schema=schema,
            private_key=_load_private_key_der(),
        )
    except Exception as exc:
        logger.error(f"Error connecting to Snowflake: {exc}")
        raise

    try:
        if sql_query:
            df = pd.read_sql(sql_query, conn)
        elif table_name:
            df = pd.read_sql(f"SELECT * FROM {table_name};", conn)
        else:
            raise ValueError("Either sql_query or table_name must be provided.")
        logger.info(f"Data successfully read from Snowflake. Shape: {df.shape}")

        if schema_obj:
            try:
                schemas[schema_obj].validate(df)
                logger.info("Data schema validation passed.")
            except pa.errors.SchemaError as exc:
                logger.error(f"Data schema validation failed: {exc}")
                raise
        return df
    except Exception as exc:
        logger.error(f"Error reading data from Snowflake: {exc}")
        raise


def write_data(
    df: pd.DataFrame,
    table_name: str,
    schema_obj: str | None,
    user: str | None = _read_project_env().get("SNOWFLAKE_USER"),
    account: str | None = _read_project_env().get("SNOWFLAKE_ACCOUNT"),
    warehouse: str | None = _read_project_env().get("SNOWFLAKE_WAREHOUSE"),
    database: str | None = _read_project_env().get("SNOWFLAKE_DATABASE"),
    schema: str | None = _read_project_env().get("SNOWFLAKE_SCHEMA"),
) -> None:
    """Write data to Snowflake using general purpose function.

    Args:
        df: The DataFrame to be written.
        table_name: The target table name in Snowflake.
        schema_obj: The schema object for validation. Optional; if not provided, schema validation will be skipped.
        user: The Snowflake username. Optional; if not provided, the default from the project environment will be used.
        account: The Snowflake account name. Optional; if not provided, the default from the project environment will be used.
        warehouse: The Snowflake warehouse name. Optional; if not provided, the default from the project environment will be used.
        database: The Snowflake database name. Optional; if not provided, the default from the project environment will be used.
        schema: The Snowflake schema name. Optional; if not provided, the default from the project environment will be used.

    Raises:
        Exception: For IO or validation errors.
        ValueError: If stage is not one of 'prepared' or 'output'.
    """
    if schema_obj is None:
        logger.warning(
            f"No schema found for '{schema_obj}'. Skipping schema validation."
        )

    try:
        conn = snowflake.connector.connect(
            user=user,
            account=account,
            warehouse=warehouse,
            database=database,
            schema=schema,
            private_key=_load_private_key_der(),
        )
    except Exception as exc:
        logger.error(f"Error connecting to Snowflake: {exc}")
        raise

    print(df.head())
    logger.info(f"Database to write to: {database}, Schema: {schema}")
    logger.info(f"Writing data to Snowflake table '{table_name}'.")

    try:
        success, _, nrows, _ = write_pandas(
            conn=conn,
            df=df,
            table_name=table_name,
            schema=schema,
            quote_identifiers=True,
            auto_create_table=True,
            overwrite=True,
        )

        # If validation fails, raise an error and abort write
        if schema_obj:
            try:
                schemas[schema_obj].validate(df)
                logger.info("Data schema validation passed.")
            except pa.errors.SchemaError as exc:
                logger.error(f"Data schema validation failed: {exc}")
                logger.info(
                    "Data write operation aborted due to schema validation failure."
                )
                conn.close()
                raise

        if success:
            logger.info(
                f"Data successfully written to Snowflake table '{table_name}'. Rows: {nrows}"
            )
            conn.close()
            logger.info("Snowflake connection closed.")
        else:
            logger.error(f"Failed to write data to Snowflake table '{table_name}'.")

    except Exception as exc:
        logger.error(f"Error writing data to Snowflake: {exc}")
        raise
