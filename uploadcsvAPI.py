from fastapi import FastAPI, UploadFile, File
from dotenv import load_dotenv
import uvicorn
import os
import csv
import re
import mysql.connector
import io
import pandas as pd
import requests
from typing import List
from fastapi.middleware.cors import CORSMiddleware
from langchain.chains import create_sql_query_chain
from sqlalchemy.orm import sessionmaker
from fastapi import FastAPI, HTTPException
from sqlalchemy import create_engine, inspect
from sqlalchemy.exc import SQLAlchemyError
from azure.storage.blob import BlobServiceClient
import logging
import os
import tempfile
from tempfile import NamedTemporaryFile
from sqlalchemy import create_engine, Table, Column, MetaData, ForeignKey, String
from sqlalchemy.dialects.mysql import VARCHAR
import chardet
import numpy as np
from pydantic import BaseModel
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain.agents import AgentExecutor
from langchain.agents.agent_types import AgentType
from langchain_community.utilities import SQLDatabase
from langchain_openai import AzureChatOpenAI
from sqlalchemy import inspect
load_dotenv()
# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Set up logging configuration
logging.basicConfig(level=logging.DEBUG)

# Global variable to store the db_name
db_name_global = None
# Global variables to store file info
uploaded_files_info = []
temp_file_paths = []
file_names = []
dataVizFile = None

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://cda-dws.azurewebsites.net"],  # Allow specific frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class CreateDBRequest(BaseModel):
    db_name: str


class TableInfo(BaseModel):  
    table_names: List[str]  
    common_column: str  


import subprocess
import logging
import threading


import os
import subprocess
import logging
import threading
import platform



@app.get("/")
async def index():
    return {"UploadWizard is Up and Running"}

@app.post("/create_db")
async def create_db(request: CreateDBRequest):
    global db_name_global
    db_name_global = request.db_name

    connection = mysql.connector.connect(host="cdaserver.mysql.database.azure.com", user="cdaadmin", password="Qwerty*1") 
    cursor = connection.cursor()

    cursor.execute("SHOW DATABASES")
    databases = cursor.fetchall()

    if (db_name_global,) in databases:
        cursor.close()
        connection.close()
        return {
            "status": f"Database {db_name_global} already exists. Using existing database."
        }
    else:
        try:
            cursor.execute(f"CREATE DATABASE {db_name_global}")
            cursor.close()
            connection.close()
            return {"status": f"Database {db_name_global} created successfully."}
        except mysql.connector.Error as err:
            cursor.close()
            connection.close()
            return {"error": f"Error creating database: {err}"}


@app.post("/upload_file_info")
async def upload_file_info(files: List[UploadFile] = File(...)):
    global uploaded_files_info
    global file_names
    global dataVizFile
    uploaded_files_info = []
    file_infos = []
    file_names = []
    for file in files:
        file_extension = os.path.splitext(file.filename)[-1].lower()
        file_content = await file.read()  # Read file content
        file_names.append(
            file.filename.split(".")[0]
        )  # Store file name without extension
        # Detect file encoding
        result = chardet.detect(file_content)
        encoding = result["encoding"]
        # Load data into DataFrame based on file extension
        if file_extension == ".csv":
            df = pd.read_csv(io.StringIO(file_content.decode(encoding)))
        elif file_extension == ".xlsx":
            df = pd.read_excel(io.BytesIO(file_content))
        else:
            return {
                "error": f"Invalid file format in {file.filename}. Please upload a CSV or XLSX file."
            }
        # Calculate file size
        file_size = len(file_content) / 1024 / 1024  # Size in MB
        # Store file info
        file_info = {
            "filename": file.filename,
            "total_rows": df.shape[0],
            "total_columns": df.shape[1],
            "file_size(MB)": file_size,
        }
        # Save the file content to a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=file_extension)
        temp_file.write(file_content)
        temp_file.close()
        dataVizFile = temp_file.name
        print(
            "-------------------------",
            dataVizFile,
            "------------------------------------",
        )
        logging.debug("File name : ", dataVizFile)
        # Append the file path and name to the global list
        uploaded_files_info.append(
            {"file_path": temp_file.name, "file_name": file.filename}
        )
        # Append the file info to the list
        file_infos.append(file_info)
        logging.debug(uploaded_files_info)
    # Log the lengths of the lists
    logging.debug(f"File names after upload_file_info: {file_names}")
    logging.debug(f"Temporary file paths after upload_file_info: {uploaded_files_info}")
    logging.debug(f"Length of file_names after upload_file_info: {len(file_names)}")
    logging.debug(
        f"Length of uploaded_files_info after upload_file_info: {len(uploaded_files_info)}"
    )

  

    return {
        "file_info": file_infos,
        "saved_files": uploaded_files_info,
        "file_names": file_names,
        "dataVizFile": dataVizFile,
    }


@app.post("/upload_and_clean")
async def upload_and_clean():
    global uploaded_files_info
    global temp_file_paths  # Reference the global variable
    sanitization_infos = []

    for file_info in uploaded_files_info:
        file_path = file_info["file_path"]
        file_name = file_info["file_name"]
        file_extension = os.path.splitext(file_path)[-1].lower()

        with open(file_path, "rb") as file:
            file_content = file.read()

        # Detect file encoding
        result = chardet.detect(file_content)
        encoding = result["encoding"]

        # Load the DataFrame based on file extension
        if file_extension == ".csv":
            df = pd.read_csv(io.StringIO(file_content.decode(encoding)))
        elif file_extension == ".xlsx":
            df = pd.read_excel(io.BytesIO(file_content))
        else:
            return {
                "error": f"Invalid file format in {file_name}. Please upload a CSV or XLSX file."
            }

        sanitization_info = {"filename": file_name}

        # Data cleaning
        sanitization_info["original_shape"] = df.shape

        # Convert column names to lower case and replace spaces with underscore
        df.columns = df.columns.str.lower().str.replace(" ", "_")
        sanitization_info["column_names_sanitized"] = True

        # Replace special characters in column names
        df.columns = df.columns.str.replace(r"\W", "", regex=True)
        sanitization_info["special_characters_removed_from_column_names"] = True

        # Strip leading/trailing whitespace from column names and values
        df.columns = df.columns.str.strip()
        df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
        sanitization_info["whitespace_removed"] = True

        # Convert date columns to standard date formats
        for col in df.select_dtypes(include=["object"]):
            try:
                df[col] = pd.to_datetime(df[col], errors="ignore")
            except Exception as e:
                pass
        sanitization_info["dates_standardized"] = True

        # Fill missing values with 'NA'
        initial_na_count = df.isna().sum().sum()
        df = df.fillna("NA")
        sanitization_info["missing_values_filled"] = int(initial_na_count)

        # Remove duplicate rows
        initial_duplicates_count = df.duplicated().sum()
        df = df.drop_duplicates()
        sanitization_info["duplicates_removed"] = int(initial_duplicates_count)

        # Save the cleaned DataFrame to a temporary file
        temp_file = NamedTemporaryFile(delete=False, suffix=file_extension)
        temp_file_path = temp_file.name
        if file_extension == ".csv":
            df.to_csv(temp_file_path, index=False)
        elif file_extension == ".xlsx":
            df.to_excel(temp_file_path, index=False)
        sanitization_info["temp_file_path"] = temp_file_path
        temp_file_paths.append(temp_file_path)
        logging.debug("-----------------------------------------------")
        logging.debug(temp_file_paths)
        sanitization_infos.append(sanitization_info)

    # Log the lengths of the lists
    logging.debug(f"Temporary file paths after cleaning: {temp_file_paths}")
    logging.debug(f"Length of temp_file_paths after cleaning: {len(temp_file_paths)}")
    logging.debug(f"Length of file_names after cleaning: {len(file_names)}")

    # Clear uploaded_files_info after processing
    uploaded_files_info.clear()

    return {
        "status": "Data cleaned and saved",
        "sanitization_infos": sanitization_infos,
    }


def infer_primary_key(df):
    for column in df.columns:
        if df[column].is_unique:
            return column
    return None

def validate_table_name(name):
    # Replace invalid characters and ensure name is suitable for SQL
    return name.lower().replace(" ", "_").replace("(", "").replace(")", "")

@app.post("/create_tables_with_relationships")
async def create_tables_with_relationships():
    global temp_file_paths
    global db_name_global
    global file_names

    if file_names is None:
        return {"error": "file_names is not initialized."}

    # Ensure temp_file_paths is clean before processing
    if len(temp_file_paths) != len(file_names):
        # Clean up old temporary files
        for temp_file_path in temp_file_paths:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
        temp_file_paths.clear()

        return {"error": "Mismatch between number of temporary files and file names."}

    global engine
    engine = create_engine(
        f"mysql+mysqlconnector://cdaadmin:Qwerty*1@cdaserver.mysql.database.azure.com:3306/{db_name_global}"
    )
    metadata = MetaData()

    messages = []

    for temp_file_path, file_name in zip(temp_file_paths, file_names):
        file_extension = os.path.splitext(temp_file_path)[-1].lower()

        if file_extension == ".csv":
            df = pd.read_csv(temp_file_path)
        elif file_extension == ".xlsx":
            df = pd.read_excel(temp_file_path)
        else:
            return {
                "error": f"Invalid file format in {temp_file_path}. Please upload a CSV or XLSX file."
            }

        table_name = validate_table_name(file_name)

        primary_key_column = infer_primary_key(df)
        if not primary_key_column:
            messages.append(
                {
                    "file": temp_file_path,
                    "status": f"No unique column found for primary key in {file_name}",
                }
            )
            continue

        columns = []
        for col in df.columns:
            max_length = df[col].astype(str).map(len).max()
            if max_length > 65535:
                df[col] = df[col].apply(lambda x: x[:65535] if len(str(x)) > 65535 else x)
                max_length = 65535
            columns.append(Column(col, String(max(max_length, 255))))  # Adjust column size

        for column in columns:
            if column.name == primary_key_column:
                column.primary_key = True
        table = Table(table_name, metadata, *columns)

        messages.append(
            {
                "file": temp_file_path,
                "status": f"Primary key {primary_key_column} identified for table {table_name}",
            }
        )

    metadata.create_all(engine)

    for temp_file_path, file_name in zip(temp_file_paths, file_names):
        file_extension = os.path.splitext(temp_file_path)[-1].lower()
        table_name = validate_table_name(file_name)

        if file_extension == ".csv":
            df = pd.read_csv(temp_file_path)
        elif file_extension == ".xlsx":
            df = pd.read_excel(temp_file_path)

        df = df.astype(str)  # Convert all columns to strings

        primary_key_column = infer_primary_key(df)  # Ensure primary_key_column is valid
        if not primary_key_column:
            messages.append(
                {
                    "file": temp_file_path,
                    "status": f"No unique column found for primary key in {file_name}",
                }
            )
            continue

        with engine.connect() as conn:
            existing_keys = pd.read_sql(f"SELECT {primary_key_column} FROM {table_name}", conn)
            new_data = df[~df[primary_key_column].isin(existing_keys[primary_key_column])]

            if not new_data.empty:
                try:
                    new_data.to_sql(table_name, con=engine, if_exists="append", index=False)
                    messages.append(
                        {"file": temp_file_path, "status": f"Data inserted into table {table_name}"}
                    )
                except Exception as e:
                    return {"error": f"Failed to insert data into table {table_name}: {e}"}
            else:
                messages.append(
                    {"file": temp_file_path, "status": f"No new data to insert into table {table_name}"}
                )

        os.remove(temp_file_path)
        messages.append({"file": temp_file_path, "status": "Temporary file deleted"})

    temp_file_paths.clear()
    file_names.clear()

    return {"status": "Data processing completed", "details": messages}




# Azure OpenAI setup
llm = AzureChatOpenAI(
    deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
    temperature=0,
    api_version=os.getenv("OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
)



# Base URL and SAS token
base_url = "https://cdaapp.blob.core.windows.net"
sas_token = "sp=racwdl&st=2024-07-28T13:19:42Z&se=2025-12-31T21:19:42Z&sv=2022-11-02&sr=c&sig=HwR4BySi15Tyc3Te3yu31%2BcnUoc%2BhJd0O3c7vS1Lrwo%3D"
container_name = "cdafiles"  # Your container name

@app.post("/create_view")
async def create_view():
    global engine
    engine = create_engine(
        f"mysql+mysqlconnector://cdaadmin:Qwerty*1@cdaserver.mysql.database.azure.com:3306/{db_name_global}"
    )
    inspector = inspect(engine)
    table_names = inspector.get_table_names()

    if len(table_names) == 0:
        return {"status": "No tables found in the database."}

    # Handle the case where there's only one table
    if len(table_names) == 1:
        table_name = table_names[0]
        sql_query = f"SELECT * FROM {table_name}"
        logger.debug(f"Single table SQL Query: {sql_query}")

        # Execute the SQL query and save the data
        Session = sessionmaker(bind=engine)
        session = Session()
        try:
            df = pd.read_sql_query(sql_query, engine)
            output_response = df.to_dict(orient="records")
            logger.debug(f"Query results: {output_response}")

            # Save the DataFrame to a CSV file with the database name
            output_filename = f"{db_name_global}.csv"
            df.to_csv(output_filename, index=False)
            logger.debug(f"Data saved to {output_filename}")

            # Upload the CSV file to Azure Blob Storage using SAS URL and Token
            try:
                blob_service_client = BlobServiceClient(account_url=base_url, credential=sas_token)
                blob_client = blob_service_client.get_blob_client(container=container_name, blob=output_filename)

                with open(output_filename, "rb") as data:
                    blob_client.upload_blob(data, overwrite=True)

                logger.debug("Data uploaded to Azure Blob Storage")
            except Exception as e:
                logger.error(f"Error uploading to Azure Blob Storage: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Error uploading to Azure Blob Storage: {str(e)}")

            return {"result": output_response, "sql_query": sql_query}
        except SQLAlchemyError as e:
            logger.error(f"SQLAlchemy error during query execution: {str(e)}")
            raise HTTPException(status_code=500, detail=f"SQLAlchemy error during query execution: {str(e)}")
        finally:
            session.close()
    else:
        # Handle the case where there are multiple tables (your existing logic)
        db = SQLDatabase(engine)
        chain = create_sql_query_chain(llm, db)
        sql_query_raw = chain.invoke({"question": "combine the tables present in the database"})
        logger.debug(f"Raw SQL Query: {sql_query_raw}")

        # Extract the actual SQL query if there is additional text
        sql_query_match = re.search(r"```sql\n(.*?)\n```", sql_query_raw, re.DOTALL)
        if sql_query_match:
            sql_query = sql_query_match.group(1).strip()
        else:
            logger.error("Failed to extract SQL query from response")
            raise HTTPException(status_code=500, detail="Failed to extract SQL query from response")

        logger.debug(f"Cleaned SQL Query: {sql_query}")

        # Execute the SQL query
        Session = sessionmaker(bind=engine)
        session = Session()
        try:
            df = pd.read_sql_query(sql_query, engine)
            output_response = df.to_dict(orient="records")
            logger.debug(f"Query results: {output_response}")

            # Save the DataFrame to a CSV file with the database name
            output_filename = f"{db_name_global.lower()}.csv"
            df.to_csv(output_filename, index=False)
            logger.debug(f"Data saved to {output_filename}")

            # Upload the CSV file to Azure Blob Storage using SAS URL and Token
            try:
                blob_service_client = BlobServiceClient(account_url=base_url, credential=sas_token)
                blob_client = blob_service_client.get_blob_client(container=container_name, blob=output_filename)

                with open(output_filename, "rb") as data:
                    blob_client.upload_blob(data, overwrite=True)

                logger.debug("Data uploaded to Azure Blob Storage")
            except Exception as e:
                logger.error(f"Error uploading to Azure Blob Storage: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Error uploading to Azure Blob Storage: {str(e)}")

            # Store the DataFrame into the 'combined_table'
            df.to_sql('combined_table', con=engine, if_exists='replace', index=False)
            logger.debug("Data stored in the 'combined_table'")
        except SQLAlchemyError as e:
            logger.error(f"SQLAlchemy error during query execution: {str(e)}")
            raise HTTPException(status_code=500, detail=f"SQLAlchemy error during query execution: {str(e)}")
        finally:
            session.close()

        return {"result": output_response, "sql_query": sql_query}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
