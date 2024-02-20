# Databricks notebook source
config = {}
#config["data_path"] = "s3://ds-databricks-playground/prarthana/ars/data_pq/"
config["data_path"] = "MOCK_DATA.csv"
# COMMAND ----------

# Default Instructor Model
#config["vector_store_path"] = "/dbfs//FileStore/tmp/ars_chromadb"
#config["data_path"] = "s3://ds-databricks-playground/prarthana/ars/data_pq/"

config["search_kwargs"] = {'k': 1000}
config["embedding_model_name"] = "sentence-transformers/all-MiniLM-L6-v2"
config["embedding_model_name_top"] = "bge-large-en-v1.5"
config["embedding_model_kwargs"] = {'device': 'cpu'}
config["embedding_model_encode_kwargs"] = {'normalize_embeddings': False}


