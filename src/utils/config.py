# Databricks notebook source
config = {}

config["search_kwargs"] = {'k': 1000}
config["embedding_model_name"] = "sentence-transformers/all-MiniLM-L6-v2"
config["embedding_save_path"] = "/tmp/embeddings/"
config["step"] = 500

# config["embedding_model_name_top"] = "bge-large-en-v1.5"
config["embedding_model_kwargs"] = {'device': 'cpu'}
config["embedding_model_encode_kwargs"] = {'normalize_embeddings': False}

