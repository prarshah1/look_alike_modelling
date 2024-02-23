# Databricks notebook source
config = {}

config["embedding_model_name"] = "sentence-transformers/all-mpnet-base-v2"
config["embedding_save_path"] = "src/resources/ars_embeddings/embeddings_01"
config["step"] = 500

# config["embedding_model_name_top"] = "bge-large-en-v1.5"
config["embedding_model_kwargs"] = {'device': 'cpu'}
config["embedding_model_encode_kwargs"] = {'normalize_embeddings': False}
config["embedding_save_path"] = "src/resources/ars_embeddings/embeddings_01"
config["ars_id_cols"] = ["infogroup_id", "econtact_id"]