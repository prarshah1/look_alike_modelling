import chromadb
import pandas as pd
import sqlite3
from io import StringIO
from pyspark.sql import SparkSession

__import__('pysqlite3')
import sys

sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
connection = sqlite3.connect('cache.db', timeout=100)
import os
import streamlit as st
from langchain.vectorstores import Chroma
from src.utils.functions import get_row_as_text, hf_embeddings, get_retrieved_df, spark
from src.utils.config import config
from PIL import Image  # Import the Image class from the PIL module

st.set_page_config(page_title='Audience recommendation System')
title_container = st.container()
start, col1, col2 = st.columns([0.3, 0.22, 0.9])
image = Image.open('src/resources/ui_components/dataverze_logo.png')
with title_container:
    with col1:
        st.image(image, width=100)
    with col2:
        st.title("Similence")

if 'current_filename' not in st.session_state:
    st.session_state.current_filename = None

if 'embeddings' not in st.session_state:
    st.session_state.embeddings = []

if "spark" not in st.session_state:
    st.session_state.spark = SparkSession.builder.appName("customer_look_alike_modelling").getOrCreate()

supported_file_formats = ["txt", "csv"]


def create_embeddings(uploaded_file, k):
    file_name = uploaded_file.name.split(".")[0]
    embedding_save_path = config["embedding_save_path"] + file_name
    file_extension = uploaded_file.name.split(".")[-1]
    if file_extension in supported_file_formats:
        stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
        st.write("Reading file")
        csv_file = StringIO(stringio.read())
        pandas_df = pd.read_csv(csv_file, header=0)
        input_df = st.session_state.spark.createDataFrame(pandas_df)
    else:
        raise Exception(f"File format {uploaded_file.name.split('.')[-1]} not supported")

    id_col = input_df.columns[0]
    # label_col = input_df.columns[-1]
    rows_to_convert = input_df.columns
    # rows_to_convert.remove(label_col)
    rows_to_convert.remove(id_col)

    train_df = get_row_as_text(input_df, rows_to_convert)
    st.write(f"Columns: {str(rows_to_convert)}, ID Column: {str(id_col)}")

    texts_list = train_df.rdd.collect()

    os.makedirs(embedding_save_path, exist_ok=True)
    client = chromadb.PersistentClient(path=embedding_save_path)
    vdb = Chroma(persist_directory=embedding_save_path, client=client, embedding_function=hf_embeddings, collection_metadata={"hnsw:space": "cosine"})

    for i in range(0, len(texts_list), config["step"]):
        st.write("Creating embeddings ...")
        texts = [x.row_as_text for x in texts_list[i:i + config["step"]]]
        metadata = [{id_col: str(eval(f"x.{id_col}"))} for x in texts_list[i:i + config["step"]]]
        vdb.add_texts(texts, metadata)
    vdb.persist()

    k = max(2000, int(vdb._collection.count() * 0.1))
    st.write(f"K set to {k}")
    retriever = vdb.as_retriever(search_kwargs={"k": k})
    st.write(f"\n\nNumber of embeddings in chromadb: {str(vdb._collection.count())}")
    st.write(f"\n\nDataframe Count: {str(input_df.count())}")
    st.session_state.embeddings.append(file_name)

    st.write("embeddings:")
    st.write(str(st.session_state.embeddings))
    return retriever, file_name

def get_embeddings(master_file_name, k, uploaded_seed_dataset):
    if master_file_name in st.session_state.embeddings:
        embedding_save_path = config["embedding_save_path"] + master_file_name
        client = chromadb.PersistentClient(path=embedding_save_path)
        vdb = Chroma(persist_directory=embedding_save_path, client=client, embedding_function=hf_embeddings, collection_metadata={"hnsw:space": "cosine"})
        k = max(k, int(vdb._collection.count() * 0.1))
        st.write(f"Setting k to {k}")
        retriever = vdb.as_retriever(search_kwargs={"k": k})
    else:
        st.exception("Embeddings not present at path!")

    generated_df =



def file_upload_form():
    with st.form('fileform'):
        uploaded_file = st.file_uploader("Upload a master file", type=supported_file_formats)
        st.write("Note:   First column will be considered as ID column")
        submitted = st.form_submit_button('Upload', disabled=(uploaded_file == ""))
        if submitted:
            if uploaded_file is not None:
                if uploaded_file.name.split(".")[-1] in supported_file_formats:
                    try:
                        with st.spinner("Uploading file and generating embeddings..."):
                            create_embeddings(uploaded_file, 0)
                    except AttributeError:
                        # Handling the AttributeError
                        st.write("Please submit the uploaded file.")
                        # You can choose to perform alternative actions here if needed
                    except Exception as e:
                        # Handling any other exceptions
                        st.write(f"An unexpected error occurred: {e}")
                        raise e
                else:
                    st.write(f"Supported file types are {', '.join(supported_file_formats)}")
            else:
                st.write("Please select a file to upload first!")


def select_exsisting_embeddings():
    with st.form('look_alike_data_generation_form'):
        master_file_name = st.selectbox('Choose from uploaded master files:', st.session_state.embeddings)
        uploaded_file = st.file_uploader("Upload seed dataset")
        k = st.text_input('Number of rows required:', placeholder='Enter number of rows required', value=20)
        submitted = st.form_submit_button('Submit', disabled=(master_file_name == ""))
        if submitted:
            if uploaded_file is not None:
                if uploaded_file.name.split(".")[-1] in supported_file_formats:
                    try:
                        with st.spinner("Uploading file and inferencing ..."):
                            file_name, retriever = get_or_create_embeddings(uploaded_file, k)

                            get_retrieved_df()
                    except AttributeError:
                        # Handling the AttributeError
                        st.write("Please submit the uploaded file.")
                        # You can choose to perform alternative actions here if needed
                    except Exception as e:
                        # Handling any other exceptions
                        st.write(f"An unexpected error occurred: {e}")
                        raise e
                else:
                    st.write(f"Supported file types are {', '.join(supported_file_formats)}")
            else:
                st.write("Please select a file to upload first!")


# Display the selected tab's form
upload_tab, query_tab = st.tabs(['Upload file', 'Generate look-alike audiences'])
with upload_tab:
    file_upload_form()
with query_tab:
    select_exsisting_embeddings()
