import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import split
import sqlite3
from io import StringIO
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType

# from pdfminer.layout import LTTextContainer
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
connection = sqlite3.connect('cache.db', timeout=100)
import os
import streamlit as st
from langchain.vectorstores import Chroma
from src.utils.functions import get_row_as_text, hf_embeddings
from src.utils.config import config

st.set_page_config(page_title='Look-alike Modelling using embeddings')
st.title('Look-alike Modelling using embeddings')

if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None

if 'current_filename' not in st.session_state:
    st.session_state.current_filename = None

if 'embeddings' not in st.session_state:
        st.session_state.embeddings = []

if 'retriever' not in locals():
        # If not defined, define it
        retriever = None

if "spark" not in st.session_state:
    st.session_state.spark = SparkSession.builder.appName("customer_look_alike_modelling").getOrCreate()


def get_or_create_embeddings(uploaded_file):

    if isinstance(uploaded_file, str):
        if uploaded_file in st.session_state.embeddings:
            embedding_save_path = config["embedding_save_path"] + uploaded_file
            vdb = Chroma(persist_directory=embedding_save_path, embedding_function=hf_embeddings, collection_metadata={"hnsw:space": "cosine"})
            k = min(2000, int(vdb._collection.count() * 0.1))
            file_name = uploaded_file
            retriever = vdb.as_retriever(search_kwargs={"k": k})
    else:
        file_name = uploaded_file.name.split(".")[0]
        embedding_save_path = config["embedding_save_path"] + file_name
        file_extension = uploaded_file.name.split(".")[-1]
        if (file_extension in ["txt", "json", "csv"]):
            stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
            st.write("reading file")
            csv_file = StringIO(stringio.read())
            pandas_df = pd.read_csv(csv_file, header=0)
            spark = SparkSession.builder.appName("example").getOrCreate()
            input_df = spark.createDataFrame(pandas_df)
        else:
            raise Exception(f"File format {uploaded_file.name.split('.')[-1]} not supported")

        id_col = input_df.columns[0]
        label_col = input_df.columns[-1]
        rows_to_convert = input_df.columns
        rows_to_convert.remove(label_col)
        rows_to_convert.remove(id_col)

        st.write("row to text")
        train_df = get_row_as_text(input_df, rows_to_convert)
        st.write(f"Columns: {str(input_df.columns)}, Label Column: {str(label_col)}")

        texts_list = train_df.rdd.collect()

        os.makedirs(embedding_save_path, exist_ok=True)
        vdb = Chroma(persist_directory=embedding_save_path, embedding_function=hf_embeddings, collection_metadata={"hnsw:space": "cosine"})

        for i in range(0, len(texts_list), config["step"]):
            st.write("Creating embeddings ...")
            texts = [x.row_as_text for x in texts_list[i:i + config["step"]]]
            metadata = [{label_col: str(eval(f"x.{label_col}")), id_col: str(eval(f"x.{id_col}"))} for
                        x in texts_list[i:i + config["step"]]]
            vdb.add_texts(texts, metadata)
        vdb.persist()

        k = min(2000, int(vdb._collection.count()*0.1))
        retriever = vdb.as_retriever(search_kwargs={"k": k})
        st.write(f"\n\nNumber of embeddings in chromadb: {str(vdb._collection.count())}")
        st.write(f"\n\nDataframe Count: {str(input_df.count())}")
        st.session_state.embeddings.append(file_name)

        st.write("embeddings:")
        st.write(str(st.session_state.embeddings))
    return retriever, file_name

def generate_response(uploaded_file):
    # Trying to access the 'LLMDATA' attribute in the 'session_state' object
    retriever, file_name = get_or_create_embeddings(uploaded_file)
    st.session_state.current_filename = file_name
    # result = get_look_alike_data(retriver)
    st.write("Got retriever********, this is output")
    st.write("Success....")

def file_upload_form():
    with st.form('fileform'):
        supported_file_types = ["csv", "txt", "json"]
        uploaded_file = st.file_uploader("Upload a file", type=(supported_file_types))
        k = st.text_input('Number of rows required:', placeholder='Enter number of rows required', value=2000)
        submitted = st.form_submit_button('Submit')
        if submitted:
            if uploaded_file is not None:
                if uploaded_file.name.split(".")[-1] in supported_file_types:
                    try:
                        generate_response(uploaded_file)
                    except AttributeError:
                        # Handling the AttributeError
                        st.write("Please submit the uploaded file.")
                        # You can choose to perform alternative actions here if needed
                    except Exception as e:
                        # Handling any other exceptions
                        st.write(f"An unexpected error occurred: {e}")
                        raise e
                else:
                    st.write(f"Supported file types are {', '.join(supported_file_types)}")
            else:
                st.write("Please select a file to upload first!")


def select_exsisting_embeddings():
    global embeddings
    with st.form('look_alike_data_generation_form'):
        file_name = st.selectbox('Choose from previously uploaded files:', st.session_state.embeddings)
        k = st.text_input('Number of rows required:', placeholder='Enter number of rows required', value=2000)
        submitted = st.form_submit_button('Submit', disabled=(file_name == ""))
        if submitted:
            with st.spinner('Generating...'):
                generate_response(file_name)

# Radio button to select between Form 1 and Form 2
selected_form = st.radio('Select a form:', ['Upload file', 'Select existing embeddings'])

# Display the selected form
if selected_form == 'Upload file':
    file_upload_form()
elif selected_form == 'Select existing embeddings':
    select_exsisting_embeddings()

