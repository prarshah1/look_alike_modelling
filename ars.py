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
from src.utils.functions import get_row_as_text, hf_embeddings, get_ars_retrieved_df, spark
from src.utils.config import config
from PIL import Image  # Import the Image class from the PIL module

st.set_page_config(page_title='Audience recommendation System')
title_container = st.container()
col1, mid, col2 = st.columns([0.4, 0.1, 0.5])
image = Image.open('src/resources/ui_components/audiance.png')
with title_container:
    with mid:
        st.image(image, width=100)
title = r'''
$\textsf{
    \Huge Audience Recommendation System
}$
'''
st.markdown(title)

rows_to_convert = 'mapped_contact_id_cont,gender_cont,management_level_cont,job_titles_cont,primary_cont,place_type_pl,state_pl,city_pl,contacts_count_pl,location_employee_count_pl,primary_sic_code_id_pl,location_sales_volume_pl,primary_naics_code_id_pl,b_abinumber,e_executivesourcecode,b_fulfillmentflag,b_countrycode,bus_abinumber_b2c,bus_contactid_b2c,cons_age_b2c,cons_maritalstatus_b2c'.split(",")

if 'generated_df' not in st.session_state:
    st.session_state.generated_df = None

if 'supported_file_formats' not in st.session_state:
    st.session_state.supported_file_formats = ["txt", "json", "csv"]

if 'vdb' not in st.session_state:
    # If not defined, define it
    db_dir = "/Users/pshah1/ps/projects/look_alike_modelling/src/resources/ars_embeddings/embeddings_all_4"
    client = chromadb.PersistentClient(path=db_dir)
    vdb = Chroma(client=client, embedding_function=hf_embeddings,
                 collection_metadata={"hnsw:space": "cosine"})
    st.session_state.vdb = vdb
    st.write(f"Original dataset size: {vdb._collection.count()}")

# if "spark" not in st.session_state:
#     st.session_state.spark = SparkSession.builder.appName("customer_look_alike_modelling").getOrCreate()


def generate_look_alike_audiences(uploaded_file, k):
    if uploaded_file.name.split(".")[-1] in st.session_state.supported_file_formats:
        with st.spinner('Uploading...'):
            stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
            csv_file = StringIO(stringio.read())
            pandas_df = pd.read_csv(csv_file, header=0)[rows_to_convert]
            st.write(pandas_df)
            spark = SparkSession.builder.appName("example").getOrCreate()
            input_df = spark.createDataFrame(pandas_df)
    else:
        raise Exception("File format {uploaded_file.name.split('.')[-1]} not supported")

    test_df = get_row_as_text(input_df, rows_to_convert)

    retriever = st.session_state.vdb.as_retriever(search_kwargs={"k": int(k)})
    generated_df = get_ars_retrieved_df(retriever, test_df, spark)
    generated_df.show()
    st.write(f"\n\nDataframe Count: {str(input_df.count())}")
    return generated_df


def ars_generate_form():
    succeeded = False
    with st.form('fileform'):
        uploaded_file = st.file_uploader("Upload audience data", type=st.session_state.supported_file_formats)
        k = st.text_input('Number of rows required:', placeholder='Enter number odf rows to fetch per query:',
                          value=20)
        submitted = st.form_submit_button('Generate', disabled=(k == ""))
        if submitted:
            if uploaded_file is not None:
                if uploaded_file.name.split(".")[-1] in st.session_state.supported_file_formats:
                    try:
                        with st.spinner('Generating...'):
                            generated_df = generate_look_alike_audiences(uploaded_file, k)
                            succeeded = True
                            st.write("Generated look-alike audiences.")
                        st.session_state.generated_df = generated_df
                    except AttributeError as e:
                        # Handling the AttributeError
                        st.write("Please submit the uploaded file.")
                        st.write(e)
                        # You can choose to perform alternative actions here if needed
                    except Exception as e:
                        # Handling any other exceptions
                        st.write(f"An unexpected error occurred: {e}")
                        raise e
                else:
                    st.write(f"Supported file types are {', '.join(st.session_state.supported_file_formats)}")
            else:
                st.write("Please select a file to upload first!")
    if succeeded:
        generated_df = st.session_state.generated_df.toPandas()
        st.write(generated_df)
        st.download_button(
            "Press to Download",
            generated_df.to_csv(index=False),
            f"{str(uploaded_file.name.split('.')[0])}_output.csv",
            "text/csv",
            key='download-csv'
        )

ars_generate_form()
