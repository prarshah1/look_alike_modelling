import chromadb
import pandas as pd
import sqlite3
from io import StringIO
from pyspark.sql import SparkSession
from streamlit_extras.switch_page_button import switch_page

__import__('pysqlite3')
import sys

sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
connection = sqlite3.connect('cache.db', timeout=100)
import os
import sys

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
import streamlit as st
from langchain.vectorstores import Chroma
from src.utils.functions import get_row_as_text, hf_embeddings, get_ars_retrieved_df, spark
from src.utils.config import config
from PIL import Image  # Import the Image class from the PIL module

st.set_page_config(page_title='Similance')
title_container = st.container()
col1, mid1, mid2, col2 = st.columns([0.25, 0.15, 0.4, 0.2])
image = Image.open('src/resources/ui_components/dataverze_logo.png')
with title_container:
    with mid1:
        st.image(image, width=100)
    with mid2:
        title = "Similance"
        st.title(title)

with st.container(border=True):
    st.markdown("""  
    ## :blue[Credit Card Renewals]  
    **Goal:** To retain credit card customers which might potentially churn away 
    
    **Story:** 
    The team at a bank is worried because many customers are quitting their credit card services. They want to figure out who might leave next so they can try to keep them happy. 
    Here's how we can help: We have a bunch of information about our customers and their credit card use. We also know who has already left us. By looking at all this data, we can find customers who are a lot like the ones who left before. 
    Then, we give each customer a score based on how much they're like the ones who left. The higher the score, the more likely they might leave, too. 
    Once we know who might leave, we can reach out to them with special offers or extra help to make them happy. It's like giving them a good reason to stick with us. 
    This way, we hope to keep more customers happy and stop them from leaving us. 
    """)

rows_to_convert_credit = 'CLIENTNUM,Attrition_Flag,Customer_Age,Gender,Dependent_count,Education_Level,Marital_Status,Income_Category,Card_Category,Months_on_book,Total_Relationship_Count,Months_Inactive_12_mon,Contacts_Count_12_mon,Credit_Limit,Total_Revolving_Bal,Avg_Open_To_Buy,Total_Amt_Chng_Q4_Q1,Total_Trans_Amt,Total_Trans_Ct,Total_Ct_Chng_Q4_Q1,Avg_Utilization_Ratio'.split(",")

if 'generated_df' not in st.session_state:
    st.session_state.generated_df = None

if 'supported_file_formats' not in st.session_state:
    st.session_state.supported_file_formats = ["txt", "json", "csv"]

if 'vdb_credit' not in st.session_state:
    # If not defined, define it
    db_dir = "src/resources/embeddings/credit"
    # client = chromadb.PersistentClient(path=db_dir)
    vdb_credit = Chroma(persist_directory=db_dir, embedding_function=hf_embeddings,
                 collection_metadata={"hnsw:space": "cosine"})
    st.session_state.vdb_credit = vdb_credit
    st.write(f"Original dataset size: {vdb_credit._collection.count()}")


# if "spark" not in st.session_state:
#     st.session_state.spark = SparkSession.builder.appName("customer_look_alike_modelling").getOrCreate()

def get_credit_retrieved_df(retriever, val_df, spark):
    input_rows = val_df.rdd.map(lambda x: x.row_as_text).collect()
    relevant_rows = []

    for i in range(0, len(input_rows)):
        print(input_rows[i])
        for relevant_row in retriever.get_relevant_documents(input_rows[i]):
            print(relevant_row.metadata)
            relevant_rows.append(
                relevant_row.page_content + f"; customer_id: {relevant_row.metadata['customer_id']}")

    converted_rows = [dict(pair.split(": ") for pair in row.split("; ")) for row in relevant_rows]
    generated_df = spark.createDataFrame(converted_rows).distinct()
    return generated_df


def generate_look_alike_credit(uploaded_file, k):
    if uploaded_file.name.split(".")[-1] in st.session_state.supported_file_formats:
        with st.spinner('Uploading...'):
            stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
            csv_file = StringIO(stringio.read())
            pandas_df = pd.read_csv(csv_file, header=0)[rows_to_convert_credit]
            st.markdown("""Uploaded Data""")
            st.write(pandas_df)
            spark = SparkSession.builder.appName("example").getOrCreate()
            input_df = spark.createDataFrame(pandas_df)
    else:
        raise Exception("File format {uploaded_file.name.split('.')[-1]} not supported")

    test_df = get_row_as_text(input_df, rows_to_convert_credit)

    retriever = st.session_state.vdb_credit.as_retriever(search_kwargs={"k": int(k)})
    generated_df = get_credit_retrieved_df(retriever, test_df, spark)
    generated_df.show()
    return generated_df


def credit_generate_form():
    succeeded = False
    st.markdown("""**Credit Card Churn Data**""")
    credit_data = pd.read_csv("src/resources/data/credit_card.csv")
    st.write(credit_data)
    st.markdown("""---""")
    st.markdown("""**Input Data**""")
    with st.form('fileform'):
        uploaded_file = st.file_uploader("Upload customer data", type=st.session_state.supported_file_formats)
        k = st.number_input('Number of rows required:', placeholder='Enter number odf rows to fetch per query:',
                            value=20)
        submitted = st.form_submit_button('Generate', disabled=(k == ""))
        if submitted:
            if uploaded_file is not None:
                if uploaded_file.name.split(".")[-1] in st.session_state.supported_file_formats:
                    try:
                        with st.spinner('Generating...'):
                            generated_df = generate_look_alike_credit(uploaded_file, k)
                            st.write("Generated look-alike audiences.")
                            st.write(generated_df)
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

credit_generate_form()
