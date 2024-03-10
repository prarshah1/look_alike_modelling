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
    ## :blue[Insurance budgeting]  
    **Goal:** To budget for Insurance claims or Premiums 
    
    **Story:** 
    Here's how we can use our customer data to manage insurance claims and premiums: 
    
    First, we gather information about our customers who have insurance and track if they've made any claims before. We also collect details about their physical features and previous claims history. 
    Next, we group our customers into different categories based on how much they've claimed in the past. This helps us understand which categories have claimed more. 
    
    Then, we look at our current customers and see which category they fit into. This helps us identify patterns and predict how much claims we might receive in a given period. 
    With this information, we can budget for the claims we expect to receive. This ensures we have enough money set aside to cover any future claims. 
    
    Additionally, we can use the same approach to budget for the premiums that our customers will pay. By understanding which categories they belong to, we can estimate how much premium income we'll receive in a given time period. 
    Overall, this method helps us manage our insurance business more effectively by predicting claims and premiums based on our customers' history and characteristics. 
    """)
rows_to_convert = 'mapped_contact_id_cont,gender_cont,management_level_cont,job_titles_cont,primary_cont,place_type_pl,state_pl,city_pl,contacts_count_pl,location_employee_count_pl,primary_sic_code_id_pl,location_sales_volume_pl,primary_naics_code_id_pl,b_abinumber,e_executivesourcecode,b_fulfillmentflag,b_countrycode,bus_abinumber_b2c,bus_contactid_b2c,cons_age_b2c,cons_maritalstatus_b2c'.split(",")

if 'generated_df' not in st.session_state:
    st.session_state.generated_df = None

if 'supported_file_formats' not in st.session_state:
    st.session_state.supported_file_formats = ["txt", "json", "csv"]

if 'vdb' not in st.session_state:
    # If not defined, define it
    db_dir = "src/resources/embeddings/insurance_emb/index"
    # client = chromadb.PersistentClient(path=db_dir)
    vdb = Chroma(persist_directory=db_dir, embedding_function=hf_embeddings,
                 collection_metadata={"hnsw:space": "cosine"})
    st.session_state.vdb = vdb
    st.write(f"Original dataset size: {vdb._collection.count()}")

# if "spark" not in st.session_state:
#     st.session_state.spark = SparkSession.builder.appName("customer_look_alike_modelling").getOrCreate()


def generate_look_alike_insurance(pandas_df, k):
    st.write(pandas_df)
    spark = SparkSession.builder.appName("example").getOrCreate()
    input_df = spark.createDataFrame(pandas_df)

    test_df = get_row_as_text(input_df, rows_to_convert)
    retriever = st.session_state.vdb.as_retriever(search_kwargs={"k": int(k)})
    generated_df = get_ars_retrieved_df(retriever, test_df, spark)
    generated_df.show()
    st.write(f"\n\nDataframe Count: {str(input_df.count())}")
    return generated_df


def insurance_generate_form():
    succeeded = False
    st.markdown("""**Insurance Data**""")
    insurance_data = pd.read_csv("src/resources/data/insurance.csv")
    st.write(insurance_data)
    st.markdown("""---""")
    st.markdown("""**Input Data**""")
    with st.form('fileform'):
        uploaded_file = None
        age = st.number_input('Age:',min_value=18, max_value=70,step=1)
        sex = st.selectbox('Sex:', ['male', 'female'])
        bmi = st.number_input('BMI:')
        children = st.number_input('Children:')
        smoker = st.selectbox('Smoker:', ['yes', 'no'])
        region = st.selectbox('Region:', ["northeast", "northwest", "southeast", "southwest"])
        k = st.number_input('Number of rows required:', placeholder='Enter number odf rows to fetch per query:', value=20)
        submitted = st.form_submit_button('Generate', disabled=(k == ""))
        if submitted:
            try:
                with st.spinner('Generating...'):
                    uploaded_row = {'age': str(age), 'sex': str(sex), 'bmi': str(bmi), 'children': str(children), 'smoker': str(smoker), 'region': str(region)}
                    uploaded_df = pd.DataFrame([uploaded_row])
                    generated_df = generate_look_alike_insurance(uploaded_df, k)
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

insurance_generate_form()
