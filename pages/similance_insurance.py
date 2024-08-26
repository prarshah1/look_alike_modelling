import pandas as pd
import sqlite3
from pyspark.sql import SparkSession

#__import__('pysqlite3')
import sys

#sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
connection = sqlite3.connect('cache.db', timeout=100)
import os
import sys

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
import streamlit as st
from langchain_community.vectorstores import Chroma
from src.utils.functions import get_row_as_text, hf_embeddings
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
    ## :blue[Insurance Premiums Prediction]  
    **Goal:** To predict for Insurance claims or Premiums 
    
    **Story:** 
    Here's how we can use our customer data to manage insurance claims and premiums: 
    
    First, we gather information about our customers who have insurance and track if they've made any claims before. We also collect details about their physical features and previous claims history. 
    Next, we group our customers into different categories based on how much they've claimed in the past. This helps us understand which categories have claimed more. 
    
    Then, we look at our current customers and see which category they fit into. This helps us identify patterns and predict how much claims we might receive in a given period. 
    With this information, we can budget for the claims we expect to receive. This ensures we have enough money set aside to cover any future claims. 
    
    Additionally, we can use the same approach to budget for the premiums that our customers will pay. By understanding which categories they belong to, we can estimate how much premium income we'll receive in a given time period. 
    Overall, this method helps us manage our insurance business more effectively by predicting claims and premiums based on our customers' history and characteristics. 
    """)
rows_to_convert_insurance = 'age,sex,bmi,children,smoker,region'.split(",")

if 'insurance_output_df' not in st.session_state:
    st.session_state.insurance_output_df = None

if 'supported_file_formats' not in st.session_state:
    st.session_state.supported_file_formats = ["txt", "json", "csv"]

if 'vdb_insurance' not in st.session_state:
    # If not defined, define it
    db_dir = "src/resources/embeddings/insurance"
    # client = chromadb.PersistentClient(path=db_dir)
    vdb_insurance = Chroma(persist_directory=db_dir, embedding_function=hf_embeddings,
                 collection_metadata={"hnsw:space": "cosine"})
    # print("Embeddings", vdb_insurance.get())
    # print("Collections :", vdb_insurance._collection)
    # print("Collections :", vdb_insurance._collection.count())
    # print("doc", dir(vdb_insurance._collection))
    st.session_state.vdb_insurance = vdb_insurance

# collections = vdb_insurance._collection
# for collection in collections:
#     print(collection.name)
# print(dir(vdb_insurance))

# Function to check if embeddings are present
def check_embeddings_present(vdb):
    # Try to count the number of documents in the collection
    collection = vdb._collection
    # print(collection)
    num_documents = collection.count()
    print(num_documents)
    
    if num_documents > 0:
        return True, num_documents
    else:
        return False, 0

# Check embeddings in the collection
embeddings_present, num_documents = check_embeddings_present(st.session_state.vdb_insurance)

if embeddings_present:
    print(f"Embeddings are present in the collection. Number of documents: {num_documents}")
else:
    print("No embeddings found in the collection.")

def get_insurance_retrieved_df(retriever, val_df, spark):
    input_rows = val_df.rdd.map(lambda x: x.row_as_text).collect()
    # print("input rows:", input_rows)
    relevant_rows = []

    # print(dir(retriever))
    for i in range(0, len(input_rows)):
        for relevant_row in retriever.get_relevant_documents(input_rows[i]):
            relevant_rows.append(
                relevant_row.page_content + f"; Id: {relevant_row.metadata['Id']}; charges_bucket_label: {relevant_row.metadata['charges_bucket_label']}")
    # print("relevant rows : ",relevant_rows)
    converted_rows = [dict(pair.split(": ") for pair in row.split("; ")) for row in relevant_rows]
    # print("converted rows: ", converted_rows)
    generated_df = spark.createDataFrame(converted_rows).distinct()
    # return input_df.join(generated_df, how="inner", on=["infogroup_id", "mapped_contact_id_cont"])
    return generated_df


def generate_look_alike_insurance(pandas_df, k):
    st.write("""Input Data""")
    st.write(pandas_df)
    spark = SparkSession.builder.appName("example").getOrCreate()
    input_df = spark.createDataFrame(pandas_df)

    test_df = get_row_as_text(input_df, rows_to_convert_insurance)
    retriever = st.session_state.vdb_insurance.as_retriever(search_kwargs={"k": int(k)})
    # print("retriever", retriever)
    generated_df = get_insurance_retrieved_df(retriever, test_df, spark)
    generated_df.show()
    return generated_df


def insurance_generate_form():
    succeeded = False
    st.markdown("""**Insurance Data**""")
    insurance_data = pd.read_csv("src/resources/data/insurance.csv")
    st.write(insurance_data)
    st.markdown("""---""")
    st.markdown("""**Input Data**""")
    # insert selectbox - insert data, upload input data
    with st.form('fileform'):
        age = st.number_input('Age:', min_value=18, max_value=70, step=1)
        sex = st.selectbox('Sex:', ['male', 'female'])
        bmi = st.number_input('BMI:')
        children = st.number_input('Children:')
        smoker = st.selectbox('Smoker:', ['yes', 'no'])
        region = st.selectbox('Region:', ["northeast", "northwest", "southeast", "southwest"])
        k = st.number_input('Number of rows required:', placeholder='Enter number of rows to fetch per query:',
                            value=20)
        submitted = st.form_submit_button('Generate', disabled=(k == ""))
        if submitted:
            try:
                with st.spinner('Generating...'):
                    uploaded_row = {'age': str(age), 'sex': str(sex), 'bmi': str(bmi), 'children': str(children),
                                    'smoker': str(smoker), 'region': str(region)}
                    uploaded_df = pd.DataFrame([uploaded_row])
                    generated_df = generate_look_alike_insurance(uploaded_df, k)
                    st.write("Generated look-alike audiences.")
                    st.write(generated_df)
                st.session_state.insurance_output_df = generated_df
            except AttributeError as e:
                # Handling the AttributeError
                st.write("Please submit the uploaded file.")
                st.write(e)
                # You can choose to perform alternative actions here if needed
            except Exception as e:
                # Handling any other exceptions
                st.write(f"An unexpected error occurred: {e}")
                raise e


insurance_generate_form()
