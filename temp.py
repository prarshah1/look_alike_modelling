import base64

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
from src.utils.functions import get_row_as_text, hf_embeddings, get_ars_vdb, get_ars_retrieved_df
from src.utils.config import config
from PIL import Image  # Import the Image class from the PIL module

uploaded = False
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

spark = SparkSession.builder.appName("customer_look_alike_modelling").getOrCreate()

generated_df = spark.read.option("header", "true").csv("/Users/pshah1/Downloads/audience_test.csv")
csv_data = generated_df.toPandas().to_csv(index=False)

st.download_button(
    "Press to Download",
    csv_data,
    f"temp_output.csv",
    "text/csv",
    key='download-csv'
)

