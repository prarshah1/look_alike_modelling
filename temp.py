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
from langchain.vectorstores import Chroma
from pyspark.sql import SparkSession
from src.utils.functions import get_row_as_text, get_embedding_model

spark = SparkSession.builder.appName("customer_look_alike_modelling").getOrCreate()

db_dir = "/Users/pshah1/ps/projects/look_alike_modelling/src/resources/ars_embeddings/embeddings_03"

vdb = Chroma(persist_directory=db_dir, embedding_function=get_embedding_model(),)

import os
import pandas as pd
from sklearn.manifold import TSNE
import plotly.express as px
import ast  # Library for parsing the 'metadata' field


# Load the embeddings from the Parquet file
embeddings_file = os.path.join(db_dir, 'chroma-embeddings.parquet')
embeddings_df = pd.read_parquet(embeddings_file)

# Extract the embeddings and metadata from the DataFrame
embeddings_list = embeddings_df['embedding'].tolist()
embeddings_np = pd.DataFrame(embeddings_list).to_numpy()
metadata_list = embeddings_df['metadata'].apply(ast.literal_eval).tolist()

# Apply dimensionality reduction using t-SNE
tsne = TSNE(n_components=2, random_state=42)
embeddings_tsne = tsne.fit_transform(embeddings_np)

# Create a DataFrame with the reduced-dimensional embeddings and metadata
embeddings_tsne_df = pd.DataFrame(embeddings_tsne, columns=['Dimension 1', 'Dimension 2'])
embeddings_tsne_df['target'] = [metadata.get('target', 'Unknown') for metadata in metadata_list]

# Create a scatter plot using Plotly Express with color-coded labels
fig = px.scatter(
    embeddings_tsne_df,
    x='Dimension 1',
    y='Dimension 2',
    color='target',
    title='t-SNE Visualization of Embeddings with Metadata',
    labels={'target': 'target'}
)

# Show the plot
fig.show()
