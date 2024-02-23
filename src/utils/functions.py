#functions

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from pyspark.sql import functions as F, SparkSession
from src.utils.config import config
# Read the dataset



hf_embeddings = HuggingFaceEmbeddings(
    model_name=config["embedding_model_name"],
    model_kwargs=config["embedding_model_kwargs"],
    encode_kwargs=config["embedding_model_encode_kwargs"]
)

def get_embedding_model():
    return hf_embeddings

# Function to create retriever
def create_retriever(vec_df, db_dir, k, step=500):
    hf_embeddings = get_embedding_model()
    texts_list = vec_df.rdd.map(lambda x: x.row_as_text).collect()
    vdb = Chroma(persist_directory=db_dir, embedding_function=hf_embeddings, collection_metadata={"hnsw:space": "cosine"})
    for i in range(0, len(texts_list), step):
        texts = texts_list[i:i+step]
        metadata = [{} for r in texts]
        vdb.add_texts(texts, metadata)
    vdb.persist()
    return vdb.as_retriever(search_kwargs={"k": k})

# Function to get row as text
def get_row_as_text(input_df, columns):
    exprs = [F.concat(F.lit(str(col_name)), F.lit(': '), F.col(col_name).cast("string")) for col_name in columns]
    return input_df.withColumn("row_as_text", F.concat_ws("; ", *exprs))

# Function to get retrieved DataFrame
def get_retrieved_df(vec_df, retriever):
    input_rows = vec_df.rdd.map(lambda x: x.row_as_text).collect()
    relevant_rows = set()
    for i in range(0, len(input_rows), 500):
        for relevant_row in retriever.get_relevant_documents("\n".join(input_rows[i:(i+500)])):
            relevant_rows.add(relevant_row.page_content)
    relevant_rows = list(relevant_rows)
    converted_rows = [dict(pair.split(": ") for pair in row.split("; ")) for row in relevant_rows]
    return spark.createDataFrame(converted_rows).distinct()

def get_ars_retrieved_df(retriever, val_df, spark):
    input_rows = val_df.rdd.map(lambda x: x.row_as_text).collect()
    relevant_rows = []

    step = 5
    for i in range(0, len(input_rows)):
        for relevant_row in retriever.get_relevant_documents("\n".join(input_rows[i])):
            relevant_rows.append(
                relevant_row.page_content + f"; target: {relevant_row.metadata['target']}")

    converted_rows = [dict(pair.split(": ") for pair in row.split("; ")) for row in relevant_rows]
    return spark.createDataFrame(converted_rows).distinct()#.join(input_df.select("ID", "Segmentation"), how="left", on=["ID"])


from collections import Counter

def print_unique_values_and_counts(input_list):
    counter = Counter(input_list)

    for value, count in counter.items():
        print(f"{value}: {count} times")


def get_ars_vdb():
    db_dir = "/Users/pshah1/ps/projects/look_alike_modelling/src/resources/ars_embeddings/embeddings_05"
    vdb = Chroma(persist_directory=db_dir, embedding_function=hf_embeddings,
                 collection_metadata={"hnsw:space": "cosine"})
    return vdb