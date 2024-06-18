# functions
import json
from datetime import datetime

from sentence_transformers import CrossEncoder

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from pyspark.sql import functions as F, SparkSession
from src.utils.config import config

# Read the dataset


hf_embeddings = HuggingFaceEmbeddings(
    model_name=config["embedding_model_name"],
    model_kwargs=config["embedding_model_kwargs"],
    encode_kwargs=config["embedding_model_encode_kwargs"]
)
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

spark = SparkSession.builder.appName("customer_look_alike_modelling").getOrCreate()
# data_path = "/Users/pshah1/Downloads/look_alike/aes_data"
# input_df = spark.read.option("header", "true").parquet(data_path).filter(F.col("job_titles_cont").isNotNull())


def get_embedding_model():
    return hf_embeddings


# Function to create retriever
def create_retriever(vec_df, db_dir, k, step=500):
    hf_embeddings = get_embedding_model()
    texts_list = vec_df.rdd.map(lambda x: x.row_as_text).collect()
    vdb = Chroma(persist_directory=db_dir, embedding_function=hf_embeddings,
                 collection_metadata={"hnsw:space": "cosine"})
    for i in range(0, len(texts_list), step):
        texts = texts_list[i:i + step]
        metadata = [{} for r in texts]
        vdb.add_texts(texts, metadata)
    vdb.persist()
    return vdb.as_retriever(search_kwargs={"k": k})


# Function to get row as text
def get_row_as_text(input_df, columns):
    exprs = [F.concat(F.lit(str(col_name)), F.lit(': '), F.col(col_name).cast("string")) for col_name in columns]
    return input_df.withColumn("row_as_text", F.concat_ws("; ", *exprs))


# Function to get retrieved DataFrame
def get_retrieved_df(vec_df, retriever, k):
    input_rows = vec_df.rdd.map(lambda x: x.row_as_text).collect()
    relevant_rows = []
    union_df = None
    for i in range(0, len(input_rows), 500):
        for input_row in input_rows:
            for relevant_row in retriever.get_relevant_documents(input_row):
                score = cross_encoder.predict([input_row, relevant_row.page_content])
                # score represents the relevance of the input_row and the retrieved row.
                relevant_rows.append(relevant_row.page_content + f"; Score: {score}")

            converted_rows = [dict(pair.split(": ") for pair in row.split("; ")) for row in relevant_rows]
            df = spark.createDataFrame(converted_rows).distinct().orderBy(F.col("Score").desc()).limit(
                max(5, int(k * 0.1)))
            if union_df is None:
                union_df = df
            else:
                union_df = union_df.union(df)
    return union_df.distinct()


def get_ars_retrieved_df(retriever, val_df, spark):
    input_rows = val_df.rdd.map(lambda x: x.row_as_text).collect()
    relevant_rows = []

    for i in range(0, len(input_rows)):
        print(input_rows[i])
        for relevant_row in retriever.get_relevant_documents(input_rows[i]):
            relevant_rows.append(
                relevant_row.page_content + f"; infogroup_id: {relevant_row.metadata['infogroup_id']}; mapped_contact_id_cont: {relevant_row.metadata['mapped_contact_id_cont']}")

    converted_rows = [dict(pair.split(": ") for pair in row.split("; ")) for row in relevant_rows]
    generated_df = spark.createDataFrame(converted_rows).distinct()
    generated_df.select("infogroup_id", "mapped_contact_id_cont").show()
    # return input_df.join(generated_df, how="inner", on=["infogroup_id", "mapped_contact_id_cont"])
    return generated_df


from collections import Counter


def print_unique_values_and_counts(input_list):
    counter = Counter(input_list)

    for value, count in counter.items():
        print(f"{value}: {count} times")

def save_chart(query):
    q_s = ' If any charts or graphs or plots were created save them locally.'
    query += ' . '+ q_s
    return query


def check_plot_query(query):
    keywords = ['chart', 'charts', 'graph', 'graphs', 'plot', 'plt', 'plots']
    for keyword in keywords:
        if keyword in query.lower():
            return True
    return False
def run_query(agent, query_):
    if check_plot_query(query_.lower()):
        query_ = save_chart(query_)
    output = agent(query_)
    response, intermediate_steps = output['output'], output['intermediate_steps']
    # thought, action, action_input, observation, steps = decode_intermediate_steps(intermediate_steps)
    # store_convo(query_, steps, response)
    return response, intermediate_steps

def decode_intermediate_steps(steps):
    log, thought_, action_, action_input_, observation_ = [], [], [], [], []
    text = ''
    for step in steps:
        thought_.append(':green[{}]'.format(step[0][2].split('Action:')[0]))
        action_.append(':green[Action:] {}'.format(step[0][2].split('Action:')[1].split('Action Input:')[0]))
        action_input_.append(':green[Action Input:] {}'.format(step[0][2].split('Action:')[1].split('Action Input:')[1]))
        observation_.append(':green[Observation:] {}'.format(step[1]))
        log.append(step[0][2])
        text = step[0][2]+' Observation: {}'.format(step[1])
    return thought_, action_, action_input_, observation_, text


def get_convo():
    convo_file = 'convo_history.json'
    with open(convo_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data, convo_file


def store_convo(query, response_, response):
    data, convo_file = get_convo()
    current_dateTime = datetime.now()
    data['{}'.format(current_dateTime)] = []
    data['{}'.format(current_dateTime)].append({'Question': query, 'Answer': response, 'Steps': response_})

    with open(convo_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
