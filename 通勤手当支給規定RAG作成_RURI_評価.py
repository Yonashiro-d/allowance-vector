# Databricks notebook source
# MAGIC %pip install -U -qqq mlflow==3.3.2 langgraph==0.3.4 databricks-langchain==0.7.1 langchain==0.3.27 langchain-core==0.3.75 langchain-community==0.3.29 langgraph-supervisor==0.0.4 pandas==2.2.3 numpy==1.26.4 openpyxl==3.1.5 requests==2.32.5 databricks-vectorsearch==0.57 databricks-sdk==0.65.0 torch==2.4.1 sentence-transformers==5.1.0 "transformers>=4.49.0,<4.51" "tokenizers>=0.21.2,<0.22" google-cloud-vision==3.10.2 google-cloud-speech==2.33.0 google-auth==2.40.3 mutagen==1.47.0 langchain-huggingface==0.3.1 langchain-tavily==0.2.11 "sentencepiece>=0.2.0,<0.3" unihan-etl==0.37.0 jaconv==0.4.0 opencc-python-reimplemented==0.1.7 rapidfuzz==3.14.1 joyokanji==1.1.0 python-docx==1.2.0 pypdf==6.0.0 pillow uv fugashi ipadic unidic-lite openevals torchaudio google-genai databricks-agents httpx
# MAGIC dbutils.library.restartPython()

# COMMAND ----------


from databricks.sdk import WorkspaceClient
# Use the current user name to create any necesary resources
w = WorkspaceClient()
user_name = w.current_user.me().user_name.split("@")[0].replace(".", "")

# UC Catalog & Schema where outputs tables/indexs are saved
# If this catalog/schema does not exist, you need create catalog/schema permissions.
UC_CATALOG = f'{user_name}_catalog'
UC_CATALOG = "hhhd_demo_itec"
UC_SCHEMA = f'rag_{user_name}'
UC_SCHEMA = "tak_commuting_allowance_policy"

# UC Model name where the POC chain is logged
UC_MODEL_NAME = f"{UC_CATALOG}.{UC_SCHEMA}.{user_name}_agent_quick_start"

# Vector Search endpoint where index is loaded
# If this does not exist, it will be created
VECTOR_SEARCH_ENDPOINT = f'{user_name}_vector_search'

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.vectorsearch import EndpointStatusState, EndpointType
from databricks.sdk.service.serving import EndpointCoreConfigInput, EndpointStateReady
from databricks.sdk.errors import ResourceDoesNotExist
import os

w = WorkspaceClient()

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import NotFound, PermissionDenied
w = WorkspaceClient()

# Create UC Catalog if it does not exist, otherwise, raise an exception
try:
    _ = w.catalogs.get(UC_CATALOG)
    print(f"PASS: UC catalog `{UC_CATALOG}` exists")
except NotFound as e:
    print(f"`{UC_CATALOG}` does not exist, trying to create...")
    try:
        _ = w.catalogs.create(name=UC_CATALOG)
    except PermissionDenied as e:
        print(f"FAIL: `{UC_CATALOG}` does not exist, and no permissions to create.  Please provide an existing UC Catalog.")
        raise ValueError(f"Unity Catalog `{UC_CATALOG}` does not exist.")
        
# Create UC Schema if it does not exist, otherwise, raise an exception
try:
    _ = w.schemas.get(full_name=f"{UC_CATALOG}.{UC_SCHEMA}")
    print(f"PASS: UC schema `{UC_CATALOG}.{UC_SCHEMA}` exists")
except NotFound as e:
    print(f"`{UC_CATALOG}.{UC_SCHEMA}` does not exist, trying to create...")
    try:
        _ = w.schemas.create(name=UC_SCHEMA, catalog_name=UC_CATALOG)
        print(f"PASS: UC schema `{UC_CATALOG}.{UC_SCHEMA}` created")
    except PermissionDenied as e:
        print(f"FAIL: `{UC_CATALOG}.{UC_SCHEMA}` does not exist, and no permissions to create.  Please provide an existing UC Schema.")
        raise ValueError("Unity Catalog Schema `{UC_CATALOG}.{UC_SCHEMA}` does not exist.")

# COMMAND ----------

# Create the Vector Search endpoint if it does not exist
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.vectorsearch import EndpointType
w = WorkspaceClient()
vector_search_endpoints = w.vector_search_endpoints.list_endpoints()
if sum([VECTOR_SEARCH_ENDPOINT == ve.name for ve in vector_search_endpoints]) == 0:
    print(f"Please wait, creating Vector Search endpoint `{VECTOR_SEARCH_ENDPOINT}`.  This can take up to 20 minutes...")
    w.vector_search_endpoints.create_endpoint_and_wait(VECTOR_SEARCH_ENDPOINT, endpoint_type=EndpointType.STANDARD)

# Make sure vector search endpoint is online and ready.
w.vector_search_endpoints.wait_get_endpoint_vector_search_endpoint_online(VECTOR_SEARCH_ENDPOINT)

print(f"PASS: Vector Search endpoint `{VECTOR_SEARCH_ENDPOINT}` exists")

# COMMAND ----------

# UC locations to store the chunked documents & index
CHUNKS_DELTA_TABLE = f"{UC_CATALOG}.{UC_SCHEMA}.databricks_docs_chunked1"
CHUNKS_VECTOR_INDEX = f"{UC_CATALOG}.{UC_SCHEMA}.databricks_docs_chunked_index1"

# COMMAND ----------

WORK_FOLDER_PASS = "/Workspace/Users/takami-m7@itec.hankyu-hanshin.co.jp/commuting_allowance_policy"
PDF_FILE = f"{WORK_FOLDER_PASS}/D005　通勤手当支給規程（2024-04-01）.pdf"

# COMMAND ----------


from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

# PDFファイルのロード
loader = PyPDFLoader(f"{PDF_FILE}")
documents = loader.load()

# 意味のある場所で分けてみる
chunks = []
for doc in documents:
    text = doc.page_content
    paragraphs = [p for p in text.split('\n \n') if p.strip()]
    for para in paragraphs:
        chunks.append(
            Document(page_content=para, metadata=doc.metadata)
        )

print(chunks[2])

# チャンキング済みchunksから必要な情報を抽出
rows = [
    {
        "chunk_id": idx,
        "chunked_text": chunk.page_content,
        "url": chunk.metadata.get("source", "")
    }
    for idx, chunk in enumerate(chunks)
]
print(rows[2])


# COMMAND ----------

####埋め込みモデルに瑠璃を使う####
from pyspark.sql import SparkSession
from databricks.vector_search.client import VectorSearchClient
from sentence_transformers import SentenceTransformer

import time

# Workspace URL for printing links to the delta table/vector index
workspace_url = SparkSession.getActiveSession().conf.get(
    "spark.databricks.workspaceUrl", None
)

# Vector Search client
vsc = VectorSearchClient(disable_notice=True)

# Download from the huggingFace Hub
model = SentenceTransformer("cl-nagoya/ruri-v3-310m")

# 2. テキストをベクトル化
texts = [row["chunked_text"] for row in rows]
embeddings = model.encode(texts)

# 3. Spark DataFrameに埋め込みを追加
import pandas as pd

df = pd.DataFrame(rows)
df["embedding"] = list(embeddings)
chunked_docs_df = spark.createDataFrame(df)

# Get the current columns of the table
columns = [f.name for f in spark.table(CHUNKS_DELTA_TABLE).schema.fields]

# Add the column only if it does not exist
if "embedding" not in columns:
    # ALTER TABLE を使用して、embedding列を追加
    spark.sql(
        f"ALTER TABLE {CHUNKS_DELTA_TABLE} ADD COLUMNS (embedding ARRAY<FLOAT>)"
    )

# 4. Deltaテーブルに保存
chunked_docs_df.write.format("delta").mode("overwrite").saveAsTable(CHUNKS_DELTA_TABLE)
spark.sql(
    f"ALTER TABLE {CHUNKS_DELTA_TABLE} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)"
)


print(
    f"View Delta Table at: https://{workspace_url}/explore/data/{UC_CATALOG}/{UC_SCHEMA}/{CHUNKS_DELTA_TABLE.split('.')[-1]}"
)

# Embed and sync chunks to a vector index
print(
    f"Embedding docs & creating Vector Search Index, this will take ~5 - 10 minutes.\nView Index Status at: https://{workspace_url}/explore/data/{UC_CATALOG}/{UC_SCHEMA}/{CHUNKS_VECTOR_INDEX.split('.')[-1]}"
)

# インデックスの削除
try:
    vsc.delete_index(
        endpoint_name=VECTOR_SEARCH_ENDPOINT, index_name=CHUNKS_VECTOR_INDEX
    )
except Exception as e:
    print(
        "Delete index request sent or index does not exist. Waiting for deletion to complete..."
    )

# インデックスの削除が完了するまで待つ
while True:
    try:
        status = vsc.get_index(
            endpoint_name=VECTOR_SEARCH_ENDPOINT, index_name=CHUNKS_VECTOR_INDEX
        )
        print(f"Index status: {status['status']}")
        if status["status"] == "DELETED":
            time.sleep(30)
            continue
        time.sleep(30)
    except Exception as e:
        # If get_index fails, index is deleted
        print("Index deleted and ready to be recreated.")
        break

time.sleep(15)
# インデックスの作成
index = vsc.create_delta_sync_index_and_wait(
    endpoint_name=VECTOR_SEARCH_ENDPOINT,
    index_name=CHUNKS_VECTOR_INDEX,
    primary_key="chunk_id",
    source_table_name=CHUNKS_DELTA_TABLE,
    pipeline_type="TRIGGERED",
    embedding_vector_column="embedding",  # 追加したembeddingカラム
    embedding_dimension=768,#embeddings.shape[1],  # Set this to your embedding dimension
    embedding_model_endpoint_name=None,  # すでに埋め込み済みなのでNone
)
    # embedding_source_column="chunked_text",
    # embedding_model_endpoint_name="databricks-gte-large-en",

# COMMAND ----------

chain_config = {
    "llm_model_serving_endpoint_name": "databricks-llama-4-maverick",  # the foundation model we want to use
    "vector_search_endpoint_name": VECTOR_SEARCH_ENDPOINT,  # Endoint for vector search
    "vector_search_index": f"{CHUNKS_VECTOR_INDEX}",
    "llm_prompt_template": """You are an assistant that answers questions. Use the following pieces of retrieved context to answer the question. Some pieces of context may be irrelevant, in which case you should not use them to form the answer.\n\nContext: {context}""", # LLM Prompt template
    "vector_search_embedding_model": "cl-nagoya/ruri-v3-310m"
}

# Here, we define an input example in the schema required by Agent Framework
input_example = {"messages": [ {"role": "user", "content": "通勤手当はいくらまで支給されますか？"}]}

# COMMAND ----------

from databricks_langchain import ChatDatabricks, UCFunctionToolkit, VectorSearchRetrieverTool, DatabricksVectorSearch
from langchain_community.embeddings import HuggingFaceEmbeddings

embedding_model = HuggingFaceEmbeddings(model_name="cl-nagoya/ruri-v3-310m")

VECTOR_SEARCH_ENDPOINT_1 = "takami-m7_vector_search"
CHUNKS_VECTOR_INDEX_1 = "hhhd_demo_itec.tak_commuting_allowance_policy.databricks_docs_chunked_index1"
vector_search_as_rule = DatabricksVectorSearch(
    endpoint=VECTOR_SEARCH_ENDPOINT_1,
    index_name=CHUNKS_VECTOR_INDEX_1,
    embedding=embedding_model,
    text_column="chunked_text",
    columns=["chunk_id", "chunked_text"]
).as_retriever(search_kwargs={"k": 10, "score_threshold": 0.003})
question = "こんにちは"
company_rule = vector_search_as_rule.invoke(question)

# COMMAND ----------

import mlflow
from langchain_community.embeddings import HuggingFaceEmbeddings
from mlflow.models import infer_signature

# Log the model to MLflow

embedding = HuggingFaceEmbeddings(model_name="cl-nagoya/ruri-v3-310m")
signature = infer_signature(
    {"messages": [{"role": "user", "content": "通勤手当はいくらまで支給されますか？"}]}, "sample_output"
)
print("!!!!!!mlflow.start_run!!!!!!")
with mlflow.start_run(run_name="databricks-docs-bot"):
    logged_chain_info = mlflow.langchain.log_model(
        model_config=chain_config,  # Chain configuration set above
        lc_model=os.path.join(
            os.getcwd(),
            f"{WORK_FOLDER_PASS}/commuting_allowance_policy_rag_chain_ruri",
        ),  # Chain code file from the quick start repo
        artifact_path="chain",  # Required by MLflow
        input_example=input_example,  # Save the chain's input schema.  MLflow will execute the chain before logging & capture it's output schema.
        signature=signature,
        extra_pip_requirements=[
            "mlflow==3.3.2",
            "langgraph==0.3.4",
            "databricks-langchain==0.7.1",
            "langchain==0.3.27",
            "langchain-core==0.3.75",
            "langchain-community==0.3.29",
            "langgraph-supervisor==0.0.4",
            "pandas==2.2.3",
            "numpy==1.26.4",
            "openpyxl==3.1.5",
            "requests==2.32.5",
            "databricks-vectorsearch==0.57",
            "databricks-sdk==0.65.0",
            "torch==2.4.1",
            "sentence-transformers==5.1.0",
            "transformers>=4.49.0,<4.51",
            "tokenizers>=0.21.2,<0.22",
            "google-cloud-vision==3.10.2",
            "google-cloud-speech==2.33.0",
            "google-auth==2.40.3",
            "mutagen==1.47.0",
            "langchain-huggingface==0.3.1",
            "langchain-tavily==0.2.11",
            "sentencepiece>=0.2.0,<0.3",
            "unihan-etl==0.37.0",
            "jaconv==0.4.0",
            "opencc-python-reimplemented==0.1.7",
            "rapidfuzz==3.14.1",
            "joyokanji==1.1.0",
            "python-docx==1.2.0",
            "pypdf==6.0.0",
            "pillow",
            "uv",
            "fugashi",
            "ipadic",
            "unidic-lite",
            "openevals",
            "torchaudio",
            "google-genai",
            "databricks-agents",
            "httpx",
        ],  #### 何度もXXXパッケージが足りないってエラーが出るので、先頭でpipしてるパッケージを全部書いた。多分よくない。
        ###これでだめなら以下を調べながらやる。
        # mlflow.models.predict()は、MLflowで保存したモデルに対して推論（予測）を行うためのAPIです。
        # この関数を使うことで、MLflowモデルをローカルやサンドボックス環境でテストし、依存パッケージの不足や推論時のエラーを事前に確認できます。
        #  extra_pip_requirements　各パッケージをリストの要素として1つずつ記述する必要がある。バージョン指定をする場合は、ダブルクォートやエスケープは不要。
    )


# Test the chain locally to see the MLflow Trace
chain = mlflow.langchain.load_model(logged_chain_info.model_uri)
chain.invoke(input_example)

# COMMAND ----------

import mlflow
from langchain_community.embeddings import HuggingFaceEmbeddings
from mlflow.models import infer_signature

# Log the model to MLflow

embedding = HuggingFaceEmbeddings(model_name="cl-nagoya/ruri-v3-310m")
signature = infer_signature(
    {"messages": [{"role": "user", "content": "通勤手当はいくらまで支給されますか？"}]}, "sample_output"
)
print("!!!!!!mlflow.start_run!!!!!!")
with mlflow.start_run(run_name="databricks-docs-bot"):
    logged_chain_info = mlflow.langchain.log_model(
        model_config=chain_config,  # Chain configuration set above
        lc_model=os.path.join(
            os.getcwd(),
            f"{WORK_FOLDER_PASS}/commuting_allowance_policy_rag_chain_ruri",
        ),  # Chain code file from the quick start repo
        artifact_path="chain",  # Required by MLflow
        input_example=input_example,  # Save the chain's input schema.  MLflow will execute the chain before logging & capture it's output schema.
        signature=signature,
        extra_pip_requirements=[
            "mlflow==3.3.2",
            "langgraph==0.3.4",
            "databricks-langchain==0.7.1",
            "langchain==0.3.27",
            "langchain-core==0.3.75",
            "langchain-community==0.3.29",
            "langgraph-supervisor==0.0.4",
            "pandas==2.2.3",
            "numpy==1.26.4",
            "openpyxl==3.1.5",
            "requests==2.32.5",
            "databricks-vectorsearch==0.57",
            "databricks-sdk==0.65.0",
            "torch==2.4.1",
            "sentence-transformers==5.1.0",
            "transformers>=4.49.0,<4.51",
            "tokenizers>=0.21.2,<0.22",
            "google-cloud-vision==3.10.2",
            "google-cloud-speech==2.33.0",
            "google-auth==2.40.3",
            "mutagen==1.47.0",
            "langchain-huggingface==0.3.1",
            "langchain-tavily==0.2.11",
            "sentencepiece>=0.2.0,<0.3",
            "unihan-etl==0.37.0",
            "jaconv==0.4.0",
            "opencc-python-reimplemented==0.1.7",
            "rapidfuzz==3.14.1",
            "joyokanji==1.1.0",
            "python-docx==1.2.0",
            "pypdf==6.0.0",
            "pillow",
            "uv",
            "fugashi",
            "ipadic",
            "unidic-lite",
            "openevals",
            "torchaudio",
            "google-genai",
            "databricks-agents",
            "httpx",
        ],  #### 何度もXXXパッケージが足りないってエラーが出るので、先頭でpipしてるパッケージを全部書いた。多分よくない。
        ###これでだめなら以下を調べながらやる。
        # mlflow.models.predict()は、MLflowで保存したモデルに対して推論（予測）を行うためのAPIです。
        # この関数を使うことで、MLflowモデルをローカルやサンドボックス環境でテストし、依存パッケージの不足や推論時のエラーを事前に確認できます。
        #  extra_pip_requirements　各パッケージをリストの要素として1つずつ記述する必要がある。バージョン指定をする場合は、ダブルクォートやエスケープは不要。
    )


# Test the chain locally to see the MLflow Trace
chain = mlflow.langchain.load_model(logged_chain_info.model_uri)
chain.invoke(input_example)

# COMMAND ----------

import mlflow
from langchain_community.embeddings import HuggingFaceEmbeddings
from mlflow.models import infer_signature

# Log the model to MLflow

embedding = HuggingFaceEmbeddings(model_name="cl-nagoya/ruri-v3-310m")
signature = infer_signature(
    {"messages": [{"role": "user", "content": "通勤手当はいくらまで支給されますか？"}]}, "sample_output"
)
print("!!!!!!mlflow.start_run!!!!!!")
with mlflow.start_run(run_name="databricks-docs-bot"):
    logged_chain_info = mlflow.langchain.log_model(
        model_config=chain_config,  # Chain configuration set above
        lc_model=os.path.join(
            os.getcwd(),
            f"{WORK_FOLDER_PASS}/commuting_allowance_policy_rag_chain_ruri",
        ),  # Chain code file from the quick start repo
        artifact_path="chain",  # Required by MLflow
        input_example=input_example,  # Save the chain's input schema.  MLflow will execute the chain before logging & capture it's output schema.
        signature=signature,
        extra_pip_requirements=[
            "mlflow==3.3.2",
            "langgraph==0.3.4",
            "databricks-langchain==0.7.1",
            "langchain==0.3.27",
            "langchain-core==0.3.75",
            "langchain-community==0.3.29",
            "langgraph-supervisor==0.0.4",
            "pandas==2.2.3",
            "numpy==1.26.4",
            "openpyxl==3.1.5",
            "requests==2.32.5",
            "databricks-vectorsearch==0.57",
            "databricks-sdk==0.65.0",
            "torch==2.4.1",
            "sentence-transformers==5.1.0",
            "transformers>=4.49.0,<4.51",
            "tokenizers>=0.21.2,<0.22",
            "google-cloud-vision==3.10.2",
            "google-cloud-speech==2.33.0",
            "google-auth==2.40.3",
            "mutagen==1.47.0",
            "langchain-huggingface==0.3.1",
            "langchain-tavily==0.2.11",
            "sentencepiece>=0.2.0,<0.3",
            "unihan-etl==0.37.0",
            "jaconv==0.4.0",
            "opencc-python-reimplemented==0.1.7",
            "rapidfuzz==3.14.1",
            "joyokanji==1.1.0",
            "python-docx==1.2.0",
            "pypdf==6.0.0",
            "pillow",
            "uv",
            "fugashi",
            "ipadic",
            "unidic-lite",
            "openevals",
            "torchaudio",
            "google-genai",
            "databricks-agents",
            "httpx",
        ],  #### 何度もXXXパッケージが足りないってエラーが出るので、先頭でpipしてるパッケージを全部書いた。多分よくない。
        ###これでだめなら以下を調べながらやる。
        # mlflow.models.predict()は、MLflowで保存したモデルに対して推論（予測）を行うためのAPIです。
        # この関数を使うことで、MLflowモデルをローカルやサンドボックス環境でテストし、依存パッケージの不足や推論時のエラーを事前に確認できます。
        #  extra_pip_requirements　各パッケージをリストの要素として1つずつ記述する必要がある。バージョン指定をする場合は、ダブルクォートやエスケープは不要。
    )


# Test the chain locally to see the MLflow Trace
chain = mlflow.langchain.load_model(logged_chain_info.model_uri)
chain.invoke(input_example)

# COMMAND ----------

import mlflow
from mlflow.genai import datasets, evaluate, scorers

# MLflowで「実験」を管理します。実験IDを指定することで、評価結果を特定の実験に紐付けて保存できます。
mlflow.set_experiment(experiment_id="2846072386114943")

# Step 1: 評価用データセットの定義
eval_dataset = [
    {
        "inputs": {"messages": [{"role": "user", "content": "こんにちは！"}]},
        "expectations": {
            "expected_response": "通勤手当支給規定に関係のない質問にはお答えしかねます。",
            "guidelines": ["回答は通勤手当支給規定に関する内容のみとすること"],
        },
    },
    {
        "inputs": {"messages": [{"role": "user", "content": "引越ししたときはどうしたらいいですか？"}]},
        "expectations": {
            "expected_response": "引越しにより通勤経路が変更になった場合、第10条（届出の義務）に従い、従業員は会社に届け出る必要があります。",
            "guidelines": ["回答は規定の該当条文を引用すること"],
        },
    },
    {
        "inputs": {"messages": [{"role": "user", "content": "通勤手当はいくらまで支給されますか？"}]},
        "expectations": {
            "expected_response": "会社規定第４条第４項によれば、通勤手当の支給額は非課税限度額を上限とすると定められています。したがって、通勤手当は非課税限度額まで支給されます。",
            "guidelines": ["回答は規定の該当条文を引用すること"],
        },
    },
]


# Step 2:  予測関数（predict_fn）の定義
# predict関数は、評価データセットの各行に対して呼び出されます。
# ここにモデルの推論処理を記述します。
def predict(messages):
    return chain.invoke({"messages": messages})


# Step 3: 評価の実行
evaluate(data=eval_dataset, predict_fn=predict, scorers=scorers.get_all_scorers())

# Results will appear back in this UI

# COMMAND ----------

# DBTITLE 1,10-minDemoのやつからこぴぺ
from databricks import agents
import time
from databricks.sdk.service.serving import EndpointStateReady, EndpointStateConfigUpdate

# Use Unity Catalog to log the chain
mlflow.set_registry_uri('databricks-uc')

# Register the chain to UC
uc_registered_model_info = mlflow.register_model(model_uri=logged_chain_info.model_uri, name=UC_MODEL_NAME)

# Deploy to enable the Review APP and create an API endpoint
deployment_info = agents.deploy(model_name=UC_MODEL_NAME, model_version=uc_registered_model_info.version)

# Wait for the Review App to be ready
print("\nWaiting for endpoint to deploy.  This can take 15 - 20 minutes.", end="")


# COMMAND ----------

endpoint_status = w.serving_endpoints.get(deployment_info.endpoint_name)
print(endpoint_status.state.ready)
print(endpoint_status.state.config_update)

# while (
#     endpoint_status.state.ready == EndpointStateReady.NOT_READY
#     or endpoint_status.state.config_update == EndpointStateConfigUpdate.IN_PROGRESS
# ):    
#     print(endpoint_status.state.ready)
#     print(endpoint_status.state.config_update)
#     if endpoint_status.state.config_update == EndpointStateConfigUpdate.UPDATE_FAILED:
#         print(
#             f"Endpoint failed to deploy.  Please check the Databricks UI for more details."
#         )
#         break

#     if endpoint_status.state.config_update ==EndpointStateConfigUpdate.NOT_UPDATING:
#         print(
#             f"complete."
#         )
#         break

#     time.sleep(30)

# COMMAND ----------

import mlflow
from databricks.agents.monitoring import (
    create_external_monitor,
    AssessmentsSuiteConfig,
    BuiltinJudge,
    GuidelinesJudge,
)

for exp in mlflow.search_experiments():
    print(f"Name: {exp.name}, ID: {exp.experiment_id}")

# Create a monitor with multiple scorers
external_monitor = create_external_monitor(
    catalog_name=UC_CATALOG,
    schema_name=UC_SCHEMA,
    experiment_id="3309131107802072",
    assessments_config=AssessmentsSuiteConfig(
        sample=0.5,  # Sample 50% of traces
        assessments=[
            BuiltinJudge(name="safety"),
            BuiltinJudge(name="relevance_to_query"),
            BuiltinJudge(
                name="groundedness", sample_rate=0.2
            ),  # Override sampling for this scorer
            GuidelinesJudge(
                guidelines={
                    "mlflow_only": [
                        "If the request is unrelated to MLflow, the response must refuse to answer."
                    ],
                    "professional_tone": [
                        "The response must maintain a professional and helpful tone."
                    ],
                }
            ),
        ],
    ),
)

# COMMAND ----------

import mlflow
from databricks.agents.monitoring import (
    create_external_monitor,get_external_monitor,update_external_monitor,
    AssessmentsSuiteConfig,
    BuiltinJudge,
    GuidelinesJudge,
)
monitor = get_external_monitor(experiment_id="3309131107802072" )

mlflow.set_experiment(experiment_id="3309131107802072")

# モニターを更新して評価指標を追加
monitor = update_external_monitor(
    experiment_id="3309131107802072",
    assessments_config=AssessmentsSuiteConfig(
        sample=0.5,  # Sample 50% of traces
        assessments=[
            BuiltinJudge(name="safety"),
            BuiltinJudge(name="relevance_to_query"),
            BuiltinJudge(
                name="groundedness", sample_rate=0.2
            ),  # Override sampling for this scorer
            GuidelinesJudge(
                guidelines={
                    "mlflow_only": [
                        "If the request is unrelated to MLflow, the response must refuse to answer."
                    ],
                    "professional_tone": [
                        "The response must maintain a professional and helpful tone."
                    ],
                }
            ),
        ],
    ),
)

# COMMAND ----------

import mlflow
from databricks.agents.monitoring import (
    create_monitor,
    get_monitor,
    AssessmentsSuiteConfig,
    BuiltinJudge,
    GuidelinesJudge,
)

# Create a monitor for the endpoint if it does not exist
assessments_config = AssessmentsSuiteConfig(
    sample=0.5,
    assessments=[
        BuiltinJudge(name="safety"),
        BuiltinJudge(name="relevance_to_query"),
        BuiltinJudge(name="groundedness", sample_rate=0.2),
        GuidelinesJudge(
            guidelines={
                "mlflow_only": [
                    "If the request is unrelated to MLflow, the response must refuse to answer."
                ],
                "professional_tone": [
                    "The response must maintain a professional and helpful tone."
                ],
            }
        ),
    ],
)

print(f"{deployment_info.endpoint_name}")
create_monitor(
    endpoint_name=deployment_info.endpoint_name,
    experiment_id="3309131107802072",
    assessments_config=assessments_config,
)

# 現在のモニター設定を取得
monitor = get_monitor(endpoint_name=deployment_info.endpoint_name)