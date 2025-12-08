# Databricks notebook source
# MAGIC %md
# MAGIC # RAGチェーン構築
# MAGIC
# MAGIC Databricks Vector Searchを使用してRAGチェーンを構築します。

# COMMAND ----------

# MAGIC %pip install -U langchain langchain-core langchain-databricks databricks-langchain databricks-vectorsearch langchain-huggingface sentence-transformers sentencepiece mlflow databricks-sdk

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from typing import Dict, Any
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedModelInput
from pyspark.sql import SparkSession
import mlflow

spark = SparkSession.builder.getOrCreate()
w = WorkspaceClient()
workspace_url = SparkSession.getActiveSession().conf.get("spark.databricks.workspaceUrl", None)

# COMMAND ----------

from rag_config import RAGConfig
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from databricks_langchain import ChatDatabricks, DatabricksVectorSearch
from langchain_huggingface import HuggingFaceEmbeddings

config = RAGConfig()
VECTOR_SEARCH_ENDPOINT = "databricks-bge-large-en-endpoint"

print(f"CATALOG: {config.catalog}")
print(f"SCHEMA: {config.schema}")
print(f"VECTOR_INDEX_NAME: {config.vector_index_name}")
print(f"QUERY_EMBEDDING_MODEL: {config.query_embedding_model}")
print(f"LLM_ENDPOINT: {config.llm_endpoint}")
print(f"RETRIEVER_TOP_K: {config.retriever_top_k}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## RAGチェーン構築

# COMMAND ----------

chain_config = {
    "llm_model_serving_endpoint_name": "databricks-llama-4-maverick",
    "vector_search_endpoint_name": VECTOR_SEARCH_ENDPOINT,
    "vector_search_index": config.vector_index_name,
    "llm_prompt_template": """あなたは質問に答えるアシスタントです。取得したコンテキストの内容をもとに質問に答えてください。一部のコンテキストが無関係な場合、それを回答に利用しないでください。\nコンテキスト: {context}""",
}

print("Chain Config:")
for key, value in chain_config.items():
    print(f"  {key}: {value}")

# COMMAND ----------

def build_rag_chain(chain_config, config):
    """RAGチェーンを構築"""
    embedding_model = HuggingFaceEmbeddings(model_name=config.query_embedding_model)
    
    vector_store = DatabricksVectorSearch(
        index_name=chain_config["vector_search_index"],
        embedding=embedding_model,
        text_column="chunked_text",
        columns=["chunk_id", "chunked_text"]
    )
    
    llm = ChatDatabricks(
        endpoint=chain_config["llm_model_serving_endpoint_name"],
        extra_params={"temperature": 0.1}
    )
    
    retriever = vector_store.as_retriever(search_kwargs={"k": config.retriever_top_k})
    prompt = ChatPromptTemplate.from_template(chain_config["llm_prompt_template"])
    document_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, document_chain)
    
    return rag_chain, retriever, vector_store

rag_chain, retriever, vector_store = build_rag_chain(chain_config, config)
print("RAG Chain created successfully")

# VectorStoreRetrieverの情報を表示
print("\n=== VectorStoreRetriever Information ===")
print(f"Retriever Type: {type(retriever).__name__}")
print(f"Vector Store Type: {type(vector_store).__name__}")
print(f"Index Name: {chain_config['vector_search_index']}")
print(f"Top K: {config.retriever_top_k}")
print(f"Text Column: chunked_text")
print(f"Columns: ['chunk_id', 'chunked_text']")
print("=" * 40)

# COMMAND ----------

def query_rag(question: str) -> Dict[str, Any]:
    result = rag_chain.invoke({"input": question})
    return {
        "question": question,
        "answer": result.get("answer", ""),
        "context": result.get("context", [])
    }

# COMMAND ----------

def setup_mlflow_experiment():
    experiment_name = f"/Users/{w.current_user.me().user_name}/rag_chain_experiment"
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            mlflow.create_experiment(experiment_name)
    except Exception:
        mlflow.create_experiment(experiment_name)
    mlflow.set_experiment(experiment_name)
    return experiment_name

experiment_name = setup_mlflow_experiment()

# COMMAND ----------

# MAGIC %md
# MAGIC ## MLflowモデル登録

# COMMAND ----------

# 入力例の定義
input_example = {
    "messages": [
        {"role": "user", "content": "通勤手当はいくらまで支給されますか？"}
    ]
}

# LangChainチェーンをMLflowにログ
with mlflow.start_run(run_name="commuting-allowance-rag-chain"):
    logged_chain_info = mlflow.langchain.log_model(
        lc_model=rag_chain,  # チェーンオブジェクトを直接渡す
        model_config=chain_config,  # チェーン設定
        artifact_path="chain",  # MLflowで必要なパス
        input_example=input_example,  # 入力スキーマを保存
        registered_model_name="commuting_allowance_rag_chain"
    )
    
    print(f"LangChain model logged: {logged_chain_info.model_uri}")
    
    mlflow.log_params({
        "llm_model_serving_endpoint_name": chain_config["llm_model_serving_endpoint_name"],
        "vector_search_endpoint_name": chain_config["vector_search_endpoint_name"],
        "vector_search_index": chain_config["vector_search_index"],
        "query_embedding_model": config.query_embedding_model,
        "retriever_top_k": config.retriever_top_k
    })
    
    mlflow.set_tag("task", "llm/v1/chat")
    mlflow.set_tag("embedding_model", config.query_embedding_model)
    mlflow.set_tag("llm", chain_config["llm_model_serving_endpoint_name"])
    mlflow.set_tag("model_type", "chat_completion")
    mlflow.set_tag("chain_type", "retrieval_chain")
    
    # チェーンをロードしてテスト実行（MLflow Trace UIで確認可能）
    print("Testing the chain locally to see the MLflow Trace...")
    loaded_chain = mlflow.langchain.load_model(logged_chain_info.model_uri)
    test_result = loaded_chain.invoke(input_example)
    
    mlflow.log_dict(test_result, "chain_test_result.json")
    print("Chain test executed successfully")
    print(f"Run ID: {mlflow.active_run().info.run_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## MLflow Servingエンドポイント作成

# COMMAND ----------

endpoint_name = config.serving_endpoint_name
model_name = "commuting_allowance_rag_chain"

try:
    from mlflow.tracking import MlflowClient
    client = MlflowClient()
    latest_version = client.get_latest_versions(model_name, stages=["None"])[0]
    model_version = int(latest_version.version)
    print(f"Using latest model version: {model_version}")
except Exception as e:
    print(f"Could not get latest model version: {e}")
    model_version = 1
    print(f"Using default model version: {model_version}")

existing_endpoints = w.serving_endpoints.list()
endpoint_exists = any(ep.name == endpoint_name for ep in existing_endpoints)

if endpoint_exists:
    import time
    max_wait_time = 120
    wait_interval = 5
    elapsed_time = 0
    
    endpoint = w.serving_endpoints.get(endpoint_name)
    state = endpoint.state
    print(f"Existing endpoint state: {state}")
    
    while hasattr(state, 'config_update') and state.config_update == "IN_PROGRESS":
        if elapsed_time >= max_wait_time:
            raise TimeoutError(f"Endpoint update timeout after {max_wait_time} seconds")
        print(f"Waiting for endpoint update to complete... ({elapsed_time}s)")
        time.sleep(wait_interval)
        elapsed_time += wait_interval
        endpoint = w.serving_endpoints.get(endpoint_name)
        state = endpoint.state

environment_vars = {}
if workspace_url:
    environment_vars["DATABRICKS_WORKSPACE_URL"] = workspace_url
    environment_vars["DATABRICKS_HOST"] = workspace_url

served_model = ServedModelInput(
    name=model_name,
    model_name=model_name,
    model_version=str(model_version),
    workload_size="Small",
    scale_to_zero_enabled=True,
    environment_vars=environment_vars if environment_vars else {}
)

if endpoint_exists:
    try:
        w.serving_endpoints.update_config(
            name=endpoint_name,
            served_models=[served_model]
        )
        print(f"Endpoint updated: {endpoint_name}")
    except Exception as e:
        if "currently being updated" in str(e):
            print(f"Endpoint is still being updated. Please wait and try again later.")
            print(f"Current endpoint state: {endpoint.state}")
        else:
            raise
else:
    endpoint_config = EndpointCoreConfigInput(
        name=endpoint_name,
        served_models=[served_model]
    )
    w.serving_endpoints.create(
        name=endpoint_name,
        config=endpoint_config
    )
    print(f"Endpoint created: {endpoint_name}")

endpoint = w.serving_endpoints.get(endpoint_name)
print(f"Status: {endpoint.state}")

