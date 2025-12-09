# Databricks notebook source
# MAGIC %md
# MAGIC # RAGチェーン構築
# MAGIC
# MAGIC Databricks Vector Searchを使用してRAGチェーンを構築します。

# COMMAND ----------

# MAGIC %pip install -r requirements.txt

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from typing import Dict, Any
from pyspark.sql import SparkSession
import mlflow

spark = SparkSession.builder.getOrCreate()

# COMMAND ----------

from rag_config import RAGConfig
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from databricks_langchain import ChatDatabricks, DatabricksVectorSearch
from langchain_huggingface import HuggingFaceEmbeddings

config = RAGConfig()
VECTOR_SEARCH_ENDPOINT = "databricks-bge-large-en-endpoint"

print(f"Config: catalog={config.catalog}, schema={config.schema}, llm={config.llm_endpoint}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## RAGチェーン構築

# COMMAND ----------

chain_config = {
    "llm_model_serving_endpoint_name": config.llm_endpoint,
    "vector_search_endpoint_name": VECTOR_SEARCH_ENDPOINT,
    "vector_search_index": config.vector_index_name,
    "llm_prompt_template": """あなたは質問に答えるアシスタントです。取得したコンテキストの内容をもとに質問に答えてください。一部のコンテキストが無関係な場合、それを回答に利用しないでください。

コンテキスト:
{context}

質問: {input}""",
}

print("Chain config:", chain_config)

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
print("RAG chain created")

# VectorStoreRetrieverの情報を表示
print(f"Retriever: {type(retriever).__name__}, Index: {chain_config['vector_search_index']}, Top K: {config.retriever_top_k}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Build: MLflow Trace記録

# COMMAND ----------

# MLflow Trace UI用にチェーンを実行してトレースを記録（Build）
with mlflow.start_run(run_name="commuting-allowance-rag-chain"):
    # チェーンの設定とパラメータをログ
    mlflow.log_params({
        "llm_model_serving_endpoint_name": chain_config["llm_model_serving_endpoint_name"],
        "vector_search_endpoint_name": chain_config["vector_search_endpoint_name"],
        "vector_search_index": chain_config["vector_search_index"],
        "query_embedding_model": config.query_embedding_model,
        "retriever_top_k": config.retriever_top_k,
        "catalog": config.catalog,
        "schema": config.schema
    })
    
    mlflow.set_tag("task", "llm/v1/chat")
    mlflow.set_tag("embedding_model", config.query_embedding_model)
    mlflow.set_tag("llm", chain_config["llm_model_serving_endpoint_name"])
    mlflow.set_tag("model_type", "chat_completion")
    mlflow.set_tag("chain_type", "retrieval_chain")
    
    # チェーン設定をアーティファクトとして保存
    mlflow.log_dict(chain_config, "chain_config.json")
    
    # MLflow Trace UI用にチェーンを実行（MLflow 2.14.0+では自動的にトレースが記録される）
    print("Executing RAG chain...")
    
    # MLflow LangChain autologを有効化（MLflow 2.14.0+の場合）
    try:
        mlflow.langchain.autolog()
    except AttributeError:
        print("Note: Traces will be recorded when the chain is invoked.")
    
    # チェーンを実行してトレースを記録（Buildの一部）
    trace_question = "通勤手当はいくらまで支給されますか？"
    trace_result = rag_chain.invoke({"input": trace_question})
    
    # トレース結果をログ
    context_docs = trace_result.get("context", [])
    mlflow.log_dict({
        "question": trace_question,
        "answer": trace_result.get("answer", ""),
        "context_documents_count": len(context_docs)
    }, "chain_trace_result.json")
    
    print(f"Trace recorded. Run ID: {mlflow.active_run().info.run_id}")

# COMMAND ----------
# Unity Catalogに接続
mlflow.set_registry_uri("databricks-uc")

# Unity Catalogモデル名
UC_MODEL_NAME = f"{config.catalog}.{config.schema}.commuting_allowance_rag_agent"

# COMMAND ----------

# エージェントをMLflowにログ
# mlflow.pyfunc.log_modelを使用
from mlflow.models.resources import DatabricksServingEndpoint

# リソース定義（LLMエンドポイント）
resources = [
    DatabricksServingEndpoint(endpoint_name=chain_config["llm_model_serving_endpoint_name"])
]

with mlflow.start_run(run_name="commuting-allowance-rag-agent"):
    with open("requirements.txt", "r") as f:
        pip_requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    
    logged_model_info = mlflow.pyfunc.log_model(
        name="agent",
        python_model="agent.py",
        code_paths=["rag_config.py"],  # rag_config.pyを含める
        pip_requirements=pip_requirements,
        resources=resources,
    )
    
    print(f"Agent logged: {logged_model_info.model_uri}")
    
    # パラメータとタグをログ
    mlflow.log_params({
        "llm_model_serving_endpoint_name": chain_config["llm_model_serving_endpoint_name"],
        "vector_search_endpoint_name": chain_config["vector_search_endpoint_name"],
        "vector_search_index": chain_config["vector_search_index"],
        "query_embedding_model": config.query_embedding_model,
        "retriever_top_k": config.retriever_top_k,
        "catalog": config.catalog,
        "schema": config.schema
    })
    
    mlflow.set_tag("task", "llm/v1/chat")
    mlflow.set_tag("embedding_model", config.query_embedding_model)
    mlflow.set_tag("llm", chain_config["llm_model_serving_endpoint_name"])
    mlflow.set_tag("model_type", "databricks-agent")
    mlflow.set_tag("chain_type", "retrieval_chain")
    
    print(f"Run ID: {mlflow.active_run().info.run_id}")

# COMMAND ----------

# Unity Catalogにモデルを登録
uc_registered_model_info = mlflow.register_model(
    model_uri=logged_model_info.model_uri,
    name=UC_MODEL_NAME
)

print(f"Model registered: {UC_MODEL_NAME} v{uc_registered_model_info.version}")

# COMMAND ----------

# エージェントをモデルサービングにデプロイ
from databricks import agents
from databricks.sdk import WorkspaceClient

print(f"Deploying: {UC_MODEL_NAME} v{uc_registered_model_info.version} to commuting-allowance-rag-agent-endpoint")

try:
    deployment_info = agents.deploy(
        model_name=UC_MODEL_NAME,
        model_version=uc_registered_model_info.version,
        endpoint_name="commuting-allowance-rag-agent-endpoint"
    )
    
    print(f"Agent deployed: {deployment_info}")
    
    w = WorkspaceClient()
    endpoint_name = "commuting-allowance-rag-agent-endpoint"
    
    try:
        endpoint = w.serving_endpoints.get(endpoint_name)
        print(f"Endpoint: {endpoint_name}, State: {endpoint.state}")
    except Exception as e:
        print(f"Could not retrieve endpoint info: {e}")
        
except Exception as e:
    print(f"Deploy error: {e}")
    raise

# COMMAND ----------

# MAGIC %md
# MAGIC ## エージェントのテスト
# MAGIC
# MAGIC デプロイされたエージェントをテストします。

# COMMAND ----------

# エージェントをテスト（オプション）
# エンドポイントが準備できるまで少し待つ必要がある場合があります
import time

endpoint_name = "commuting-allowance-rag-agent-endpoint"
try:
    w = WorkspaceClient()
    endpoint = w.serving_endpoints.get(endpoint_name)
    
    if endpoint.state.get("ready") == "READY":
        client = w.serving_endpoints.get_open_ai_client()
        response = client.chat.completions.create(
            model=endpoint_name,
            messages=[{"role": "user", "content": "通勤手当はいくらまで支給されますか？"}]
        )
        print(f"Test response: {response.choices[0].message.content}")
    else:
        print(f"Endpoint not ready: {endpoint.state}")
except Exception as e:
    print(f"Test error: {e}")