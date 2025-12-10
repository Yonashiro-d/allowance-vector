# Databricks notebook source
# MAGIC %md
# MAGIC # RAGチェーン構築とエージェントデプロイ
# MAGIC
# MAGIC このノートブックでは以下の処理を実行します：
# MAGIC 1. RAGチェーンの構築（Vector Search + LLM）
# MAGIC 2. MLflow Trace記録（Build）
# MAGIC 3. ChatAgentモデルのログとUnity Catalogへの登録
# MAGIC 4. モデルサービングエンドポイントへのデプロイ
# MAGIC 5. デプロイされたエージェントのテスト

# COMMAND ----------

# MAGIC %md
# MAGIC ## 依存関係のインストール

# COMMAND ----------

# MAGIC %pip install -r requirements.txt

# COMMAND ----------

# MAGIC %md
# MAGIC ## ライブラリの再起動

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 基本インポート

# COMMAND ----------

from typing import Any
import uuid
import os
from pyspark.sql import SparkSession
import mlflow

spark = SparkSession.builder.getOrCreate()

# COMMAND ----------

# MAGIC %md
# MAGIC ## RAG設定の読み込み

# COMMAND ----------

from rag_config import RAGConfig
from agent import AGENT
from mlflow.models.signature import infer_signature

config = RAGConfig()
print(f"Catalog: {config.catalog}, Schema: {config.schema}")
print(f"Delta Table: {config.delta_table_name}")
print(f"Vector Index: {config.vector_index_name}")
print(f"LLM Endpoint: {config.llm_endpoint}")
print(f"Serving Endpoint Name: {config.serving_endpoint_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## LangChain関連のインポート

# COMMAND ----------

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from databricks_langchain import ChatDatabricks, DatabricksVectorSearch
from langchain_huggingface import HuggingFaceEmbeddings

# COMMAND ----------

# MAGIC %md
# MAGIC ## チェーン設定の定義

# COMMAND ----------

chain_config = {
    "llm_model_serving_endpoint_name": config.llm_endpoint,
    "vector_search_endpoint_name": config.vector_search_endpoint,
    "vector_search_index": config.vector_index_name,
    "llm_prompt_template": """あなたは質問に答えるアシスタントです。取得したコンテキストの内容をもとに質問に答えてください。一部のコンテキストが無関係な場合、それを回答に利用しないでください。

コンテキスト:
{context}

質問: {input}""",
}

# COMMAND ----------

# MAGIC %md
# MAGIC ## RAGチェーン構築関数の定義
# MAGIC ## RAGチェーンの構築
# MAGIC
# MAGIC `agent.py`の`create_rag_chain`関数と同様のロジックでRAGチェーンを構築します。
# MAGIC この関数はBuildフェーズ（MLflow Trace記録）で使用されます。

# COMMAND ----------

def build_rag_chain(chain_config: dict[str, Any], config: RAGConfig) -> tuple[Any, Any, Any]:
    embedding_model = HuggingFaceEmbeddings(model_name=config.query_embedding_model)
    
    vector_store = DatabricksVectorSearch(
        index_name=chain_config["vector_search_index"],
        embedding=embedding_model,
        text_column="chunked_text"
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
print(f"RAG chain created: {type(rag_chain).__name__}")
print(f"Retriever: {type(retriever).__name__}, Top K: {config.retriever_top_k}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Build: MLflow Trace記録
# MAGIC
# MAGIC RAGチェーンを実行し、MLflow Trace UIで確認できるようにトレースを記録します。

# COMMAND ----------

with mlflow.start_run(run_name="yona-commuting-allowance-rag-chain"):
    mlflow.log_params({
        "llm_model_serving_endpoint_name": chain_config["llm_model_serving_endpoint_name"],
        "vector_search_endpoint_name": chain_config["vector_search_endpoint_name"],
        "vector_search_index": chain_config["vector_search_index"],
        "delta_table_name": config.delta_table_name,
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
    
    mlflow.log_dict(chain_config, "chain_config.json")
    
    try:
        mlflow.langchain.autolog()
    except AttributeError:
        pass
    
    trace_question = "通勤手当はいくらまで支給されますか？"
    trace_result = rag_chain.invoke({"input": trace_question})
    
    context_docs = trace_result.get("context", [])
    mlflow.log_dict({
        "question": trace_question,
        "answer": trace_result.get("answer", ""),
        "context_documents_count": len(context_docs)
    }, "chain_trace_result.json")
    
    print(f"Trace recorded. Run ID: {mlflow.active_run().info.run_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Deploy: Unity Catalog設定

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")
UC_MODEL_NAME = f"{config.catalog}.{config.schema}.commuting_allowance_rag_agent"
print(f"Model name: {UC_MODEL_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## リソース定義
# MAGIC
# MAGIC モデルサービング環境で必要なリソース（LLMエンドポイント、Vector Searchインデックス）を定義します。

# COMMAND ----------

from mlflow.models.resources import DatabricksServingEndpoint, DatabricksVectorSearchIndex

resources = [
    DatabricksServingEndpoint(endpoint_name=chain_config["llm_model_serving_endpoint_name"]),
    DatabricksVectorSearchIndex(index_name=chain_config["vector_search_index"])
]

# COMMAND ----------

# MAGIC %md
# MAGIC ## ChatAgentモデルのログ
# MAGIC
# MAGIC `agent.py`で定義されたエージェントをMLflowにログします。
# MAGIC `python_model`には`AGENT`オブジェクトを直接指定します。

# COMMAND ----------

with mlflow.start_run(run_name="yona-commuting-allowance-rag-agent"):
    with open("requirements.txt", "r") as f:
        pip_requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    
    input_example = {
        "messages": [
            {
                "role": "user",
                "content": "通勤手当はいくらまで支給されますか？",
            }
        ]
    }
    
    # 出力例を生成して署名を付与（UC登録に必須）
    resp_example = AGENT.predict(input_example["messages"])
    output_example = {
        "messages": [
            {
                "id": m.id,
                "role": m.role,
                "content": m.content,
            }
            for m in resp_example.messages
        ]
    }
    signature = infer_signature(input_example, output_example)
    
    logged_model_info = mlflow.pyfunc.log_model(
        artifact_path="agent",
        python_model="agent_model.py",
        code_paths=["agent.py", "rag_config.py"],
        pip_requirements=pip_requirements,
        resources=resources,
        input_example=input_example,
        signature=signature,
    )
    
    print(f"Agent logged: {logged_model_info.model_uri}")
    
    mlflow.log_params({
        "llm_model_serving_endpoint_name": chain_config["llm_model_serving_endpoint_name"],
        "vector_search_endpoint_name": chain_config["vector_search_endpoint_name"],
        "vector_search_index": chain_config["vector_search_index"],
        "delta_table_name": config.delta_table_name,
        "query_embedding_model": config.query_embedding_model,
        "retriever_top_k": config.retriever_top_k,
        "catalog": config.catalog,
        "schema": config.schema
    })
    
    
    print(f"Run ID: {mlflow.active_run().info.run_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Unity Catalogへのモデル登録

# COMMAND ----------

uc_registered_model_info = mlflow.register_model(
    model_uri=logged_model_info.model_uri,
    name=UC_MODEL_NAME
)

print(f"Model registered: {UC_MODEL_NAME} v{uc_registered_model_info.version}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## モデルサービングエンドポイントへのデプロイ

# COMMAND ----------

from databricks import agents
from databricks.sdk import WorkspaceClient

endpoint_name = config.serving_endpoint_name
print(f"Deploying: {UC_MODEL_NAME} v{uc_registered_model_info.version} to {endpoint_name}")

w = WorkspaceClient()

try:
    deployment_info = agents.deploy(
        model_name=UC_MODEL_NAME,
        model_version=uc_registered_model_info.version,
        endpoint_name=endpoint_name
    )
    print(f"Agent deployed: {deployment_info}")
except Exception as e:
    print(f"Deploy error: {e}")
    raise

# COMMAND ----------

# MAGIC %md
# MAGIC ## エージェントのテスト
# MAGIC
# MAGIC デプロイされたエージェントをテストします。成功した場合はPlaygroundでも同様にテストできます。

# COMMAND ----------

endpoint_name = config.serving_endpoint_name
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
