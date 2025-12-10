# Databricks notebook source
# MAGIC %md
# MAGIC # ChatAgentデプロイ
# MAGIC
# MAGIC このノートブックでは以下の処理を実行します：
# MAGIC 1. ChatAgentモデルのログとUnity Catalogへの登録
# MAGIC 2. モデルサービングエンドポイントへのデプロイ
# MAGIC 3. デプロイされたエージェントのテスト
# MAGIC
# MAGIC **目的**: 本番環境へのChatAgentデプロイ

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

from pyspark.sql import SparkSession
import mlflow
import uuid
from mlflow.models import infer_signature
from mlflow.types.agent import ChatAgentMessage
from agent import AGENT

spark = SparkSession.builder.getOrCreate()

# COMMAND ----------

# MAGIC %md
# MAGIC ## RAG設定の読み込み

# COMMAND ----------

from rag_config import RAGConfig

config = RAGConfig()
print(f"Catalog: {config.catalog}, Schema: {config.schema}")
print(f"Delta Table: {config.delta_table_name}")
print(f"Vector Index: {config.vector_index_name}")
print(f"LLM Endpoint: {config.llm_endpoint}")
print(f"Serving Endpoint Name: {config.serving_endpoint_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Deploy: Unity Catalog設定

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")
UC_MODEL_NAME = f"{config.catalog}.{config.schema}.yona_commuting_allowance_rag_agent"
print(f"Model name: {UC_MODEL_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## リソース定義
# MAGIC
# MAGIC モデルサービング環境で必要なリソース（LLMエンドポイント、Vector Searchインデックス）を定義します。

# COMMAND ----------

from mlflow.models.resources import DatabricksServingEndpoint, DatabricksVectorSearchIndex

chain_config = {
    "llm_model_serving_endpoint_name": config.llm_endpoint,
    "vector_search_index": config.vector_index_name,
}

resources = [
    DatabricksServingEndpoint(endpoint_name=chain_config["llm_model_serving_endpoint_name"]),
    DatabricksVectorSearchIndex(index_name=chain_config["vector_search_index"])
]

# COMMAND ----------

# MAGIC %md
# MAGIC ## ChatAgentモデルのログ
# MAGIC
# MAGIC `agent.py`で定義されたエージェントをMLflowにログします。
# MAGIC models-from-code方式を使用します。

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
    
    # 出力例を生成して署名を推論（Unity Catalog登録に必須）
    messages_for_predict = [
        ChatAgentMessage(
            id=str(uuid.uuid4()),
            role=msg["role"],
            content=msg["content"]
        )
        for msg in input_example["messages"]
    ]
    resp_example = AGENT.predict(messages_for_predict)
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
        "vector_search_endpoint_name": config.vector_search_endpoint,
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

