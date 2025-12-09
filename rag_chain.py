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

print(f"CATALOG: {config.catalog}")
print(f"SCHEMA: {config.schema}")
print(f"VECTOR_INDEX_NAME: {config.vector_index_name}")
print(f"VECTOR_SEARCH_ENDPOINT: {VECTOR_SEARCH_ENDPOINT}")
print(f"QUERY_EMBEDDING_MODEL: {config.query_embedding_model}")
print(f"LLM_ENDPOINT: {config.llm_endpoint}")
print(f"RETRIEVER_TOP_K: {config.retriever_top_k}")

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
    print("Executing RAG chain for MLflow Trace UI...")
    
    # MLflow LangChain autologを有効化（MLflow 2.14.0+の場合）
    try:
        mlflow.langchain.autolog()
    except AttributeError:
        print("Note: mlflow.langchain.autolog() is not available in this MLflow version.")
        print("Traces will still be recorded when the chain is invoked.")
    
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
    
    print("✅ MLflow Trace recorded successfully")
    print(f"Run ID: {mlflow.active_run().info.run_id}")
    print("💡 You can view the trace in MLflow UI under the 'Traces' tab")

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
        pip_requirements=pip_requirements,
        resources=resources,
    )
    
    print(f"✅ Agent logged: {logged_model_info.model_uri}")
    
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

print(f"✅ Model registered to Unity Catalog: {UC_MODEL_NAME}")
print(f"   Version: {uc_registered_model_info.version}")

# COMMAND ----------

# エージェントをモデルサービングにデプロイ
from databricks import agents
from databricks.sdk import WorkspaceClient

print(f"Deploying agent to serving endpoint...")
print(f"  Model: {UC_MODEL_NAME}")
print(f"  Version: {uc_registered_model_info.version}")
print(f"  Endpoint: commuting-allowance-rag-agent-endpoint")

try:
    deployment_info = agents.deploy(
        model_name=UC_MODEL_NAME,
        model_version=uc_registered_model_info.version,
        endpoint_name="commuting-allowance-rag-agent-endpoint"
    )
    
    print(f"✅ Agent deployed successfully!")
    print(f"   Deployment info: {deployment_info}")
    
    # エンドポイント情報を取得
    w = WorkspaceClient()
    endpoint_name = "commuting-allowance-rag-agent-endpoint"
    
    try:
        endpoint = w.serving_endpoints.get(endpoint_name)
        endpoint_url = f"{w.config.host}/serving-endpoints/{endpoint_name}/invocations"
        
        print(f"\n=== Endpoint Information ===")
        print(f"Endpoint Name: {endpoint_name}")
        print(f"Endpoint URL: {endpoint_url}")
        print(f"Endpoint State: {endpoint.state}")
        print(f"\n💡 You can now use the agent in Databricks Playground!")
        print(f"💡 Review App and API endpoint are available")
        print(f"\n📝 Test the agent:")
        print(f"   from databricks.sdk import WorkspaceClient")
        print(f"   w = WorkspaceClient()")
        print(f"   client = w.serving_endpoints.get_open_ai_client()")
        print(f"   response = client.chat.completions.create(")
        print(f"       model=\"{endpoint_name}\",")
        print(f"       messages=[{{\"role\": \"user\", \"content\": \"通勤手当はいくらまで支給されますか？\"}}]")
        print(f"   )")
        print(f"   print(response.choices[0].message.content)")
    except Exception as e:
        print(f"⚠️ Could not retrieve endpoint information: {e}")
        print(f"💡 Please check the Databricks UI for the endpoint details")
        
except Exception as e:
    print(f"❌ Error deploying agent: {e}")
    print(f"\nTroubleshooting steps:")
    print(f"1. Check if the model is correctly registered in Unity Catalog")
    print(f"2. Verify model version: {uc_registered_model_info.version}")
    print(f"3. Check if endpoint name 'commuting-allowance-rag-agent-endpoint' is available")
    print(f"4. Verify permissions for model serving")
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
print(f"Testing agent endpoint: {endpoint_name}")

try:
    w = WorkspaceClient()
    
    # エンドポイントの状態を確認
    endpoint = w.serving_endpoints.get(endpoint_name)
    print(f"Endpoint state: {endpoint.state}")
    
    if endpoint.state.get("ready") == "READY":
        # OpenAI互換クライアントを取得
        client = w.serving_endpoints.get_open_ai_client()
        
        # テストクエリ
        test_messages = [
            {"role": "user", "content": "通勤手当はいくらまで支給されますか？"}
        ]
        
        print(f"\nSending test query: {test_messages[0]['content']}")
        response = client.chat.completions.create(
            model=endpoint_name,
            messages=test_messages
        )
        
        print(f"\n✅ Agent response:")
        print(f"   {response.choices[0].message.content}")
    else:
        print(f"⚠️ Endpoint is not ready yet. State: {endpoint.state}")
        print(f"💡 Please wait a few minutes and test manually in Databricks Playground")
        
except Exception as e:
    print(f"⚠️ Could not test endpoint automatically: {e}")
    print(f"💡 You can test the agent manually:")
    print(f"   1. Go to Databricks UI > Serving > {endpoint_name}")
    print(f"   2. Click 'Open in Playground'")
    print(f"   3. Send a test message: '通勤手当はいくらまで支給されますか？'")