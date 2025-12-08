# Databricks notebook source
# MAGIC %md
# MAGIC # RAGチェーン構築
# MAGIC
# MAGIC Databricks Vector Searchを使用してRAGチェーンを構築します。

# COMMAND ----------

# MAGIC %pip install -U mlflow==3.3.2 langgraph==0.3.4 databricks-langchain==0.7.1 langchain==0.3.27 langchain-core==0.3.75 langchain-community==0.3.29 langgraph-supervisor==0.0.4 pandas==2.2.3 numpy==1.26.4 openpyxl==3.1.5 requests==2.32.5 databricks-vectorsearch==0.57 databricks-sdk==0.65.0 torch==2.4.1 sentence-transformers==5.1.0 "transformers>=4.49.0,<4.51" "tokenizers>=0.21.2,<0.22" google-cloud-vision==3.10.2 google-cloud-speech==2.33.0 google-auth==2.40.3 mutagen==1.47.0 langchain-huggingface==0.3.1 langchain-tavily==0.2.11 "sentencepiece>=0.2.0,<0.3" unihan-etl==0.37.0 jaconv==0.4.0 opencc-python-reimplemented==0.1.7 rapidfuzz==3.14.1 joyokanji==1.1.0 python-docx==1.2.0 pypdf==6.0.0 pillow uv fugashi ipadic unidic-lite openevals torchaudio google-genai databricks-agents httpx

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
    # ChatAgentとしてログ（agent.pyファイルを指定）
    # ChatAgentインターフェースを使用するため、シグネチャは自動的に推論される
    logged_model_info = mlflow.pyfunc.log_model(
        artifact_path="agent",
        python_model="agent.py",
        pip_requirements=[
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
        ],
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

deployment_info = agents.deploy(
    model_name=UC_MODEL_NAME,
    model_version=uc_registered_model_info.version,
    endpoint_name="commuting-allowance-rag-agent-endpoint"
)

print(f"✅ Agent deployed successfully!")
print(f"   Deployment info: {deployment_info}")
print(f"💡 You can now use the agent in Databricks Playground!")
print(f"💡 Review App and API endpoint are available")