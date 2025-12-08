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
    # Databricksのユーザー名を取得
    try:
        user_name = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
    except:
        import os
        user_name = os.environ.get("USER", "default_user")
    
    experiment_name = f"/Users/{user_name}/rag_chain_experiment"
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

# MLflow Trace UI用にチェーンを実行してトレースを記録
test_question = "通勤手当はいくらまで支給されますか？"

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
    print(f"Test question: {test_question}")
    
    # MLflow LangChain autologを有効化（MLflow 2.14.0+の場合）
    try:
        mlflow.langchain.autolog()
    except AttributeError:
        print("Note: mlflow.langchain.autolog() is not available in this MLflow version.")
        print("Traces will still be recorded when the chain is invoked.")
    
    # チェーンを実行してトレースを記録
    test_result = rag_chain.invoke({"input": test_question})
    
    # 結果をログ
    mlflow.log_dict({
        "question": test_question,
        "answer": test_result.get("answer", ""),
        "context_documents_count": len(test_result.get("context", []))
    }, "chain_test_result.json")
    
    print("Chain executed successfully")
    print(f"Answer: {test_result.get('answer', '')}")
    print(f"Run ID: {mlflow.active_run().info.run_id}")
    print("You can view the trace in MLflow UI under the 'Traces' tab")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 注意事項
# MAGIC
# MAGIC このノートブックでは、LangChainチェーンを直接MLflowモデルとしてログすることはできません。
# MAGIC 理由：
# MAGIC - `VectorStoreRetriever`には`loader_fn`が必要
# MAGIC - `ChatDatabricks`はカスタムコンポーネントのため直接保存できない
# MAGIC
# MAGIC 代わりに、チェーンを実行してMLflow Trace UIでトレースを確認できます。
# MAGIC MLflow UIの「Traces」タブでチェーンの実行フローを確認してください。

