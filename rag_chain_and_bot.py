# Databricks notebook source
# MAGIC %md
# MAGIC # RAGチェーン構築とBot実装
# MAGIC
# MAGIC Databricks Vector Searchを使用してRAGを構築し、MLflowでBotとしてデプロイします。

# COMMAND ----------

# MAGIC %pip install -U langchain langchain-databricks databricks-langchain databricks-vectorsearch databricks-sdk mlflow pandas langchain-huggingface sentence-transformers sentencepiece

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from typing import Dict, Any, List
from pyspark.sql import SparkSession
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedModelInput
from databricks_langchain import ChatDatabricks, DatabricksVectorSearch
from langchain_huggingface import HuggingFaceEmbeddings
import mlflow
import mlflow.pyfunc

spark = SparkSession.builder.getOrCreate()
w = WorkspaceClient()

workspace_url = SparkSession.getActiveSession().conf.get("spark.databricks.workspaceUrl", None)

# COMMAND ----------

CATALOG = "hhhd_demo_itec"
SCHEMA = "allowance_payment_rules"
VECTOR_INDEX_NAME = f"{CATALOG}.{SCHEMA}.commuting_allowance_index"
QUERY_EMBEDDING_MODEL = "cl-nagoya/ruri-v3-310m"
LLM_ENDPOINT = "databricks-meta-llama-3-1-405b-instruct"
RETRIEVER_TOP_K = 5

print(f"CATALOG: {CATALOG}")
print(f"SCHEMA: {SCHEMA}")
print(f"VECTOR_INDEX_NAME: {VECTOR_INDEX_NAME}")
print(f"QUERY_EMBEDDING_MODEL: {QUERY_EMBEDDING_MODEL}")
print(f"LLM_ENDPOINT: {LLM_ENDPOINT}")
print(f"RETRIEVER_TOP_K: {RETRIEVER_TOP_K}")

# COMMAND ----------

embedding_model = HuggingFaceEmbeddings(model_name=QUERY_EMBEDDING_MODEL)

vector_store = DatabricksVectorSearch(
    index_name=VECTOR_INDEX_NAME,
    embedding=embedding_model,
    text_column="chunked_text",
    columns=["chunk_id", "chunked_text"]
)

llm = ChatDatabricks(
    endpoint=LLM_ENDPOINT,
    extra_params={"temperature": 0.1}
)

# COMMAND ----------

def query_rag(question: str) -> Dict[str, Any]:
    documents = vector_store.similarity_search(question, k=RETRIEVER_TOP_K)
    context = "\n\n".join([doc.page_content for doc in documents])
    
    prompt = f"""以下のコンテキスト情報を使用して、質問に答えてください。
コンテキスト情報に基づいて回答し、コンテキストにない情報は推測せずに「わかりません」と答えてください。

コンテキスト情報:
{context}

質問: {question}

回答:"""
    
    response = llm.invoke(prompt)
    answer = response.content if hasattr(response, 'content') else str(response)
    
    sources = [{"content": doc.page_content, "metadata": doc.metadata} for doc in documents]
    
    return {
        "question": question,
        "answer": answer,
        "sources": sources,
        "num_sources": len(sources)
    }

# COMMAND ----------

class RAGModel(mlflow.pyfunc.PythonModel):
    def __init__(self):
        self.vector_store = None
        self.llm = None
        self.retriever_top_k = 5
    
    def load_context(self, context):
        import os
        from databricks_langchain import ChatDatabricks, DatabricksVectorSearch
        from langchain_huggingface import HuggingFaceEmbeddings
        
        catalog = os.environ.get("CATALOG", "hhhd_demo_itec")
        schema = os.environ.get("SCHEMA", "allowance_payment_rules")
        vector_index_name = os.environ.get("VECTOR_INDEX_NAME", f"{catalog}.{schema}.commuting_allowance_index")
        query_embedding_model = os.environ.get("QUERY_EMBEDDING_MODEL", "cl-nagoya/ruri-v3-310m")
        llm_endpoint = os.environ.get("LLM_ENDPOINT", "databricks-meta-llama-3-1-405b-instruct")
        retriever_top_k = int(os.environ.get("RETRIEVER_TOP_K", "5"))
        
        embedding_model = HuggingFaceEmbeddings(model_name=query_embedding_model)
        
        self.vector_store = DatabricksVectorSearch(
            index_name=vector_index_name,
            embedding=embedding_model,
            text_column="chunked_text",
            columns=["chunk_id", "chunked_text"]
        )
        
        self.llm = ChatDatabricks(
            endpoint=llm_endpoint,
            extra_params={"temperature": 0.1}
        )
        
        self.retriever_top_k = retriever_top_k
    
    def predict(self, context, model_input):
        questions: List[str] = []
        if isinstance(model_input, str):
            questions = [model_input]
        elif isinstance(model_input, list):
            questions = model_input
        elif hasattr(model_input, 'iloc'):
            questions = model_input.iloc[:, 0].tolist()
        else:
            questions = [str(model_input)]
        
        results = []
        for question in questions:
            try:
                documents = self.vector_store.similarity_search(question, k=self.retriever_top_k)
                context_text = "\n\n".join([doc.page_content for doc in documents])
                
                prompt = f"""以下のコンテキスト情報を使用して、質問に答えてください。
コンテキスト情報に基づいて回答し、コンテキストにない情報は推測せずに「わかりません」と答えてください。

コンテキスト情報:
{context_text}

質問: {question}

回答:"""
                
                response = self.llm.invoke(prompt)
                answer = response.content if hasattr(response, 'content') else str(response)
                
                sources = [{"content": doc.page_content[:200], "metadata": doc.metadata} for doc in documents]
                
                results.append({
                    "question": question,
                    "answer": answer,
                    "num_sources": len(sources),
                    "sources": sources
                })
            except Exception as e:
                results.append({
                    "question": question,
                    "answer": f"エラーが発生しました: {str(e)}",
                    "num_sources": 0,
                    "sources": [],
                    "error": str(e)
                })
        
        return results

# COMMAND ----------

experiment_name = f"/Users/{w.current_user.me().user_name}/rag_chain_experiment"
try:
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        mlflow.create_experiment(experiment_name)
except Exception:
    mlflow.create_experiment(experiment_name)

mlflow.set_experiment(experiment_name)

with mlflow.start_run():
    from mlflow.models import ModelSignature
    from mlflow.types import DataType, Schema, ColSpec
    
    input_schema = Schema([ColSpec(DataType.string, "question")])
    output_schema = Schema([
        ColSpec(DataType.string, "question"),
        ColSpec(DataType.string, "answer"),
        ColSpec(DataType.integer, "num_sources"),
        ColSpec(DataType.string, "sources")
    ])
    signature = ModelSignature(inputs=input_schema, outputs=output_schema)
    
    input_example = {"question": "通勤手当はいくらまで支給されますか？"}
    sample_output = query_rag(input_example["question"])
    
    env_vars = {
        "CATALOG": CATALOG,
        "SCHEMA": SCHEMA,
        "VECTOR_INDEX_NAME": VECTOR_INDEX_NAME,
        "QUERY_EMBEDDING_MODEL": QUERY_EMBEDDING_MODEL,
        "LLM_ENDPOINT": LLM_ENDPOINT,
        "RETRIEVER_TOP_K": str(RETRIEVER_TOP_K)
    }
    
    import sys
    conda_env = {
        "channels": ["defaults", "conda-forge"],
        "dependencies": [
            f"python={sys.version.split()[0]}",
            "pip",
            {
                "pip": [
                    "langchain>=0.1.0",
                    "langchain-databricks>=0.1.0",
                    "databricks-langchain>=0.1.0",
                    "databricks-vectorsearch>=0.1.0",
                    "databricks-sdk>=0.1.0",
                    "mlflow>=2.0.0",
                    "pandas>=1.5.0",
                    "langchain-huggingface>=0.0.1",
                    "sentence-transformers>=2.0.0",
                    "sentencepiece>=0.1.0"
                ]
            }
        ]
    }
    
    mlflow.pyfunc.log_model(
        artifact_path="rag_model",
        python_model=RAGModel(),
        signature=signature,
        input_example=input_example,
        conda_env=conda_env,
        registered_model_name="commuting_allowance_rag_model"
    )
    
    mlflow.log_params(env_vars)
    mlflow.log_metric("num_sources", sample_output.get("num_sources", 0))
    mlflow.set_tag("embedding_model", QUERY_EMBEDDING_MODEL)
    mlflow.set_tag("llm", LLM_ENDPOINT)
    
    print(f"Model logged: {mlflow.active_run().info.run_id}")

# COMMAND ----------

endpoint_name = "commuting-allowance-rag-endpoint"
model_name = "commuting_allowance_rag_model"
model_version = 1

existing_endpoints = w.serving_endpoints.list()
endpoint_exists = any(ep.name == endpoint_name for ep in existing_endpoints)

served_model = ServedModelInput(
    name=model_name,
    model_name=model_name,
    model_version=str(model_version),
    workload_size="Small",
    scale_to_zero_enabled=True
)

if endpoint_exists:
    w.serving_endpoints.update_config(
        name=endpoint_name,
        served_models=[served_model]
    )
    print(f"Endpoint updated: {endpoint_name}")
else:
    config = EndpointCoreConfigInput(
        served_models=[served_model]
    )
    w.serving_endpoints.create(
        name=endpoint_name,
        config=config
    )
    print(f"Endpoint created: {endpoint_name}")

endpoint = w.serving_endpoints.get(endpoint_name)
print(f"Status: {endpoint.state}")
