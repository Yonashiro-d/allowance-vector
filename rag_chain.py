# Databricks notebook source
# MAGIC %md
# MAGIC # RAGチェーン構築
# MAGIC
# MAGIC Databricks Vector Searchを使用してRAGチェーンを構築します。

# COMMAND ----------

# MAGIC %pip install -U langchain langchain-core langchain-databricks databricks-langchain databricks-vectorsearch langchain-huggingface sentence-transformers sentencepiece mlflow databricks-sdk pypdf

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from typing import Dict, Any
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedModelInput
from pyspark.sql import SparkSession
import mlflow
import mlflow.pyfunc

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
# MAGIC ## ベクトルデータ準備の実行

# COMMAND ----------

def run_vector_preparation():
    """ベクトルデータ準備ノートブックを実行"""
    notebook_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
    current_dir = "/".join(notebook_path.split("/")[:-1]) if "/" in notebook_path else "."
    vector_prep_path = f"{current_dir}/vector_preparation"
    
    print(f"Running vector preparation notebook: {vector_prep_path}")
    
    try:
        dbutils.notebook.run(vector_prep_path, timeout_seconds=3600)
        print("Vector preparation completed successfully")
    except Exception as e:
        print(f"Error running vector preparation: {e}")
        raise

run_vector_preparation()

# COMMAND ----------

# MAGIC %md
# MAGIC ## RAGチェーン構築

# COMMAND ----------

chain_config = {
    "llm_model_serving_endpoint_name": "databricks-llama-4-maverick",
    "vector_search_endpoint_name": VECTOR_SEARCH_ENDPOINT,
    "vector_search_index": config.vector_index_name,
    "llm_prompt_template": """You are an assistant that answers questions. Use the following pieces of retrieved context to answer the question. Some pieces of context may be irrelevant, in which case you should not use them to form the answer.\n\nContext: {context}""",
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
    
    return rag_chain

rag_chain = build_rag_chain(chain_config, config)
print("RAG Chain created successfully")

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
    """MLflow実験をセットアップ"""
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

class RAGModel(mlflow.pyfunc.PythonModel):
    def __init__(self):
        self.rag_chain = None
    
    def load_context(self, context):
        import traceback
        import os
        
        try:
            if not os.environ.get("DATABRICKS_HOST"):
                workspace_url_env = os.environ.get("DATABRICKS_WORKSPACE_URL")
                if workspace_url_env:
                    os.environ["DATABRICKS_HOST"] = workspace_url_env
            
            if not os.environ.get("DATABRICKS_TOKEN"):
                api_token = os.environ.get("DATABRICKS_API_TOKEN")
                if api_token:
                    os.environ["DATABRICKS_TOKEN"] = api_token
            
            from rag_config import RAGConfig
            from langchain.chains import create_retrieval_chain
            from langchain.chains.combine_documents import create_stuff_documents_chain
            from langchain_core.prompts import ChatPromptTemplate
            from databricks_langchain import ChatDatabricks, DatabricksVectorSearch
            from langchain_huggingface import HuggingFaceEmbeddings
            
            config = RAGConfig()
            vector_search_endpoint = os.environ.get("VECTOR_SEARCH_ENDPOINT", VECTOR_SEARCH_ENDPOINT)
            
            chain_config_local = {
                "llm_model_serving_endpoint_name": chain_config["llm_model_serving_endpoint_name"],
                "vector_search_endpoint_name": vector_search_endpoint,
                "vector_search_index": config.vector_index_name,
                "llm_prompt_template": chain_config["llm_prompt_template"],
            }
            
            embedding_model = HuggingFaceEmbeddings(model_name=config.query_embedding_model)
            
            vector_store = DatabricksVectorSearch(
                index_name=chain_config_local["vector_search_index"],
                embedding=embedding_model,
                text_column="chunked_text",
                columns=["chunk_id", "chunked_text"]
            )
            
            llm = ChatDatabricks(
                endpoint=chain_config_local["llm_model_serving_endpoint_name"],
                extra_params={"temperature": 0.1}
            )
            
            retriever = vector_store.as_retriever(search_kwargs={"k": config.retriever_top_k})
            prompt = ChatPromptTemplate.from_template(chain_config_local["llm_prompt_template"])
            document_chain = create_stuff_documents_chain(llm, prompt)
            self.rag_chain = create_retrieval_chain(retriever, document_chain)
            
        except Exception as e:
            error_msg = f"Error loading context: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            raise RuntimeError(error_msg) from e
    
    def predict(self, context, model_input):
        import json
        
        if isinstance(model_input, str):
            try:
                model_input = json.loads(model_input)
            except:
                model_input = {"messages": [{"role": "user", "content": model_input}]}
        elif hasattr(model_input, 'iloc'):
            model_input = model_input.to_dict('records')[0] if len(model_input) > 0 else {}
        elif isinstance(model_input, list) and len(model_input) > 0:
            model_input = model_input[0]
        
        if not isinstance(model_input, dict):
            model_input = {"messages": [{"role": "user", "content": str(model_input)}]}
        
        messages = model_input.get("messages", [])
        if not messages:
            return {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": "メッセージが見つかりませんでした。"
                    }
                }]
            }
        
        last_message = messages[-1]
        question = last_message.get("content", "") if isinstance(last_message, dict) else str(last_message)
        
        if not question:
            return {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": "質問内容が空です。"
                    }
                }]
            }
        
        try:
            result = self.rag_chain.invoke({"input": question})
            answer = result.get("answer", "")
            
            return {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": answer
                    }
                }]
            }
        except Exception as e:
            return {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": f"エラーが発生しました: {str(e)}"
                    }
                }]
            }

with mlflow.start_run():
    input_example = {
        "messages": [
            {"role": "user", "content": "通勤手当はいくらまで支給されますか？"}
        ]
    }
    
    import sys
    import os
    import tempfile
    import shutil
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        import importlib.util
        
        for module_name in ["rag_config"]:
            try:
                module = __import__(module_name)
                module_file = module.__file__
                
                if module_file:
                    if module_file.endswith('.pyc'):
                        module_file = module_file[:-1]
                    
                    temp_file = os.path.join(temp_dir, f"{module_name}.py")
                    shutil.copy2(module_file, temp_file)
                    print(f"Copied {module_file} to {temp_file}")
            except Exception as e:
                print(f"Warning: Could not copy {module_name}: {e}")
        
        code_paths = [
            os.path.join(temp_dir, "rag_config.py")
        ]
        
        for code_path in code_paths:
            if not os.path.exists(code_path):
                raise FileNotFoundError(f"File not found: {code_path}")
        
        print(f"Using code_paths: {code_paths}")
    except Exception as e:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise
    
    conda_env = {
        "channels": ["defaults", "conda-forge"],
        "dependencies": [
            f"python={sys.version.split()[0]}",
            "pip",
            {
                "pip": [
                    "langchain>=0.1.0",
                    "langchain-core>=0.1.0",
                    "langchain-databricks>=0.1.0",
                    "databricks-langchain>=0.1.0",
                    "databricks-vectorsearch>=0.1.0",
                    "databricks-sdk>=0.1.0",
                    "databricks-feature-lookup==1.9",
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
        signature=None,
        input_example=input_example,
        conda_env=conda_env,
        code_paths=code_paths,
        registered_model_name="commuting_allowance_rag_model"
    )
    
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
    
    print(f"Model logged: {mlflow.active_run().info.run_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## MLflow Servingエンドポイント作成

# COMMAND ----------

endpoint_name = config.serving_endpoint_name
model_name = "commuting_allowance_rag_model"

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

