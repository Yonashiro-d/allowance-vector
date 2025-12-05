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

from typing import Dict, Any
from pyspark.sql import SparkSession
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedModelInput
import mlflow
import mlflow.pyfunc

spark = SparkSession.builder.getOrCreate()
w = WorkspaceClient()

workspace_url = SparkSession.getActiveSession().conf.get("spark.databricks.workspaceUrl", None)

# COMMAND ----------

from rag_config import RAGConfig
from rag_client import RAGClient

config = RAGConfig()
print(f"CATALOG: {config.catalog}")
print(f"SCHEMA: {config.schema}")
print(f"VECTOR_INDEX_NAME: {config.vector_index_name}")
print(f"QUERY_EMBEDDING_MODEL: {config.query_embedding_model}")
print(f"LLM_ENDPOINT: {config.llm_endpoint}")
print(f"RETRIEVER_TOP_K: {config.retriever_top_k}")

# COMMAND ----------

rag_client = RAGClient(config)
rag_client._initialize()

# COMMAND ----------

def query_rag(question: str) -> Dict[str, Any]:
    return rag_client.query(question)

# COMMAND ----------

class RAGModel(mlflow.pyfunc.PythonModel):
    def __init__(self):
        self.vector_store = None
        self.llm = None
        self.retriever_top_k = 5
    
    def load_context(self, context):
        import traceback
        try:
            from rag_config import RAGConfig
            from rag_client import RAGClient
            config = RAGConfig()
            self.rag_client = RAGClient(config)
            self.rag_client._initialize()
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
            result = self.rag_client.query_chat_completion(messages)
            return result
        except Exception as e:
            return {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": f"エラーが発生しました: {str(e)}"
                    }
                }]
            }

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
    input_example = {
        "messages": [
            {"role": "user", "content": "通勤手当はいくらまで支給されますか？"}
        ]
    }
    
    env_vars = config.to_dict()
    
    import sys
    import os
    import tempfile
    import shutil
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        import importlib.util
        
        for module_name in ["rag_config", "rag_client"]:
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
            os.path.join(temp_dir, "rag_config.py"),
            os.path.join(temp_dir, "rag_client.py")
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
    
    mlflow.log_params(env_vars)
    mlflow.set_tag("task", "llm/v1/chat")
    mlflow.set_tag("embedding_model", config.query_embedding_model)
    mlflow.set_tag("llm", config.llm_endpoint)
    mlflow.set_tag("model_type", "chat_completion")
    
    print(f"Model logged: {mlflow.active_run().info.run_id}")

# COMMAND ----------

endpoint_name = "commuting-allowance-rag-endpoint"
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

served_model = ServedModelInput(
    name=model_name,
    model_name=model_name,
    model_version=str(model_version),
    workload_size="Small",
    scale_to_zero_enabled=True
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
    config = EndpointCoreConfigInput(
        name=endpoint_name,
        served_models=[served_model]
    )
    w.serving_endpoints.create(
        name=endpoint_name,
        config=config
    )
    print(f"Endpoint created: {endpoint_name}")

endpoint = w.serving_endpoints.get(endpoint_name)
print(f"Status: {endpoint.state}")
