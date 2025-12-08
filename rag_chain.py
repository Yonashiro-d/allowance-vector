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
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedModelInput
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

# MAGIC %md
# MAGIC ## Deploy: MLflow PyFuncモデル登録とエンドポイント作成

# COMMAND ----------

class RAGModel(mlflow.pyfunc.PythonModel):
    #RAGチェーンをMLflow PyFuncモデルとして実装
    
    def __init__(self):
        self.rag_chain = None
        self.chain_config = None
    
    def load_context(self, context):
        import traceback
        import os
        
        try:
            # 環境変数の設定
            if not os.environ.get("DATABRICKS_HOST"):
                workspace_url_env = os.environ.get("DATABRICKS_WORKSPACE_URL")
                if workspace_url_env:
                    os.environ["DATABRICKS_HOST"] = workspace_url_env
            
            if not os.environ.get("DATABRICKS_TOKEN"):
                api_token = os.environ.get("DATABRICKS_API_TOKEN")
                if api_token:
                    os.environ["DATABRICKS_TOKEN"] = api_token
            
            # 依存関係のインポート
            from rag_config import RAGConfig
            from langchain.chains import create_retrieval_chain
            from langchain.chains.combine_documents import create_stuff_documents_chain
            from langchain_core.prompts import ChatPromptTemplate
            from databricks_langchain import ChatDatabricks, DatabricksVectorSearch
            from langchain_huggingface import HuggingFaceEmbeddings
            
            # 設定の読み込み
            config = RAGConfig()
            vector_search_endpoint = os.environ.get("VECTOR_SEARCH_ENDPOINT", "databricks-bge-large-en-endpoint")
            
            # chain_configの再構築
            self.chain_config = {
                "llm_model_serving_endpoint_name": config.llm_endpoint,
                "vector_search_endpoint_name": vector_search_endpoint,
                "vector_search_index": config.vector_index_name,
                "llm_prompt_template": """あなたは質問に答えるアシスタントです。取得したコンテキストの内容をもとに質問に答えてください。一部のコンテキストが無関係な場合、それを回答に利用しないでください。

コンテキスト:
{context}

質問: {input}""",
            }
            
            # RAGチェーンの構築
            embedding_model = HuggingFaceEmbeddings(model_name=config.query_embedding_model)
            
            vector_store = DatabricksVectorSearch(
                index_name=self.chain_config["vector_search_index"],
                embedding=embedding_model,
                text_column="chunked_text",
                columns=["chunk_id", "chunked_text"]
            )
            
            llm = ChatDatabricks(
                endpoint=self.chain_config["llm_model_serving_endpoint_name"],
                extra_params={"temperature": 0.1}
            )
            
            retriever = vector_store.as_retriever(search_kwargs={"k": config.retriever_top_k})
            prompt = ChatPromptTemplate.from_template(self.chain_config["llm_prompt_template"])
            document_chain = create_stuff_documents_chain(llm, prompt)
            self.rag_chain = create_retrieval_chain(retriever, document_chain)
            
            print("RAG chain loaded successfully in PyFunc model")
            
        except Exception as e:
            error_msg = f"Error loading context: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            raise RuntimeError(error_msg) from e
    
    def predict(self, context, model_input):
        #チャット補完API形式の入出力を処理
        import json
        
        # 入力の正規化
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
        
        # メッセージの抽出
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
        
        # 最後のユーザーメッセージを取得
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
        
        # RAGチェーンを実行
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

# COMMAND ----------

# MLflow PyFuncモデルとして登録
with mlflow.start_run(run_name="commuting-allowance-rag-model"):
    import sys
    import os
    import tempfile
    import shutil
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        # rag_config.pyを一時ディレクトリにコピー
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
    
    # conda環境の定義
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
    
    # 入力例の定義
    input_example = {
        "messages": [
            {"role": "user", "content": "通勤手当はいくらまで支給されますか？"}
        ]
    }
    
    # PyFuncモデルをログ
    logged_model_info = mlflow.pyfunc.log_model(
        artifact_path="rag_model",
        python_model=RAGModel(),
        signature=None,
        input_example=input_example,
        conda_env=conda_env,
        code_paths=code_paths,
        registered_model_name="commuting_allowance_rag_model"
    )
    
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
    mlflow.set_tag("model_type", "chat_completion")
    mlflow.set_tag("chain_type", "retrieval_chain")
    mlflow.set_tag("deployment_target", "playground")
    
    print(f"PyFunc model logged: {logged_model_info.model_uri}")
    print(f"Run ID: {mlflow.active_run().info.run_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## MLflow Servingエンドポイント作成

# COMMAND ----------

endpoint_name = config.serving_endpoint_name
model_name = "commuting_allowance_rag_model"

# モデル情報の取得
try:
    from mlflow.tracking import MlflowClient
    client = MlflowClient()
    latest_versions = client.get_latest_versions(model_name, stages=["None"])
    if latest_versions:
        latest_version = latest_versions[0]
        model_version = int(latest_version.version)
    else:
        raise ValueError("No model versions found")
except Exception as e:
    model_version = 1

# エンドポイントの存在確認
try:
    existing_endpoints = w.serving_endpoints.list()
    endpoint_exists = any(ep.name == endpoint_name for ep in existing_endpoints)
except Exception as e:
    endpoint_exists = False

if endpoint_exists:
    import time
    max_wait_time = 300  # 5分に延長
    wait_interval = 10  # 10秒間隔に変更
    elapsed_time = 0
    
    endpoint = w.serving_endpoints.get(endpoint_name)
    state = endpoint.state
    
    # エンドポイントの更新が完了するまで待機
    while hasattr(state, 'config_update') and state.config_update == "IN_PROGRESS":
        if elapsed_time >= max_wait_time:
            break
        time.sleep(wait_interval)
        elapsed_time += wait_interval
        try:
            endpoint = w.serving_endpoints.get(endpoint_name)
            state = endpoint.state
        except Exception as e:
            break

# 環境変数の設定
environment_vars = {}
if workspace_url:
    environment_vars["DATABRICKS_WORKSPACE_URL"] = workspace_url
    environment_vars["DATABRICKS_HOST"] = workspace_url

environment_vars["VECTOR_SEARCH_ENDPOINT"] = VECTOR_SEARCH_ENDPOINT

# Unity Catalog形式のentity_nameを準備（MLflow Deployments SDK用）
entity_name = f"{config.catalog}.{config.schema}.{model_name}"

# ServedModelInputの作成
served_model = ServedModelInput(
    name=f"{model_name}-{model_version}",
    model_name=model_name,
    model_version=str(model_version),
    workload_size="Small",
    scale_to_zero_enabled=True,
    environment_vars=environment_vars if environment_vars else {}
)

# エンドポイントの作成/更新
if endpoint_exists:
    endpoint = w.serving_endpoints.get(endpoint_name)
    state = endpoint.state
    
    # 更新中でないことを確認
    is_updating = hasattr(state, 'config_update') and state.config_update == "IN_PROGRESS"
    
    if not is_updating:
        try:
            w.serving_endpoints.update_config(
                name=endpoint_name,
                served_models=[served_model]
            )
            print(f"✅ Endpoint updated: {endpoint_name}")
        except Exception as e:
            error_msg = str(e)
            if "currently being updated" not in error_msg and "IN_PROGRESS" not in error_msg:
                # MLflow Deployments SDKを試す
                try:
                    import mlflow.deployments
                    deploy_client = mlflow.deployments.get_deploy_client("databricks")
                    config_dict = {
                        "served_entities": [
                            {
                                "entity_name": entity_name,
                                "entity_version": str(model_version),
                                "workload_size": "Small",
                                "scale_to_zero_enabled": True,
                                "environment_vars": environment_vars
                            }
                        ],
                        "traffic_config": {
                            "routes": [
                                {
                                    "served_model_name": f"{model_name}-{model_version}",
                                    "traffic_percentage": 100
                                }
                            ]
                        }
                    }
                    deploy_client.update_endpoint(endpoint=endpoint_name, config=config_dict)
                    print(f"✅ Endpoint updated: {endpoint_name}")
                except:
                    raise
else:
    try:
        endpoint_config = EndpointCoreConfigInput(
            name=endpoint_name,
            served_models=[served_model]
        )
        w.serving_endpoints.create(
            name=endpoint_name,
            config=endpoint_config
        )
        print(f"✅ Endpoint created: {endpoint_name}")
    except Exception as e:
        # MLflow Deployments SDKを試す
        try:
            import mlflow.deployments
            deploy_client = mlflow.deployments.get_deploy_client("databricks")
            config_dict = {
                "served_entities": [
                    {
                        "entity_name": entity_name,
                        "entity_version": str(model_version),
                        "workload_size": "Small",
                        "scale_to_zero_enabled": True,
                        "environment_vars": environment_vars
                    }
                ],
                "traffic_config": {
                    "routes": [
                        {
                            "served_model_name": f"{model_name}-{model_version}",
                            "traffic_percentage": 100
                        }
                    ]
                }
            }
            deploy_client.create_endpoint(endpoint=endpoint_name, config=config_dict)
            print(f"✅ Endpoint created: {endpoint_name}")
        except:
            raise

# エンドポイントの状態確認
try:
    endpoint = w.serving_endpoints.get(endpoint_name)
    print(f"✅ Endpoint Status: {endpoint.state}")
    if workspace_url:
        print(f"💡 Access: https://{workspace_url}/serving-endpoints/{endpoint_name}")
except Exception as e:
    print(f"❌ Error getting endpoint status: {e}")

# COMMAND ----------

