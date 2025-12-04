# Databricks notebook source
# MAGIC %md
# MAGIC # RAGチェーン構築とBot実装
# MAGIC
# MAGIC このノートブックでは、Databricks Vector Searchを使用してRAG（Retrieval-Augmented Generation）チェーンを構築し、
# MAGIC MLflowでBotとしてデプロイします。
# MAGIC
# MAGIC ## 概要
# MAGIC
# MAGIC - **Vector Store**: `DatabricksVectorSearch`を使用してベクトル検索を実装
# MAGIC - **LLM**: Databricks Foundation Model（databricks-dbrx-instruct）を使用
# MAGIC - **デプロイ**: MLflow Model Servingエンドポイントとしてデプロイ
# MAGIC
# MAGIC 参考: [genai-cookbook](https://github.com/databricks/genai-cookbook)

# COMMAND ----------

# MAGIC %pip install -U \
# MAGIC   langchain \
# MAGIC   langchain-databricks \
# MAGIC   databricks-langchain \
# MAGIC   databricks-vectorsearch \
# MAGIC   databricks-sdk \
# MAGIC   mlflow \
# MAGIC   pandas \
# MAGIC   sentence-transformers

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from typing import Dict, Any, List
from pyspark.sql import SparkSession
from databricks.sdk import WorkspaceClient
from databricks_langchain import ChatDatabricks, DatabricksVectorSearch
from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings
import mlflow
import mlflow.pyfunc

spark = SparkSession.builder.getOrCreate()
w = WorkspaceClient()

workspace_url = SparkSession.getActiveSession().conf.get("spark.databricks.workspaceUrl", None)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 設定

# COMMAND ----------

# 設定値
CATALOG = "hhhd_demo_itec"
SCHEMA = "allowance_payment_rules"
DELTA_TABLE_NAME = f"{CATALOG}.{SCHEMA}.commuting_allowance_vectors"
VECTOR_SEARCH_ENDPOINT = "databricks-bge-large-en-endpoint"
VECTOR_INDEX_NAME = f"{CATALOG}.{SCHEMA}.commuting_allowance_index"
QUERY_EMBEDDING_MODEL = "cl-nagoya/ruri-v3-310m"

# LLM設定
LLM_ENDPOINT = "databricks-dbrx-instruct"
LLM_TEMPERATURE = 0.1
LLM_MAX_TOKENS = 500

# RAG設定
RETRIEVER_TOP_K = 5

print(f"Catalog: {CATALOG}")
print(f"Schema: {SCHEMA}")
print(f"Table: {DELTA_TABLE_NAME}")
print(f"Vector Search Endpoint: {VECTOR_SEARCH_ENDPOINT}")
print(f"Vector Index: {VECTOR_INDEX_NAME}")
print(f"LLM Endpoint: {LLM_ENDPOINT}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## セクション1: RAG実装
# MAGIC
# MAGIC Databricks Vector Searchを使用してシンプルなRAGを実装します。

# COMMAND ----------

# MAGIC %md
# MAGIC ### Embeddingモデルの初期化

# COMMAND ----------

# Embeddingモデルを初期化（vector_preparation.pyと同じモデルを使用）
class SentenceTransformerEmbeddings(Embeddings):
    """SentenceTransformerをLangChainのEmbeddingsインターフェースに適合させる"""
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)
    
    def embed_query(self, text: str) -> List[float]:
        """クエリテキストをベクトル化"""
        return self.model.encode(text, show_progress_bar=False).tolist()
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """複数のテキストをベクトル化"""
        embeddings = self.model.encode(texts, show_progress_bar=False)
        return embeddings.tolist()

# Embeddingモデルを初期化
embeddings = SentenceTransformerEmbeddings(QUERY_EMBEDDING_MODEL)
print(f"Embedding model initialized: {QUERY_EMBEDDING_MODEL}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Vector Storeの設定

# COMMAND ----------

# DatabricksVectorSearchにEmbeddingモデルを指定
vector_store = DatabricksVectorSearch(
    index_name=VECTOR_INDEX_NAME,
    embedding=embeddings
)
print("Vector Store initialized with databricks-langchain")

# COMMAND ----------

# MAGIC %md
# MAGIC ### LLMの初期化

# COMMAND ----------

# Databricks Foundation ModelをLLMとして使用
llm = ChatDatabricks(
    endpoint_name=LLM_ENDPOINT,
    temperature=LLM_TEMPERATURE,
    max_tokens=LLM_MAX_TOKENS
)

print(f"LLM initialized: {LLM_ENDPOINT}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### RAGクエリ関数の実装
# MAGIC
# MAGIC Vector Searchでドキュメントを取得し、LLMに直接質問するシンプルな実装です。

# COMMAND ----------

# MAGIC %md
# MAGIC ### RAGクエリ関数の実装

# COMMAND ----------

def query_rag(question: str) -> Dict[str, Any]:
    """
    RAGを使用して質問に回答する関数（シンプルな実装）
    
    1. Vector Searchで関連ドキュメントを取得
    2. ドキュメントをコンテキストとして結合
    3. LLMに質問とコンテキストを送信
    
    Args:
        question: ユーザーの質問文字列
    
    Returns:
        以下のキーを含む辞書:
        - question: 質問内容
        - answer: LLMが生成した回答
        - sources: 参照元ドキュメントのリスト（各要素はcontentとmetadataを含む）
        - num_sources: 参照元ドキュメントの数
        - error: エラーが発生した場合のエラーメッセージ（オプション）
    """
    try:
        # 1. Vector Searchで関連ドキュメントを取得
        documents = vector_store.similarity_search(question, k=RETRIEVER_TOP_K)
        
        # 2. ドキュメントのテキストを結合してコンテキストを作成
        context = "\n\n".join([doc.page_content for doc in documents])
        
        # 3. プロンプトを作成
        prompt = f"""以下のコンテキスト情報を使用して、質問に答えてください。
コンテキスト情報に基づいて回答し、コンテキストにない情報は推測せずに「わかりません」と答えてください。

コンテキスト情報:
{context}

質問: {question}

回答:"""
        
        # 4. LLMに質問を送信
        response = llm.invoke(prompt)
        answer = response.content if hasattr(response, 'content') else str(response)
        
        # 5. 参照元を抽出
        sources = []
        for doc in documents:
            sources.append({
                "content": doc.page_content,
                "metadata": doc.metadata
            })
        
        return {
            "question": question,
            "answer": answer,
            "sources": sources,
            "num_sources": len(sources)
        }
    except Exception as e:
        return {
            "question": question,
            "answer": f"エラーが発生しました: {str(e)}",
            "sources": [],
            "num_sources": 0,
            "error": str(e)
        }

# COMMAND ----------

# MAGIC %md
# MAGIC ## セクション2: MLflowモデル登録
# MAGIC
# MAGIC RAGチェーンをMLflowモデルとして登録し、デプロイ可能な形式にします。

# COMMAND ----------

# MAGIC %md
# MAGIC ### MLflowモデルクラスの定義

# COMMAND ----------

class RAGModel(mlflow.pyfunc.PythonModel):
    """
    RAGチェーンをMLflowモデルとしてパッケージ化するクラス
    
    このクラスは、MLflowのPythonModelインターフェースを実装し、
    RAGチェーンをMLflowモデルとして保存・デプロイできるようにします。
    """
    
    def __init__(self):
        """モデルを初期化"""
        self.vector_store = None
        self.llm = None
        self.retriever_top_k = 5
    
    def load_context(self, context):
        """
        モデルのコンテキストをロード
        
        このメソッドは、モデルがロードされる際に呼び出されます。
        環境変数から設定を読み取り、RAGチェーンを初期化します。
        
        Args:
            context: MLflowのコンテキストオブジェクト
        """
        import os
        from databricks_langchain import ChatDatabricks, DatabricksVectorSearch
        from sentence_transformers import SentenceTransformer
        from langchain.embeddings.base import Embeddings
        
        # 設定を環境変数から取得
        catalog = os.environ.get("CATALOG", "hhhd_demo_itec")
        schema = os.environ.get("SCHEMA", "allowance_payment_rules")
        vector_index_name = os.environ.get("VECTOR_INDEX_NAME", f"{catalog}.{schema}.commuting_allowance_index")
        query_embedding_model = os.environ.get("QUERY_EMBEDDING_MODEL", "cl-nagoya/ruri-v3-310m")
        llm_endpoint = os.environ.get("LLM_ENDPOINT", "databricks-dbrx-instruct")
        llm_temperature = float(os.environ.get("LLM_TEMPERATURE", "0.1"))
        llm_max_tokens = int(os.environ.get("LLM_MAX_TOKENS", "500"))
        retriever_top_k = int(os.environ.get("RETRIEVER_TOP_K", "5"))
        
        # Embeddingモデルを初期化
        class SentenceTransformerEmbeddings(Embeddings):
            def __init__(self, model_name: str):
                self.model = SentenceTransformer(model_name)
            def embed_query(self, text: str):
                return self.model.encode(text, show_progress_bar=False).tolist()
            def embed_documents(self, texts: List[str]):
                embeddings = self.model.encode(texts, show_progress_bar=False)
                return embeddings.tolist()
        
        embeddings = SentenceTransformerEmbeddings(query_embedding_model)
        
        # Vector Storeを設定
        self.vector_store = DatabricksVectorSearch(
            index_name=vector_index_name,
            embedding=embeddings
        )
        
        # LLMを初期化
        self.llm = ChatDatabricks(
            endpoint_name=llm_endpoint,
            temperature=llm_temperature,
            max_tokens=llm_max_tokens
        )
        
        # 設定を保存
        self.retriever_top_k = retriever_top_k
    
    def predict(self, context, model_input):
        """
        予測を実行
        
        このメソッドは、モデルが呼び出される際に実行されます。
        文字列、リスト、またはpandas DataFrameを受け取ることができます。
        
        Args:
            context: MLflowのコンテキストオブジェクト
            model_input: 質問（文字列、リスト、またはpandas DataFrame）
        
        Returns:
            質問と回答を含む辞書のリスト
        """
        questions: List[str] = []
        if isinstance(model_input, str):
            questions = [model_input]
        elif isinstance(model_input, list):
            questions = model_input
        elif hasattr(model_input, 'iloc'):  # pandas DataFrame
            questions = model_input.iloc[:, 0].tolist()
        else:
            questions = [str(model_input)]
        
        results = []
        for question in questions:
            try:
                # 1. Vector Searchで関連ドキュメントを取得
                documents = self.vector_store.similarity_search(question, k=self.retriever_top_k)
                
                # 2. ドキュメントのテキストを結合してコンテキストを作成
                context = "\n\n".join([doc.page_content for doc in documents])
                
                # 3. プロンプトを作成
                prompt = f"""以下のコンテキスト情報を使用して、質問に答えてください。
コンテキスト情報に基づいて回答し、コンテキストにない情報は推測せずに「わかりません」と答えてください。

コンテキスト情報:
{context}

質問: {question}

回答:"""
                
                # 4. LLMに質問を送信
                response = self.llm.invoke(prompt)
                answer = response.content if hasattr(response, 'content') else str(response)
                
                # 5. 参照元を抽出
                sources = []
                for doc in documents:
                    sources.append({
                        "content": doc.page_content[:200],  # 最初の200文字のみ
                        "metadata": doc.metadata
                    })
                
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

print("RAGModel class defined")

# COMMAND ----------

# MAGIC %md
# MAGIC ### MLflowモデルの登録

# COMMAND ----------

# MLflow実験を設定
# 実験が存在しない場合は作成し、存在する場合は使用します
experiment_name = f"/Users/{w.current_user.me().user_name}/rag_chain_experiment"
try:
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id
except Exception as e:
    print(f"Experiment creation error: {e}")
    experiment_id = mlflow.create_experiment(experiment_name)

mlflow.set_experiment(experiment_name)

# モデルを登録
with mlflow.start_run():
    # モデルのシグネチャを明示的に定義
    # シグネチャは、モデルの入力と出力の形式を定義します
    from mlflow.models import ModelSignature
    from mlflow.types import DataType, Schema, ColSpec
    
    # 入力スキーマ: 文字列（質問）
    input_schema = Schema([
        ColSpec(DataType.string, "question")
    ])
    
    # 出力スキーマ: 辞書（回答、参照元など）
    output_schema = Schema([
        ColSpec(DataType.string, "question"),
        ColSpec(DataType.string, "answer"),
        ColSpec(DataType.integer, "num_sources"),
        ColSpec(DataType.string, "sources")  # JSON文字列として保存
    ])
    
    signature = ModelSignature(inputs=input_schema, outputs=output_schema)
    
    # 入力例を定義（モデルのテストとドキュメント用）
    input_example = {"question": "通勤手当はいくらまで支給されますか？"}
    
    # サンプル出力を取得（メトリクス用）
    sample_output = query_rag(input_example["question"])
    
    # 環境変数を設定（モデルロード時に使用）
    # これらの値は、モデルがデプロイされた環境で使用されます
    env_vars = {
        "CATALOG": CATALOG,
        "SCHEMA": SCHEMA,
        "VECTOR_SEARCH_ENDPOINT": VECTOR_SEARCH_ENDPOINT,
        "VECTOR_INDEX_NAME": VECTOR_INDEX_NAME,
        "QUERY_EMBEDDING_MODEL": QUERY_EMBEDDING_MODEL,
        "LLM_ENDPOINT": LLM_ENDPOINT,
        "LLM_TEMPERATURE": str(LLM_TEMPERATURE),
        "LLM_MAX_TOKENS": str(LLM_MAX_TOKENS),
        "RETRIEVER_TOP_K": str(RETRIEVER_TOP_K)
    }
    
    # 依存関係を明示的に定義
    # モデルがデプロイされる際に必要なパッケージを指定します
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
                    "sentence-transformers>=2.0.0"
                ]
            }
        ]
    }
    
    # モデルをMLflowにログ
    mlflow.pyfunc.log_model(
        artifact_path="rag_model",
        python_model=RAGModel(),
        signature=signature,
        input_example=input_example,
        conda_env=conda_env,
        registered_model_name="commuting_allowance_rag_model"
    )
    
    # 環境変数をパラメータとしてログ
    mlflow.log_params(env_vars)
    
    # メトリクスをログ（サンプル出力の参照元数）
    mlflow.log_metric("num_sources", sample_output.get("num_sources", 0))
    
    # モデルの説明をタグとして追加
    mlflow.set_tag("model_description", "RAG model for commuting allowance Q&A using Databricks Vector Search")
    mlflow.set_tag("embedding_model", QUERY_EMBEDDING_MODEL)
    mlflow.set_tag("llm", LLM_ENDPOINT)
    
    run_id = mlflow.active_run().info.run_id
    print(f"Model logged with run_id: {run_id}")

print("Model registered in MLflow")

# COMMAND ----------

# MAGIC %md
# MAGIC ## セクション3: エンドポイントデプロイメント
# MAGIC
# MAGIC MLflow Model Servingを使用して、RAGモデルをエンドポイントとしてデプロイします。

# COMMAND ----------

# MAGIC %md
# MAGIC ### MLflow Model Servingエンドポイントの作成

# COMMAND ----------

# エンドポイント名とモデル情報を定義
endpoint_name = "commuting-allowance-rag-endpoint"
model_name = "commuting_allowance_rag_model"
model_version = 1  # 使用するモデルバージョン（最新バージョンを使用する場合は変更が必要）

# エンドポイントが既に存在するか確認
existing_endpoints = w.serving_endpoints.list()
endpoint_exists = any(ep.name == endpoint_name for ep in existing_endpoints)

if endpoint_exists:
    print(f"Endpoint '{endpoint_name}' already exists. Updating...")
    # エンドポイントを更新
    w.serving_endpoints.update_config(
        name=endpoint_name,
        served_models=[
            {
                "name": model_name,
                "model_name": model_name,
                "model_version": str(model_version),
                "workload_size": "Small",
                "scale_to_zero_enabled": True
            }
        ]
    )
    print(f"Endpoint '{endpoint_name}' updated")
else:
    print(f"Creating new endpoint '{endpoint_name}'...")
    # 新しいエンドポイントを作成
    w.serving_endpoints.create(
        name=endpoint_name,
        config={
            "served_models": [
                {
                    "name": model_name,
                    "model_name": model_name,
                    "model_version": str(model_version),
                    "workload_size": "Small",
                    "scale_to_zero_enabled": True
                }
            ]
        }
    )
    print(f"Endpoint '{endpoint_name}' created")

# エンドポイントの状態を確認
endpoint = w.serving_endpoints.get(endpoint_name)
print(f"\nEndpoint Status: {endpoint.state}")
print(f"Endpoint URL: {endpoint.url if hasattr(endpoint, 'url') else 'N/A'}")

if workspace_url:
    print(f"View Endpoint: https://{workspace_url}/serving-endpoints/{endpoint_name}/")

# COMMAND ----------


