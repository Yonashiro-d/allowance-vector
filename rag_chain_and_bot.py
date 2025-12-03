# Databricks notebook source
# MAGIC %md
# MAGIC # RAGチェーン構築とBot実装
# MAGIC
# MAGIC ベクトルデータベースを使用してRAGチェーンを構築し、MLflowでBotとしてデプロイします。
# MAGIC とりあえず動かす
# MAGIC 参考: [genai-cookbook](https://github.com/databricks/genai-cookbook)

# COMMAND ----------

# MAGIC %pip install -U \
# MAGIC   langchain \
# MAGIC   langchain-databricks \
# MAGIC   databricks-langchain \
# MAGIC   langchain-community \
# MAGIC   databricks-vectorsearch \
# MAGIC   databricks-sdk \
# MAGIC   mlflow \
# MAGIC   requests \
# MAGIC   pandas \
# MAGIC   pydantic \
# MAGIC   sentence-transformers

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import os
from typing import List, Dict, Any, Optional, Union
from pyspark.sql import SparkSession
from databricks.sdk import WorkspaceClient
from databricks.vector_search.client import VectorSearchClient
from databricks_langchain import ChatDatabricks
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import BaseRetriever, Document
from sentence_transformers import SentenceTransformer
import mlflow
import mlflow.pyfunc
from pydantic import BaseModel, Field, field_validator

spark = SparkSession.builder.getOrCreate()
w = WorkspaceClient()
vsc = VectorSearchClient(disable_notice=True)

workspace_url = SparkSession.getActiveSession().conf.get("spark.databricks.workspaceUrl", None)

# 設定
CATALOG = "hhhd_demo_itec"
SCHEMA = "allowance_payment_rules"
DELTA_TABLE_NAME = f"{CATALOG}.{SCHEMA}.commuting_allowance_vectors"
VECTOR_SEARCH_ENDPOINT = "databricks-bge-large-en-endpoint"
VECTOR_INDEX_NAME = f"{CATALOG}.{SCHEMA}.commuting_allowance_index"
# 768次元のEmbeddingモデルを使用（インデックスと同じ次元）
# インデックスは cl-nagoya/ruri-v3-310m (768次元) で作成されているため、同じモデルを使用
QUERY_EMBEDDING_MODEL = "cl-nagoya/ruri-v3-310m"

print(f"Catalog: {CATALOG}")
print(f"Schema: {SCHEMA}")
print(f"Table: {DELTA_TABLE_NAME}")
print(f"Vector Search Endpoint: {VECTOR_SEARCH_ENDPOINT}")
print(f"Vector Index: {VECTOR_INDEX_NAME}")
print(f"Query Embedding Model: {QUERY_EMBEDDING_MODEL}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## セクション0: Pydanticスキーマ定義と変換関数

# COMMAND ----------

# MAGIC %md
# MAGIC ### 検索結果のスキーマ定義

# COMMAND ----------

class VectorSearchResult(BaseModel):
    """
    Vector Searchの検索結果を表す統一スキーマ
    """
    chunk_id: Optional[int] = Field(None, description="チャンクID")
    chunked_text: str = Field("", description="チャンクのテキスト内容")
    file_name: str = Field("", description="元のファイル名")
    score: float = Field(0.0, description="類似度スコア（距離の場合は負の値になる可能性がある）")
    
    @field_validator('score', mode='before')
    @classmethod
    def normalize_score(cls, v: Any) -> float:
        """スコアを正規化（distanceの場合は負の値になる可能性がある）"""
        if v is None:
            return 0.0
        if isinstance(v, (int, float)):
            return float(v)
        return 0.0
    
    @field_validator('chunked_text', 'file_name', mode='before')
    @classmethod
    def ensure_string(cls, v: Any) -> str:
        """文字列を保証"""
        if v is None:
            return ""
        return str(v)
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            "chunk_id": self.chunk_id,
            "chunked_text": self.chunked_text,
            "file_name": self.file_name,
            "score": self.score
        }

# COMMAND ----------

# MAGIC %md
# MAGIC ### 複数フォーマットから統一形式への変換関数

# COMMAND ----------

def normalize_vector_search_result(result: Any) -> Optional[VectorSearchResult]:
    """
    Databricks Vector Searchが返す様々なフォーマットの結果を統一形式に変換
    
    対応フォーマット:
    1. 辞書形式: {"chunk_id": 1, "chunked_text": "...", "file_name": "...", "score": 0.9}
    2. リスト形式: [chunk_id, chunked_text, file_name, score]
    3. 距離ベース: {"chunk_id": 1, "chunked_text": "...", "distance": 0.1}
    
    Args:
        result: 変換対象の結果（辞書、リスト、またはその他の形式）
    
    Returns:
        VectorSearchResultオブジェクト、または変換できない場合はNone
    """
    if result is None:
        return None
    
    # パターン1: 辞書形式
    if isinstance(result, dict):
        try:
            # scoreまたはdistanceを取得
            score = result.get("score")
            if score is None:
                score = result.get("distance", 0.0)
            
            return VectorSearchResult(
                chunk_id=result.get("chunk_id"),
                chunked_text=result.get("chunked_text", ""),
                file_name=result.get("file_name", ""),
                score=score
            )
        except Exception as e:
            print(f"Warning: Failed to parse dict result: {e}")
            return None
    
    # パターン2: リスト形式 [chunk_id, chunked_text, file_name, score]
    if isinstance(result, list) and len(result) >= 3:
        try:
            return VectorSearchResult(
                chunk_id=result[0] if len(result) > 0 and result[0] is not None else None,
                chunked_text=result[1] if len(result) > 1 else "",
                file_name=result[2] if len(result) > 2 else "",
                score=result[3] if len(result) > 3 else 0.0
            )
        except Exception as e:
            print(f"Warning: Failed to parse list result: {e}")
            return None
    
    return None

def parse_vector_search_response(query_result: Any) -> List[VectorSearchResult]:
    """
    Databricks Vector Searchのレスポンスを解析して統一形式のリストに変換
    辞書形式のみに対応
    
    Args:
        query_result: Vector Search APIのレスポンス（辞書形式）
    
    Returns:
        VectorSearchResultのリスト
    """
    results = []
    
    if query_result is None:
        return results
    
    # パターン1: 辞書形式（Databricks APIの標準形式）
    # {"manifest": {...}, "result": {"data_array": [[...], [...]], "row_count": 2}}
    if isinstance(query_result, dict):
        # Databricks APIの標準形式: {"manifest": {...}, "result": {"data_array": [...]}}
        if "result" in query_result and isinstance(query_result["result"], dict):
            result_data = query_result["result"]
            if "data_array" in result_data and isinstance(result_data["data_array"], list):
                # manifestからカラム順序を取得
                column_order = None
                if "manifest" in query_result and isinstance(query_result["manifest"], dict):
                    if "columns" in query_result["manifest"]:
                        columns = query_result["manifest"]["columns"]
                        column_order = []
                        for col in columns:
                            if isinstance(col, dict):
                                column_order.append(col.get("name"))
                            elif hasattr(col, "name"):
                                column_order.append(col.name)
                            else:
                                column_order.append(str(col))
                
                # デフォルトのカラム順序: [chunk_id, chunked_text, file_name, score]
                if column_order is None:
                    column_order = ["chunk_id", "chunked_text", "file_name", "score"]
                
                # data_arrayの各行を辞書形式に変換
                for row in result_data["data_array"]:
                    if isinstance(row, list) and len(row) >= 3:
                        row_dict = {}
                        for i, col_name in enumerate(column_order):
                            if i < len(row) and col_name:
                                row_dict[col_name] = row[i]
                        # scoreが含まれていない場合は0.0を設定
                        if "score" not in row_dict:
                            row_dict["score"] = row[3] if len(row) > 3 else 0.0
                        normalized = normalize_vector_search_result(row_dict)
                        if normalized is not None:
                            results.append(normalized)
                return results
        
        # 単一の結果辞書
        if "chunk_id" in query_result or "chunked_text" in query_result:
            normalized = normalize_vector_search_result(query_result)
            if normalized is not None:
                results.append(normalized)
            return results
        
        # resultsキーにリストが含まれている場合（各要素は辞書形式）
        if "results" in query_result and isinstance(query_result["results"], list):
            for item in query_result["results"]:
                if isinstance(item, dict):
                    normalized = normalize_vector_search_result(item)
                    if normalized is not None:
                        results.append(normalized)
            return results
    
    # パターン2: オブジェクト形式（result属性を持つ）
    if hasattr(query_result, 'result') and query_result.result:
        data = query_result.result
        # data_array形式（Databricks APIの標準形式）
        if hasattr(data, 'data_array') and data.data_array:
            # manifestからカラム順序を取得
            column_order = None
            if hasattr(query_result, 'manifest') and query_result.manifest:
                if hasattr(query_result.manifest, 'columns'):
                    column_order = []
                    for col in query_result.manifest.columns:
                        if hasattr(col, 'name'):
                            column_order.append(col.name)
                        elif isinstance(col, dict):
                            column_order.append(col.get("name"))
                        else:
                            column_order.append(str(col))
            
            if column_order is None:
                column_order = ["chunk_id", "chunked_text", "file_name", "score"]
            
            # data_arrayの各行を辞書形式に変換
            for row in data.data_array:
                if isinstance(row, list) and len(row) >= 3:
                    row_dict = {}
                    for i, col_name in enumerate(column_order):
                        if i < len(row) and col_name:
                            row_dict[col_name] = row[i]
                    if "score" not in row_dict:
                        row_dict["score"] = row[3] if len(row) > 3 else 0.0
                    normalized = normalize_vector_search_result(row_dict)
                    if normalized is not None:
                        results.append(normalized)
            return results
    
    return results

# COMMAND ----------

# MAGIC %md
# MAGIC ## セクション1: クエリ用Embeddingモデルの実装

# COMMAND ----------

# MAGIC %md
# MAGIC ### Embeddingモデルの初期化

# COMMAND ----------

# クエリ用Embeddingモデルを初期化
# 768次元のモデルを使用（SentenceTransformerを使用）
query_embedding_model = SentenceTransformer(QUERY_EMBEDDING_MODEL)
print(f"Query Embedding Model: {QUERY_EMBEDDING_MODEL}")
print(f"Embedding Dimension: {query_embedding_model.get_sentence_embedding_dimension()}")

# LangChain用のラッパーを作成
class SentenceTransformerEmbeddings:
    """SentenceTransformerをLangChainのEmbeddingsインターフェースに適合させる"""
    def __init__(self, model):
        self.model = model
    
    def embed_query(self, text: str) -> List[float]:
        """クエリテキストをベクトル化"""
        return self.model.encode(text, show_progress_bar=False).tolist()
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """複数のテキストをベクトル化"""
        embeddings = self.model.encode(texts, show_progress_bar=False)
        return embeddings.tolist()

query_embeddings = SentenceTransformerEmbeddings(query_embedding_model)

# COMMAND ----------

# MAGIC %md
# MAGIC ## セクション2: Vector Searchクエリ機能

# COMMAND ----------

# MAGIC %md
# MAGIC ### Vector Searchインデックスの確認

# COMMAND ----------

# インデックスの状態を確認
# VectorSearchIndexオブジェクトにはstatus属性がないため、インデックスが取得できれば使用可能とみなす
try:
    # インデックスオブジェクトを取得（クエリ用）
    index = vsc.get_index(
        endpoint_name=VECTOR_SEARCH_ENDPOINT,
        index_name=VECTOR_INDEX_NAME
    )
    print(f"✓ Index retrieved successfully")
    print(f"  Index Name: {VECTOR_INDEX_NAME}")
    print(f"  Endpoint: {VECTOR_SEARCH_ENDPOINT}")
    print("  Note: Index status check skipped (VectorSearchIndex object doesn't have status attribute)")
    print("  If queries fail, ensure the index is ONLINE in the Databricks UI")
    
except Exception as e:
    print(f"✗ Error getting index: {e}")
    print("\nTroubleshooting:")
    print(f"  1. Check if endpoint '{VECTOR_SEARCH_ENDPOINT}' exists")
    print(f"  2. Check if index '{VECTOR_INDEX_NAME}' exists")
    print(f"  3. Verify you have permissions to access the index")
    print(f"  4. Ensure the index was created in vector_preparation.py")
    if workspace_url:
        index_name_only = VECTOR_INDEX_NAME.split('.')[-1]
        print(f"  5. Check index status in UI: https://{workspace_url}/explore/data/{CATALOG}/{SCHEMA}/{index_name_only}")
    raise

# COMMAND ----------

# MAGIC %md
# MAGIC ### Vector Searchクエリ関数の実装

# COMMAND ----------

def search_similar_chunks(
    query: str,
    top_k: int = 5,
    filters: Optional[Dict[str, Any]] = None
) -> List[VectorSearchResult]:
    """
    Vector Searchを使用して類似チャンクを検索
    
    Args:
        query: 検索クエリ
        top_k: 返すチャンクの数（デフォルト: 5）
        filters: フィルタ条件（オプション）
    
    Returns:
        VectorSearchResultのリスト（統一形式）
    """
    # クエリをベクトル化
    query_vector = query_embeddings.embed_query(query)
    
    # Vector Searchで検索
    try:
        # インデックスオブジェクトを取得
        index = vsc.get_index(
            endpoint_name=VECTOR_SEARCH_ENDPOINT,
            index_name=VECTOR_INDEX_NAME
        )
        
        # インデックスの次元を確認（可能な場合）
        index_dimension = None
        if hasattr(index, 'embedding_dimension'):
            index_dimension = index.embedding_dimension
        elif hasattr(index, 'primary_key') and hasattr(index, 'embedding_vector_column'):
            # テーブルから次元を取得
            try:
                sample_df = spark.sql(f"SELECT embedding FROM {DELTA_TABLE_NAME} LIMIT 1")
                if sample_df.count() > 0:
                    sample_embedding = sample_df.collect()[0]['embedding']
                    index_dimension = len(sample_embedding) if sample_embedding else None
            except:
                pass
        
        # ベクトル次元の確認
        if index_dimension and len(query_vector) != index_dimension:
            print(f"Warning: Query vector dimension ({len(query_vector)}) doesn't match index dimension ({index_dimension})")
            print(f"Note: Using query vector as-is. Results may be inaccurate.")
        
        # フィルタ条件を文字列形式に変換（SQL WHERE句形式）
        filters_str = None
        if filters:
            # 辞書形式のフィルタをSQL WHERE句形式の文字列に変換
            filter_parts = []
            for key, value in filters.items():
                if isinstance(value, str):
                    filter_parts.append(f"{key} = '{value}'")
                else:
                    filter_parts.append(f"{key} = {value}")
            filters_str = " AND ".join(filter_parts)
        
        # インデックスオブジェクトのsimilarity_searchメソッドを使用（ドキュメントに基づく）
        query_result = index.similarity_search(
            query_vector=query_vector,
            columns=["chunk_id", "chunked_text", "file_name"],
            num_results=top_k,
            filters=filters_str
        )
        
        # 結果を解析（辞書形式のみに対応）
        if query_result is not None:
            return parse_vector_search_response(query_result)
        else:
            print("Warning: Query returned None")
            return []
        
    except Exception as e:
        print(f"Error in vector search: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return []

# COMMAND ----------

# MAGIC %md
# MAGIC ## セクション3: RAG Chain構築

# COMMAND ----------

# MAGIC %md
# MAGIC ### LangChain Databricks Vector Storeの設定

# COMMAND ----------

# Databricks Vector SearchをVector Storeとして設定
# databricks-langchainパッケージのDatabricksVectorSearchを使用（ドキュメントに基づく）
try:
    from databricks_langchain import DatabricksVectorSearch
    
    vector_store = DatabricksVectorSearch(
        index_name=VECTOR_INDEX_NAME
    )
    print("Vector Store initialized with databricks-langchain")
except Exception as e:
    print(f"Warning: Could not initialize DatabricksVectorSearch from databricks-langchain: {e}")
    print("Using custom retriever instead")
    vector_store = None

# COMMAND ----------

# MAGIC %md
# MAGIC ### LLMの初期化

# COMMAND ----------

# Databricks Foundation ModelをLLMとして使用
# デフォルトはdatabricks-dbrx-instructを使用
llm = ChatDatabricks(
    endpoint_name="databricks-dbrx-instruct",
    temperature=0.1,
    max_tokens=500
)

print("LLM initialized")

# COMMAND ----------

# MAGIC %md
# MAGIC ### プロンプトテンプレートの作成

# COMMAND ----------

# プロンプトテンプレートを定義
prompt_template = """以下のコンテキスト情報を使用して、質問に答えてください。
コンテキスト情報に基づいて回答し、コンテキストにない情報は推測せずに「わかりません」と答えてください。

コンテキスト情報:
{context}

質問: {question}

回答:"""

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

print("Prompt template created")

# COMMAND ----------

# MAGIC %md
# MAGIC ### RAG Chainの構築

# COMMAND ----------

# カスタムリトリーバーを作成（Vector Searchを直接使用）
class CustomRetriever(BaseRetriever):
    def __init__(self, search_func, top_k=5):
        super().__init__()
        self.search_func = search_func
        self.top_k = top_k
    
    def _get_relevant_documents(self, query: str):
        # search_similar_chunksはVectorSearchResultのリストを返す
        results = self.search_func(query, top_k=self.top_k)
        documents = []
        for result in results:
            # VectorSearchResultオブジェクトからDocumentを作成
            doc = Document(
                page_content=result.chunked_text,
                metadata={
                    "chunk_id": result.chunk_id,
                    "file_name": result.file_name,
                    "score": result.score
                }
            )
            documents.append(doc)
        return documents
    
    async def _aget_relevant_documents(self, query: str):
        # 非同期サポート（オプション）
        return self._get_relevant_documents(query)

# RetrievalQAチェーンを作成
if vector_store is not None:
    try:
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(
                search_kwargs={"k": 5}
            ),
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=True
        )
        print("RAG Chain created with DatabricksVectorSearch")
    except Exception as e:
        print(f"Warning: Could not use DatabricksVectorSearch: {e}")
        # カスタムリトリーバーを使用
        custom_retriever = CustomRetriever(search_similar_chunks, top_k=5)
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=custom_retriever,
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=True
        )
        print("RAG Chain created with custom retriever")
else:
    # カスタムリトリーバーを使用
    custom_retriever = CustomRetriever(search_similar_chunks, top_k=5)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=custom_retriever,
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=True
    )
    print("RAG Chain created with custom retriever")

# COMMAND ----------

# MAGIC %md
# MAGIC ### RAGクエリ関数の実装

# COMMAND ----------

def query_rag(question: str) -> Dict[str, Any]:
    """
    RAGチェーンを使用して質問に回答
    
    Args:
        question: ユーザーの質問
    
    Returns:
        回答と参照元を含む辞書
    """
    try:
        # RAGチェーンで質問を実行
        result = qa_chain.invoke({"query": question})
        
        # 結果を整形
        answer = result.get("result", "")
        source_documents = result.get("source_documents", [])
        
        # 参照元を抽出
        sources = []
        for doc in source_documents:
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
# MAGIC ## セクション4: MLflowモデル登録

# COMMAND ----------

# MAGIC %md
# MAGIC ### MLflowモデルクラスの定義

# COMMAND ----------

class RAGModel(mlflow.pyfunc.PythonModel):
    """
    RAGチェーンをMLflowモデルとしてパッケージ化
    """
    
    def __init__(self):
        self.qa_chain = None
        self.query_embeddings = None
    
    def load_context(self, context):
        """モデルのコンテキストをロード"""
        import os
        from databricks_langchain import ChatDatabricks
        from langchain.chains import RetrievalQA
        from langchain.prompts import PromptTemplate
        
        # 設定を環境変数から取得
        catalog = os.environ.get("CATALOG", "hhhd_demo_itec")
        schema = os.environ.get("SCHEMA", "allowance_payment_rules")
        vector_search_endpoint = os.environ.get("VECTOR_SEARCH_ENDPOINT", "databricks-bge-large-en-endpoint")
        vector_index_name = os.environ.get("VECTOR_INDEX_NAME", f"{catalog}.{schema}.commuting_allowance_index")
        query_embedding_model = os.environ.get("QUERY_EMBEDDING_MODEL", "databricks-gte-large-en")
        
        # VectorSearchResultクラスを定義（load_context内で使用）
        class VectorSearchResult(BaseModel):
            chunk_id: Optional[int] = Field(None)
            chunked_text: str = Field("")
            file_name: str = Field("")
            score: float = Field(0.0)
            
            @field_validator('score', mode='before')
            @classmethod
            def normalize_score(cls, v):
                if v is None:
                    return 0.0
                if isinstance(v, (int, float)):
                    return float(v)
                return 0.0
            
            @field_validator('chunked_text', 'file_name', mode='before')
            @classmethod
            def ensure_string(cls, v):
                if v is None:
                    return ""
                return str(v)
        
        # Embeddingモデルを初期化（768次元のSentenceTransformerを使用）
        self.query_embedding_model = SentenceTransformer(query_embedding_model)
        
        # LangChain用のラッパー
        class SentenceTransformerEmbeddings:
            def __init__(self, model):
                self.model = model
            def embed_query(self, text: str):
                return self.model.encode(text, show_progress_bar=False).tolist()
            def embed_documents(self, texts: List[str]):
                embeddings = self.model.encode(texts, show_progress_bar=False)
                return embeddings.tolist()
        
        self.query_embeddings = SentenceTransformerEmbeddings(self.query_embedding_model)
        
        # Vector Storeを設定（エラーハンドリング付き）
        try:
            from databricks_langchain import DatabricksVectorSearch
            
            vector_store = DatabricksVectorSearch(
                index_name=vector_index_name
            )
        except Exception as e:
            print(f"Warning: Could not initialize DatabricksVectorSearch: {e}")
            vector_store = None
        
        # LLMを初期化
        llm = ChatDatabricks(
            endpoint_name="databricks-dbrx-instruct",
            temperature=0.1,
            max_tokens=500
        )
        
        # プロンプトテンプレート
        prompt_template = """以下のコンテキスト情報を使用して、質問に答えてください。
コンテキスト情報に基づいて回答し、コンテキストにない情報は推測せずに「わかりません」と答えてください。

コンテキスト情報:
{context}

質問: {question}

回答:"""
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # カスタムリトリーバーを作成
        from databricks.vector_search.client import VectorSearchClient
        from langchain.schema import BaseRetriever, Document
        
        class ModelCustomRetriever(BaseRetriever):
            def __init__(self, embeddings, endpoint, index_name, top_k=5):
                super().__init__()
                self.embeddings = embeddings
                self.endpoint = endpoint
                self.index_name = index_name
                self.top_k = top_k
                self.vsc = VectorSearchClient(disable_notice=True)
            
            def _get_relevant_documents(self, query: str):
                try:
                    query_vector = self.embeddings.embed_query(query)
                    
                    # インデックスオブジェクトを取得
                    index = self.vsc.get_index(
                        endpoint_name=self.endpoint,
                        index_name=self.index_name
                    )
                    
                    # インデックスオブジェクトのsimilarity_searchメソッドを使用（ドキュメントに基づく）
                    query_result = index.similarity_search(
                        query_vector=query_vector,
                        columns=["chunk_id", "chunked_text", "file_name"],
                        num_results=self.top_k
                    )
                    
                    # 統一形式に変換（load_context内ではグローバル関数にアクセスできないため、
                    # ここでも変換ロジックを定義）
                    def normalize_result(result):
                        """辞書形式のみに対応"""
                        if result is None:
                            return None
                        if isinstance(result, dict):
                            score = result.get("score")
                            if score is None:
                                score = result.get("distance", 0.0)
                            # metadataから取得を試みる（LangChain形式の場合）
                            chunk_id = result.get("chunk_id")
                            chunked_text = result.get("chunked_text") or result.get("text") or result.get("page_content", "")
                            file_name = result.get("file_name")
                            if not chunk_id and "metadata" in result and isinstance(result["metadata"], dict):
                                chunk_id = result["metadata"].get("chunk_id")
                                file_name = result["metadata"].get("file_name") or file_name
                            return VectorSearchResult(
                                chunk_id=chunk_id,
                                chunked_text=chunked_text,
                                file_name=file_name,
                                score=score
                            )
                        return None
                    
                    def parse_response(query_result):
                        """辞書形式のみに対応"""
                        results = []
                        if query_result is None:
                            return results
                        # パターン1: 辞書形式（Databricks APIの標準形式）
                        if isinstance(query_result, dict):
                            # Databricks APIの標準形式: {"manifest": {...}, "result": {"data_array": [...]}}
                            if "result" in query_result and isinstance(query_result["result"], dict):
                                result_data = query_result["result"]
                                if "data_array" in result_data and isinstance(result_data["data_array"], list):
                                    # manifestからカラム順序を取得
                                    column_order = None
                                    if "manifest" in query_result and isinstance(query_result["manifest"], dict):
                                        if "columns" in query_result["manifest"]:
                                            columns = query_result["manifest"]["columns"]
                                            column_order = []
                                            for col in columns:
                                                if isinstance(col, dict):
                                                    column_order.append(col.get("name"))
                                                elif hasattr(col, "name"):
                                                    column_order.append(col.name)
                                                else:
                                                    column_order.append(str(col))
                                    if column_order is None:
                                        column_order = ["chunk_id", "chunked_text", "file_name", "score"]
                                    
                                    # data_arrayの各行を辞書形式に変換
                                    for row in result_data["data_array"]:
                                        if isinstance(row, list) and len(row) >= 3:
                                            row_dict = {}
                                            for i, col_name in enumerate(column_order):
                                                if i < len(row) and col_name:
                                                    row_dict[col_name] = row[i]
                                            if "score" not in row_dict:
                                                row_dict["score"] = row[3] if len(row) > 3 else 0.0
                                            normalized = normalize_result(row_dict)
                                            if normalized is not None:
                                                results.append(normalized)
                                    return results
                            
                            # 単一の結果辞書
                            if "chunk_id" in query_result or "chunked_text" in query_result:
                                normalized = normalize_result(query_result)
                                if normalized is not None:
                                    results.append(normalized)
                                return results
                            
                            # resultsキーにリストが含まれている場合（各要素は辞書形式）
                            if "results" in query_result and isinstance(query_result["results"], list):
                                for item in query_result["results"]:
                                    if isinstance(item, dict):
                                        normalized = normalize_result(item)
                                        if normalized is not None:
                                            results.append(normalized)
                                return results
                        
                        # パターン2: オブジェクト形式（result属性を持つ）
                        if hasattr(query_result, 'result') and query_result.result:
                            data = query_result.result
                            # data_array形式（Databricks APIの標準形式）
                            if hasattr(data, 'data_array') and data.data_array:
                                # manifestからカラム順序を取得
                                column_order = None
                                if hasattr(query_result, 'manifest') and query_result.manifest:
                                    if hasattr(query_result.manifest, 'columns'):
                                        column_order = []
                                        for col in query_result.manifest.columns:
                                            if hasattr(col, 'name'):
                                                column_order.append(col.name)
                                            elif isinstance(col, dict):
                                                column_order.append(col.get("name"))
                                            else:
                                                column_order.append(str(col))
                                if column_order is None:
                                    column_order = ["chunk_id", "chunked_text", "file_name", "score"]
                                
                                # data_arrayの各行を辞書形式に変換
                                for row in data.data_array:
                                    if isinstance(row, list) and len(row) >= 3:
                                        row_dict = {}
                                        for i, col_name in enumerate(column_order):
                                            if i < len(row) and col_name:
                                                row_dict[col_name] = row[i]
                                        if "score" not in row_dict:
                                            row_dict["score"] = row[3] if len(row) > 3 else 0.0
                                        normalized = normalize_result(row_dict)
                                        if normalized is not None:
                                            results.append(normalized)
                                return results
                        
                        return results
                    
                    normalized_results = parse_response(query_result)
                    
                    # VectorSearchResultからDocumentに変換
                    documents = []
                    for result in normalized_results:
                        doc = Document(
                            page_content=result.chunked_text,
                            metadata={
                                "chunk_id": result.chunk_id,
                                "file_name": result.file_name,
                                "score": result.score
                            }
                        )
                        documents.append(doc)
                    return documents
                except Exception as e:
                    print(f"Error in vector search: {e}")
                    return []
        
        # RAGチェーンを作成
        if vector_store is not None:
            try:
                self.qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=vector_store.as_retriever(
                        search_kwargs={"k": 5}
                    ),
                    chain_type_kwargs={"prompt": PROMPT},
                    return_source_documents=True
                )
            except Exception:
                custom_retriever = ModelCustomRetriever(
                    self.query_embeddings,
                    vector_search_endpoint,
                    vector_index_name,
                    top_k=5
                )
                self.qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=custom_retriever,
                    chain_type_kwargs={"prompt": PROMPT},
                    return_source_documents=True
                )
        else:
            custom_retriever = ModelCustomRetriever(
                self.query_embeddings,
                vector_search_endpoint,
                vector_index_name,
                top_k=5
            )
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=custom_retriever,
                chain_type_kwargs={"prompt": PROMPT},
                return_source_documents=True
            )
    
    def predict(self, context, model_input):
        """予測を実行"""
        questions = []
        if isinstance(model_input, str):
            questions = [model_input]
        if isinstance(model_input, list):
            questions = model_input
        if not isinstance(model_input, str) and not isinstance(model_input, list):
            if hasattr(model_input, 'iloc'):
                questions = model_input.iloc[:, 0].tolist()
            else:
                questions = [str(model_input)]
        
        results = []
        for question in questions:
            try:
                result = self.qa_chain.invoke({"query": question})
                answer = result.get("result", "")
                source_documents = result.get("source_documents", [])
                
                sources = []
                for doc in source_documents:
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

# MLflowを設定
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
    
    # 入力例を定義
    input_example = {"question": "通勤手当はいくらまで支給されますか？"}
    
    # サンプル出力を取得（メトリクス用）
    sample_output = query_rag(input_example["question"])
    
    # 環境変数を設定（モデルロード時に使用）
    env_vars = {
        "CATALOG": CATALOG,
        "SCHEMA": SCHEMA,
        "VECTOR_SEARCH_ENDPOINT": VECTOR_SEARCH_ENDPOINT,
        "VECTOR_INDEX_NAME": VECTOR_INDEX_NAME,
        "QUERY_EMBEDDING_MODEL": QUERY_EMBEDDING_MODEL
    }
    
    # 依存関係を明示的に定義
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
                    "langchain-community>=0.0.20",
                    "databricks-vectorsearch>=0.1.0",
                    "databricks-sdk>=0.1.0",
                    "mlflow>=2.0.0",
                    "pydantic>=2.0.0",
                    "requests>=2.28.0",
                    "pandas>=1.5.0"
                ]
            }
        ]
    }
    
    # モデルをログ
    mlflow.pyfunc.log_model(
        artifact_path="rag_model",
        python_model=RAGModel(),
        signature=signature,
        input_example=input_example,
        conda_env=conda_env,
        registered_model_name="commuting_allowance_rag_model"
    )
    
    # 環境変数をログ
    mlflow.log_params(env_vars)
    
    # メトリクスをログ
    mlflow.log_metric("num_sources", sample_output.get("num_sources", 0))
    
    # モデルの説明を追加
    mlflow.set_tag("model_description", "RAG model for commuting allowance Q&A using Databricks Vector Search")
    mlflow.set_tag("embedding_model", QUERY_EMBEDDING_MODEL)
    mlflow.set_tag("llm", "databricks-dbrx-instruct")
    
    run_id = mlflow.active_run().info.run_id
    print(f"Model logged with run_id: {run_id}")

print("Model registered in MLflow")

# COMMAND ----------

# MAGIC %md
# MAGIC ## セクション5: エンドポイントデプロイメント

# COMMAND ----------

# MAGIC %md
# MAGIC ### MLflow Model Servingエンドポイントの作成

# COMMAND ----------

# エンドポイント名を定義
endpoint_name = "commuting-allowance-rag-endpoint"
model_name = "commuting_allowance_rag_model"
model_version = 1  # 最新バージョンを使用

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

# MAGIC %md
# MAGIC ## まとめ

# COMMAND ----------

# MAGIC %md
# MAGIC ### 実装完了項目
# MAGIC
# MAGIC ✅ **クエリ用Embeddingモデル**: `databricks-gte-large-en`を使用
# MAGIC
# MAGIC ✅ **Vector Searchクエリ機能**: 類似チャンク検索を実装
# MAGIC
# MAGIC ✅ **RAG Chain構築**: LangChain Databricksを使用してRAGチェーンを実装
# MAGIC
# MAGIC ✅ **MLflowモデル登録**: RAGチェーンをMLflowモデルとして登録
# MAGIC
# MAGIC ✅ **エンドポイントデプロイメント**: MLflow Model Servingエンドポイントを作成
# MAGIC
# MAGIC ### 使用方法
# MAGIC
# MAGIC 1. **直接クエリ**: `query_rag("質問")`関数を使用
# MAGIC 2. **MLflowモデル**: `mlflow.pyfunc.load_model("models:/commuting_allowance_rag_model/1")`でロード
# MAGIC 3. **エンドポイント**: REST API経由でエンドポイントにリクエストを送信
# MAGIC
# MAGIC ### 次のステップ
# MAGIC
# MAGIC - エンドポイントの監視とログ確認
# MAGIC - プロンプトの最適化
# MAGIC - 評価メトリクスの追加
# MAGIC - チャット履歴の管理（オプション）


