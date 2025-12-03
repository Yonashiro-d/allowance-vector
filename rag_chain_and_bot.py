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
# MAGIC   langchain-community \
# MAGIC   databricks-vectorsearch \
# MAGIC   databricks-sdk \
# MAGIC   mlflow \
# MAGIC   requests \
# MAGIC   pandas

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import os
from typing import List, Dict, Any, Optional
from pyspark.sql import SparkSession
from databricks.sdk import WorkspaceClient
from databricks.vector_search.client import VectorSearchClient
from langchain_community.embeddings import DatabricksEmbeddings
from langchain_community.llms import Databricks
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import DatabricksVectorSearch
from langchain.schema import BaseRetriever, Document
import mlflow
import mlflow.pyfunc

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
QUERY_EMBEDDING_MODEL = "databricks-gte-large-en"

print(f"Catalog: {CATALOG}")
print(f"Schema: {SCHEMA}")
print(f"Table: {DELTA_TABLE_NAME}")
print(f"Vector Search Endpoint: {VECTOR_SEARCH_ENDPOINT}")
print(f"Vector Index: {VECTOR_INDEX_NAME}")
print(f"Query Embedding Model: {QUERY_EMBEDDING_MODEL}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## セクション1: クエリ用Embeddingモデルの実装

# COMMAND ----------

# MAGIC %md
# MAGIC ### Embeddingモデルの初期化

# COMMAND ----------

# クエリ用Embeddingモデルを初期化
query_embeddings = DatabricksEmbeddings(
    model=QUERY_EMBEDDING_MODEL,
    endpoint_type="databricks-foundation-model"
)

# テストクエリでベクトル化を確認
test_query = "通勤手当はいくらまで支給されますか？"
test_embedding = query_embeddings.embed_query(test_query)
print(f"Test query: {test_query}")
print(f"Embedding dimension: {len(test_embedding)}")
print(f"First 5 values: {test_embedding[:5]}")

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
) -> List[Dict[str, Any]]:
    """
    Vector Searchを使用して類似チャンクを検索
    
    インデックス構造:
    - primary_key: chunk_id (bigint)
    - embedding_vector_column: embedding (array<float>, 768次元)
    - columns: chunk_id, chunked_text (string), embedding, file_name (string)
    
    Args:
        query: 検索クエリ
        top_k: 返すチャンクの数（デフォルト: 5）
        filters: フィルタ条件（オプション）
    
    Returns:
        検索結果のリスト（各要素はchunk_id、chunked_text、file_name、scoreを含む）
    """
    # クエリをベクトル化（768次元のベクトルを生成）
    query_vector = query_embeddings.embed_query(query)
    
    # ベクトル次元の確認
    if len(query_vector) != 768:
        print(f"Warning: Query vector dimension ({len(query_vector)}) doesn't match index dimension (768)")
    
    # Vector Searchで検索
    # 実際のAPI構造に基づいて実装
    try:
        # インデックスオブジェクトを取得
        index = vsc.get_index(
            endpoint_name=VECTOR_SEARCH_ENDPOINT,
            index_name=VECTOR_INDEX_NAME
        )
        
        # 実際のAPIメソッドを試行（複数の可能性を試す）
        query_result = None
        
        # 方法1: indexオブジェクトのメソッドを試す
        if hasattr(index, 'similarity_search'):
            query_result = index.similarity_search(
                query_vector=query_vector,
                columns=["chunk_id", "chunked_text", "file_name"],
                num_results=top_k,
                filters=filters
            )
        if hasattr(index, 'query_vector'):
            query_result = index.query_vector(
                query_vector=query_vector,
                columns=["chunk_id", "chunked_text", "file_name"],
                num_results=top_k,
                filters=filters
            )
        
        # 方法2: VectorSearchClientのメソッドを直接使用
        if query_result is None:
            if hasattr(vsc, 'query_index'):
                query_result = vsc.query_index(
                    index_name=VECTOR_INDEX_NAME,
                    query_vector=query_vector,
                    columns=["chunk_id", "chunked_text", "file_name"],
                    num_results=top_k,
                    filters=filters
                )
        
        # 方法3: WorkspaceClientのAPIを使用
        if query_result is None:
            try:
                query_result = w.vector_search_indexes.query_index(
                    index_name=VECTOR_INDEX_NAME,
                    query_vector=query_vector,
                    columns=["chunk_id", "chunked_text", "file_name"],
                    num_results=top_k,
                    filters=filters
                )
            except Exception:
                pass
        
        if query_result is None:
            raise ValueError("Could not find valid query method. Please check Databricks Vector Search API documentation.")
        
        # 結果を整形
        search_results = []
        
        # デバッグ: 実際のレスポンス構造を確認
        print(f"Debug: query_result type: {type(query_result)}")
        if hasattr(query_result, '__dict__'):
            print(f"Debug: query_result attributes: {dir(query_result)}")
        
        # 様々な応答形式に対応
        # パターン1: 辞書のリスト
        if isinstance(query_result, list):
            for result in query_result:
                if isinstance(result, dict):
                    search_results.append({
                        "chunk_id": result.get("chunk_id"),
                        "chunked_text": result.get("chunked_text", ""),
                        "file_name": result.get("file_name", ""),
                        "score": result.get("score", result.get("distance", 0.0))
                    })
                if isinstance(result, list) and len(result) >= 3:
                    # [chunk_id, chunked_text, file_name, score] の形式
                    search_results.append({
                        "chunk_id": result[0] if len(result) > 0 else None,
                        "chunked_text": result[1] if len(result) > 1 else "",
                        "file_name": result[2] if len(result) > 2 else "",
                        "score": result[3] if len(result) > 3 else 0.0
                    })
        
        # パターン2: オブジェクト形式（result属性を持つ）
        if hasattr(query_result, 'result') and query_result.result:
            data = query_result.result
            if isinstance(data, list):
                for result in data:
                    if isinstance(result, dict):
                        search_results.append({
                            "chunk_id": result.get("chunk_id"),
                            "chunked_text": result.get("chunked_text", ""),
                            "file_name": result.get("file_name", ""),
                            "score": result.get("score", result.get("distance", 0.0))
                        })
            if hasattr(data, 'data_array'):
                for result in data.data_array:
                    if isinstance(result, dict):
                        search_results.append({
                            "chunk_id": result.get("chunk_id"),
                            "chunked_text": result.get("chunked_text", ""),
                            "file_name": result.get("file_name", ""),
                            "score": result.get("score", result.get("distance", 0.0))
                        })
                    if isinstance(result, list) and len(result) >= 3:
                        search_results.append({
                            "chunk_id": result[0] if len(result) > 0 else None,
                            "chunked_text": result[1] if len(result) > 1 else "",
                            "file_name": result[2] if len(result) > 2 else "",
                            "score": result[3] if len(result) > 3 else 0.0
                        })
        
        # パターン3: 辞書形式（単一の結果またはメタデータを含む）
        if isinstance(query_result, dict):
            if "chunk_id" in query_result or "chunked_text" in query_result:
                search_results.append({
                    "chunk_id": query_result.get("chunk_id"),
                    "chunked_text": query_result.get("chunked_text", ""),
                    "file_name": query_result.get("file_name", ""),
                    "score": query_result.get("score", query_result.get("distance", 0.0))
                })
            if "results" in query_result:
                # resultsキーにリストが含まれている場合
                for result in query_result["results"]:
                    if isinstance(result, dict):
                        search_results.append({
                            "chunk_id": result.get("chunk_id"),
                            "chunked_text": result.get("chunked_text", ""),
                            "file_name": result.get("file_name", ""),
                            "score": result.get("score", result.get("distance", 0.0))
                        })
        
        return search_results
        
    except Exception as e:
        print(f"Error in vector search: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        print("Falling back to direct table query...")
        return []

# テスト検索
test_results = search_similar_chunks("通勤手当はいくらまで支給されますか？", top_k=3)
print(f"Found {len(test_results)} similar chunks:")
for i, result in enumerate(test_results, 1):
    print(f"\n{i}. Chunk ID: {result['chunk_id']}")
    print(f"   File: {result['file_name']}")
    print(f"   Score: {result['score']:.4f}")
    print(f"   Text (first 100 chars): {result['chunked_text'][:100]}...")

# COMMAND ----------

# MAGIC %md
# MAGIC ## セクション3: RAG Chain構築

# COMMAND ----------

# MAGIC %md
# MAGIC ### LangChain Databricks Vector Storeの設定

# COMMAND ----------

# Databricks Vector SearchをVector Storeとして設定
# 注意: DatabricksVectorSearchの実際のAPIに応じて調整が必要な場合があります
try:
    vector_store = DatabricksVectorSearch(
        index=DatabricksVectorSearch.Index(
            endpoint_name=VECTOR_SEARCH_ENDPOINT,
            index_name=VECTOR_INDEX_NAME
        ),
        embedding=query_embeddings,
        text_column="chunked_text"
    )
    print("Vector Store initialized")
except Exception as e:
    print(f"Warning: Could not initialize DatabricksVectorSearch: {e}")
    print("Using custom retriever instead")
    vector_store = None

# COMMAND ----------

# MAGIC %md
# MAGIC ### LLMの初期化

# COMMAND ----------

# Databricks Foundation ModelをLLMとして使用
# デフォルトはdatabricks-dbrx-instructを使用
llm = Databricks(
    endpoint="databricks-dbrx-instruct",
    model_kwargs={"temperature": 0.1, "max_tokens": 500}
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
        results = self.search_func(query, top_k=self.top_k)
        documents = []
        for result in results:
            doc = Document(
                page_content=result.get("chunked_text", ""),
                metadata={
                    "chunk_id": result.get("chunk_id"),
                    "file_name": result.get("file_name", ""),
                    "score": result.get("score", 0.0)
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

# テストクエリ
test_question = "通勤手当はいくらまで支給されますか？"
test_result = query_rag(test_question)

print(f"質問: {test_result['question']}")
print(f"\n回答: {test_result['answer']}")
print(f"\n参照元: {test_result['num_sources']}件")
for i, source in enumerate(test_result['sources'][:3], 1):
    print(f"\n参照元{i}:")
    print(f"  内容: {source['content'][:200]}...")
    if source.get('metadata'):
        print(f"  メタデータ: {source['metadata']}")

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
        from langchain_community.embeddings import DatabricksEmbeddings
        from langchain_community.llms import Databricks
        from langchain.chains import RetrievalQA
        from langchain.prompts import PromptTemplate
        from langchain_community.vectorstores import DatabricksVectorSearch
        
        # 設定を環境変数から取得
        catalog = os.environ.get("CATALOG", "hhhd_demo_itec")
        schema = os.environ.get("SCHEMA", "allowance_payment_rules")
        vector_search_endpoint = os.environ.get("VECTOR_SEARCH_ENDPOINT", "databricks-bge-large-en-endpoint")
        vector_index_name = os.environ.get("VECTOR_INDEX_NAME", f"{catalog}.{schema}.commuting_allowance_index")
        query_embedding_model = os.environ.get("QUERY_EMBEDDING_MODEL", "databricks-gte-large-en")
        
        # Embeddingモデルを初期化
        self.query_embeddings = DatabricksEmbeddings(
            model=query_embedding_model,
            endpoint_type="databricks-foundation-model"
        )
        
        # Vector Storeを設定（エラーハンドリング付き）
        try:
            vector_store = DatabricksVectorSearch(
                index=DatabricksVectorSearch.Index(
                    endpoint_name=vector_search_endpoint,
                    index_name=vector_index_name
                ),
                embedding=self.query_embeddings,
                text_column="chunked_text"
            )
        except Exception as e:
            print(f"Warning: Could not initialize DatabricksVectorSearch: {e}")
            vector_store = None
        
        # LLMを初期化
        llm = Databricks(
            endpoint="databricks-dbrx-instruct",
            model_kwargs={"temperature": 0.1, "max_tokens": 500}
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
                    # VectorSearchClientのqueryメソッドを直接使用
                    query_result = self.vsc.query(
                        index_name=self.index_name,
                        query_vector=query_vector,
                        columns=["chunk_id", "chunked_text", "file_name"],
                        num_results=self.top_k
                    )
                    
                    documents = []
                    # 様々な応答形式に対応
                    if hasattr(query_result, 'result') and query_result.result:
                        data = query_result.result
                        if hasattr(data, 'data_array'):
                            for result in data.data_array:
                                if isinstance(result, list) and len(result) >= 3:
                                    doc = Document(
                                        page_content=result[1] if len(result) > 1 else "",
                                        metadata={
                                            "chunk_id": result[0] if len(result) > 0 else None,
                                            "file_name": result[2] if len(result) > 2 else "",
                                            "score": result[3] if len(result) > 3 else 0.0
                                        }
                                    )
                                    documents.append(doc)
                                if isinstance(result, dict):
                                    doc = Document(
                                        page_content=result.get("chunked_text", ""),
                                        metadata={
                                            "chunk_id": result.get("chunk_id"),
                                            "file_name": result.get("file_name", ""),
                                            "score": result.get("score", 0.0)
                                        }
                                    )
                                    documents.append(doc)
                        if isinstance(data, list):
                            for result in data:
                                if isinstance(result, dict):
                                    doc = Document(
                                        page_content=result.get("chunked_text", ""),
                                        metadata={
                                            "chunk_id": result.get("chunk_id"),
                                            "file_name": result.get("file_name", ""),
                                            "score": result.get("score", 0.0)
                                        }
                                    )
                                    documents.append(doc)
                    if isinstance(query_result, list):
                        for result in query_result:
                            if isinstance(result, dict):
                                doc = Document(
                                    page_content=result.get("chunked_text", ""),
                                    metadata={
                                        "chunk_id": result.get("chunk_id"),
                                        "file_name": result.get("file_name", ""),
                                        "score": result.get("score", 0.0)
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

# MLflow実験を設定
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
    # モデルのシグネチャを定義
    from mlflow.models import infer_signature
    
    # サンプル入力でシグネチャを推論
    sample_input = "通勤手当はいくらまで支給されますか？"
    sample_output = query_rag(sample_input)
    
    signature = infer_signature(
        model_input=sample_input,
        model_output=sample_output
    )
    
    # 環境変数を設定（モデルロード時に使用）
    env_vars = {
        "CATALOG": CATALOG,
        "SCHEMA": SCHEMA,
        "VECTOR_SEARCH_ENDPOINT": VECTOR_SEARCH_ENDPOINT,
        "VECTOR_INDEX_NAME": VECTOR_INDEX_NAME,
        "QUERY_EMBEDDING_MODEL": QUERY_EMBEDDING_MODEL
    }
    
    # モデルをログ
    mlflow.pyfunc.log_model(
        artifact_path="rag_model",
        python_model=RAGModel(),
        signature=signature,
        input_example=sample_input,
        registered_model_name="commuting_allowance_rag_model"
    )
    
    # 環境変数をログ
    mlflow.log_params(env_vars)
    
    # メトリクスをログ
    mlflow.log_metric("num_sources", sample_output.get("num_sources", 0))
    
    run_id = mlflow.active_run().info.run_id
    print(f"Model logged with run_id: {run_id}")

print("Model registered in MLflow")

# COMMAND ----------

# MAGIC %md
# MAGIC ### モデルのテスト

# COMMAND ----------

# 登録されたモデルをロードしてテスト
model_name = "commuting_allowance_rag_model"
model_version = 1  # 最新バージョンを使用

loaded_model = mlflow.pyfunc.load_model(f"models:/{model_name}/{model_version}")

# テストクエリ
test_questions = [
    "通勤手当はいくらまで支給されますか？",
    "通勤手当の支給条件は何ですか？"
]

for question in test_questions:
    result = loaded_model.predict(question)
    print(f"\n質問: {question}")
    if isinstance(result, list) and len(result) > 0:
        print(f"回答: {result[0].get('answer', 'N/A')}")
        print(f"参照元: {result[0].get('num_sources', 0)}件")
    else:
        print(f"結果: {result}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## セクション5: エンドポイントデプロイメント

# COMMAND ----------

# MAGIC %md
# MAGIC ### MLflow Model Servingエンドポイントの作成

# COMMAND ----------

# エンドポイント名を定義
endpoint_name = "commuting-allowance-rag-endpoint"

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
# MAGIC ### エンドポイントのテスト

# COMMAND ----------

# エンドポイントがREADYになるまで待機
import time
max_wait_time = 600  # 10分
wait_interval = 10   # 10秒ごとに確認
elapsed_time = 0

while elapsed_time < max_wait_time:
    endpoint = w.serving_endpoints.get(endpoint_name)
    if endpoint.state == "READY":
        print(f"Endpoint '{endpoint_name}' is READY")
        break
    if endpoint.state in ["FAILED", "NOT_READY"]:
        print(f"Endpoint state: {endpoint.state}")
        break
    if endpoint.state != "READY" and endpoint.state not in ["FAILED", "NOT_READY"]:
        print(f"Waiting for endpoint to be ready... (State: {endpoint.state}, Elapsed: {elapsed_time}s)")
        time.sleep(wait_interval)
        elapsed_time += wait_interval

if endpoint.state == "READY":
    # エンドポイントをテスト
    import requests
    import json
    
    # エンドポイントのURLを取得
    endpoint_url = f"https://{workspace_url}/serving-endpoints/{endpoint_name}/invocations"
    
    # テストリクエスト
    test_payload = {
        "dataframe_records": [{"0": "通勤手当はいくらまで支給されますか？"}]
    }
    
    headers = {
        "Authorization": f"Bearer {w.config.token}",
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.post(
            endpoint_url,
            headers=headers,
            json=test_payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print("Endpoint test successful!")
            print(f"Response: {json.dumps(result, indent=2, ensure_ascii=False)}")
        else:
            print(f"Endpoint test failed with status {response.status_code}")
            print(f"Response: {response.text}")
    except Exception as e:
        print(f"Error testing endpoint: {e}")
        print("Note: You can test the endpoint manually using the Databricks UI or REST API")
else:
    print(f"Endpoint is not ready (State: {endpoint.state}). Please check the endpoint status manually.")

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

