# Databricks notebook source
# MAGIC %md
# MAGIC # RAGクエリ

# COMMAND ----------

# MAGIC %run ./00_common/setup

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient
from databricks.sdk import WorkspaceClient
import requests

w = WorkspaceClient()
vsc = VectorSearchClient(w)
TOP_K = 5

# COMMAND ----------

def embed_query(query_text: str) -> list:
    endpoint_url = f"https://{w.config.host}/serving-endpoints/{EMBEDDING_ENDPOINT}/invocations"
    headers = {
        "Authorization": f"Bearer {w.config.token}",
        "Content-Type": "application/json"
    }
    data = {"inputs": [query_text], "params": {}}
    response = requests.post(endpoint_url, json=data, headers=headers)
    response.raise_for_status()
    return response.json()["predictions"][0]

# COMMAND ----------

def search_chunks(query_vector: list) -> list:
    return vsc.query_index(
        index_name=INDEX_NAME,
        query_vector=query_vector,
        top_k=TOP_K
    )

# COMMAND ----------

def build_context(chunk_ids: list) -> str:
    chunk_ids_str = "', '".join(chunk_ids)
    query = f"""
        SELECT text, file_name, page_number
        FROM {CATALOG}.{SCHEMA}.chunks_with_vectors
        WHERE chunk_id IN ('{chunk_ids_str}')
        ORDER BY page_number
    """
    chunks = spark.sql(query).collect()
    return "\n---\n".join([
        f"[{chunk['file_name']}, ページ{chunk['page_number']}]\n{chunk['text']}"
        for chunk in chunks
    ])

# COMMAND ----------

def query_rag(query_text: str) -> dict:
    query_vector = embed_query(query_text)
    results = search_chunks(query_vector)
    
    if not results:
        return {"query": query_text, "answer": "情報が見つかりませんでした", "sources": []}
    
    chunk_ids = [r['chunk_id'] for r in results]
    context = build_context(chunk_ids)
    
    prompt = f"""以下の規定文書を参照して、質問に回答してください。

【規定文書】
{context}

【質問】
{query_text}

【回答】
"""
    
    sources = [
        {
            "chunk_id": r['chunk_id'],
            "score": r.get('score', 0)
        }
        for r in results
    ]
    
    return {
        "query": query_text,
        "answer": prompt,
        "sources": sources,
        "context": context
    }

# COMMAND ----------

result = query_rag("通勤手当はいくらまで支給されますか？")
print(f"質問: {result['query']}")
print(f"回答: {result['answer']}")
print(f"参照元: {len(result['sources'])}件")
