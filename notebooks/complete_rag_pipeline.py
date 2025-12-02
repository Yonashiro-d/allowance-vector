# Databricks notebook source
# MAGIC %md
# MAGIC # Databricks PDF RAG パイプライン（統合版）
# MAGIC 
# MAGIC このノートブックは、PDFの取り込みからRAGクエリまでの全プロセスを1つのノートブックで実行できます。

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. 共通セットアップ

# COMMAND ----------

import sys
from pathlib import Path

try:
    context = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
    notebook_path = context.notebookPath().get()
    
    if "Repos" in notebook_path:
        parts = notebook_path.split("/")
        repos_idx = parts.index("Repos")
        repo_name = parts[repos_idx + 2]
        repo_root = Path("/Workspace/Repos") / parts[repos_idx + 1] / repo_name
    else:
        repo_root = Path.cwd()
    
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
except Exception:
    repo_root = Path.cwd()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

# COMMAND ----------

from pyspark.sql import SparkSession

CATALOG = "main"
SCHEMA = "rag_pipeline"
INDEX_NAME = "commute_allowance_index"
EMBEDDING_ENDPOINT = "databricks-bge-large-en-endpoint"

spark = SparkSession.builder.getOrCreate()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. 環境セットアップ（スキーマ・テーブル作成）

# COMMAND ----------

spark.sql(f"""
CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}
COMMENT 'RAG Pipeline schema'
""")

# COMMAND ----------

spark.sql(f"""
CREATE TABLE IF NOT EXISTS {CATALOG}.{SCHEMA}.chunks_with_vectors (
    chunk_id STRING NOT NULL,
    text STRING NOT NULL,
    vector ARRAY<FLOAT> NOT NULL,
    file_name STRING NOT NULL,
    page_number INT NOT NULL,
    created_at TIMESTAMP NOT NULL,
    PRIMARY KEY (chunk_id)
)
USING DELTA
TBLPROPERTIES (
    'delta.autoOptimize.optimizeWrite' = 'true',
    'delta.autoOptimize.autoCompact' = 'true'
)
""")

# COMMAND ----------

display(spark.sql(f"SHOW TABLES IN {CATALOG}.{SCHEMA}"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. PDF取り込み

# COMMAND ----------

import pdfplumber
import json
from pathlib import Path

# Repos内のDataディレクトリからPDFファイルを検索
try:
    context = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
    notebook_path = context.notebookPath().get()
    if "Repos" in notebook_path:
        parts = notebook_path.split("/")
        repos_idx = parts.index("Repos")
        repo_name = parts[repos_idx + 2]
        DATA_PATH = f"/Workspace/Repos/{parts[repos_idx + 1]}/{repo_name}/Data"
    else:
        raise ValueError("Notebook must be run from Repos")
except Exception as e:
    raise ValueError(f"Failed to determine Data folder path: {e}")

# COMMAND ----------

def find_pdf_files(data_path: str) -> list:
    """DataディレクトリからPDFファイルを検索"""
    pdf_files = []
    try:
        if dbutils.fs.exists(data_path):
            files = dbutils.fs.ls(data_path)
            for file_info in files:
                if file_info.name.lower().endswith('.pdf'):
                    pdf_files.append({
                        "path": file_info.path,
                        "name": file_info.name,
                        "size": file_info.size
                    })
    except Exception as e:
        raise FileNotFoundError(f"Error accessing Data folder: {e}")
    return pdf_files

# PDFファイルを検索
pdf_files = find_pdf_files(DATA_PATH)

if not pdf_files:
    raise FileNotFoundError(
        f"PDF file not found in Data folder: {DATA_PATH}\n"
        "Please upload a PDF file to the Data folder."
    )

# 複数のPDFがある場合は最初のものを使用
if len(pdf_files) > 1:
    print(f"Found {len(pdf_files)} PDF files in Data folder:")
    for i, pdf in enumerate(pdf_files, 1):
        print(f"  {i}. {pdf['name']} ({pdf['size']} bytes)")
    print(f"\nUsing first PDF: {pdf_files[0]['name']}")

PDF_PATH = pdf_files[0]['path']
print(f"\nSelected PDF: {pdf_files[0]['name']} ({pdf_files[0]['size']} bytes)")
print(f"Path: {PDF_PATH}")

# COMMAND ----------

def extract_text_from_pdf(pdf_path: str) -> list:
    local_path = f"/dbfs{pdf_path}" if not pdf_path.startswith("/dbfs") else pdf_path
    pages = []
    with pdfplumber.open(local_path) as pdf:
        for page_num, page in enumerate(pdf.pages, 1):
            text = page.extract_text()
            if text:
                pages.append({
                    "page_number": page_num,
                    "text": text.strip(),
                    "file_name": Path(pdf_path).name
                })
    return pages

# COMMAND ----------

pages = extract_text_from_pdf(PDF_PATH)
print(f"Total pages: {len(pages)}")

# COMMAND ----------

pages_json_path = "/tmp/pages.json"
with open(f"/dbfs{pages_json_path}", "w", encoding="utf-8") as f:
    json.dump(pages, f, ensure_ascii=False, indent=2, default=str)
print(f"Saved: {pages_json_path}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. チャンキング・ベクトル化

# COMMAND ----------

from langchain.text_splitter import RecursiveCharacterTextSplitter
from datetime import datetime
import uuid
import json
import time
import requests
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, TimestampType, ArrayType, FloatType
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

# COMMAND ----------

with open("/dbfs/tmp/pages.json", "r", encoding="utf-8") as f:
    pages = json.load(f)

# COMMAND ----------

CHUNK_SIZE = 50
CHUNK_OVERLAP = 10

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", "。", " ", ""]
)

chunks = []
for page in pages:
    page_chunks = text_splitter.create_documents([page['text']])
    for i, chunk in enumerate(page_chunks):
        chunks.append({
            "chunk_id": f"{page['file_name']}_p{page['page_number']}_c{i}_{uuid.uuid4().hex[:8]}",
            "text": chunk.page_content,
            "file_name": page['file_name'],
            "page_number": page['page_number'],
            "created_at": datetime.now()
        })

print(f"Total chunks: {len(chunks)}")

# COMMAND ----------

def embed_text(text: str) -> list:
    endpoint_url = f"https://{w.config.host}/serving-endpoints/{EMBEDDING_ENDPOINT}/invocations"
    headers = {
        "Authorization": f"Bearer {w.config.token}",
        "Content-Type": "application/json"
    }
    data = {"inputs": [text], "params": {}}
    response = requests.post(endpoint_url, json=data, headers=headers)
    response.raise_for_status()
    return response.json()["predictions"][0]

# COMMAND ----------

BATCH_SIZE = 32
vectors = []

for i in range(0, len(chunks), BATCH_SIZE):
    batch = chunks[i:i+BATCH_SIZE]
    batch_vectors = [embed_text(chunk['text']) for chunk in batch]
    
    for chunk, vector in zip(batch, batch_vectors):
        vectors.append({
            "chunk_id": chunk["chunk_id"],
            "text": chunk["text"],
            "vector": vector,
            "file_name": chunk["file_name"],
            "page_number": chunk["page_number"],
            "created_at": chunk["created_at"]
        })
    
    print(f"Processed {min(i+BATCH_SIZE, len(chunks))}/{len(chunks)}")
    time.sleep(0.1)

# COMMAND ----------

schema = StructType([
    StructField("chunk_id", StringType(), False),
    StructField("text", StringType(), False),
    StructField("vector", ArrayType(FloatType()), False),
    StructField("file_name", StringType(), False),
    StructField("page_number", IntegerType(), False),
    StructField("created_at", TimestampType(), False)
])

df = spark.createDataFrame(vectors, schema)
df.write.mode("append").saveAsTable(f"{CATALOG}.{SCHEMA}.chunks_with_vectors")
print(f"Saved {len(vectors)} vectors")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Vector Searchインデックス作成

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient
from databricks.sdk import WorkspaceClient
import time

w = WorkspaceClient()
vsc = VectorSearchClient(w)
source_table = f"{CATALOG}.{SCHEMA}.chunks_with_vectors"

# COMMAND ----------

result = spark.sql(f"SELECT SIZE(vector) as dim FROM {source_table} LIMIT 1").collect()
assert result[0]['dim'] == 1024, "Vector dimension must be 1024"

# COMMAND ----------

index_spec = {
    "name": INDEX_NAME,
    "primary_key": "chunk_id",
    "index_type": "DELTA_SYNC",
    "delta_sync_index_spec": {
        "source_table": source_table,
        "pipeline_type": "TRIGGERED",
        "embedding_source_columns": [
            {
                "embedding_model_endpoint_name": EMBEDDING_ENDPOINT,
                "name": "text"
            }
        ]
    }
}

index = vsc.create_index(index_spec)
print(f"Index created: {index.name}")

# COMMAND ----------

def wait_for_ready(index_name: str, timeout: int = 600) -> bool:
    start_time = time.time()
    while time.time() - start_time < timeout:
        index = vsc.get_index(index_name)
        print(f"Status: {index.status}")
        if index.status == "ONLINE":
            return True
        if index.status == "FAILED":
            return False
        time.sleep(10)
    return False

wait_for_ready(INDEX_NAME)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. RAGクエリ

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

# サンプルクエリの実行
result = query_rag("通勤手当はいくらまで支給されますか？")
print(f"質問: {result['query']}")
print(f"回答: {result['answer']}")
print(f"参照元: {len(result['sources'])}件")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 完了
# MAGIC 
# MAGIC パイプラインの実行が完了しました。以下の関数を使用して、任意のクエリを実行できます：
# MAGIC 
# MAGIC ```python
# MAGIC result = query_rag("あなたの質問")
# MAGIC print(result['answer'])
# MAGIC ```

