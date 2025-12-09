# Databricks notebook source
# MAGIC %md
# MAGIC # ベクトルデータ準備
# MAGIC
# MAGIC このノートブックでは以下の処理を実行します：
# MAGIC 1. Unity Catalog/スキーマの確認・作成
# MAGIC 2. Vector Searchエンドポイントの確認・作成
# MAGIC 3. PDFからテキストを抽出
# MAGIC 4. テキストのチャンキング
# MAGIC 5. チャンクのベクトル化
# MAGIC 6. Deltaテーブルへの保存
# MAGIC 7. Vector Searchインデックスの作成
# MAGIC
# MAGIC 参考: [genai-cookbook](https://github.com/databricks/genai-cookbook)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 依存関係のインストール

# COMMAND ----------

# MAGIC %pip install -U pypdf sentence-transformers sentencepiece databricks-vectorsearch databricks-sdk

# COMMAND ----------

# MAGIC %md
# MAGIC ## ライブラリの再起動

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 基本インポート

# COMMAND ----------

import re
import time
from pathlib import Path
from pyspark.sql import SparkSession
from pyspark.sql.functions import expr

spark = SparkSession.builder.getOrCreate()
workspace_url = SparkSession.getActiveSession().conf.get("spark.databricks.workspaceUrl", None)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Databricks SDK関連のインポート

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import NotFound, PermissionDenied
from databricks.vector_search.client import VectorSearchClient
from databricks.sdk.service.vectorsearch import EndpointType

w = WorkspaceClient()
vsc = VectorSearchClient(disable_notice=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 設定の定義

# COMMAND ----------

CATALOG = "hhhd_demo_itec"
SCHEMA = "allowance_payment_rules"
DELTA_TABLE_NAME = f"{CATALOG}.{SCHEMA}.commuting_allowance_vectors"
VECTOR_SEARCH_ENDPOINT = "databricks-bge-large-en-endpoint"
VECTOR_INDEX_NAME = f"{CATALOG}.{SCHEMA}.commuting_allowance_index"

print(f"Catalog: {CATALOG}")
print(f"Schema: {SCHEMA}")
print(f"Delta Table: {DELTA_TABLE_NAME}")
print(f"Vector Search Endpoint: {VECTOR_SEARCH_ENDPOINT}")
print(f"Vector Index: {VECTOR_INDEX_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Unity Catalog/スキーマの確認・作成

# COMMAND ----------

try:
    _ = w.catalogs.get(CATALOG)
    print(f"PASS: UC catalog `{CATALOG}` exists")
except NotFound:
    print(f"`{CATALOG}` does not exist, trying to create...")
    try:
        _ = w.catalogs.create(name=CATALOG)
        print(f"PASS: UC catalog `{CATALOG}` created")
    except PermissionDenied:
        print(f"FAIL: `{CATALOG}` does not exist, and no permissions to create. Please provide an existing UC Catalog.")
        raise ValueError(f"Unity Catalog `{CATALOG}` does not exist.")

try:
    _ = w.schemas.get(full_name=f"{CATALOG}.{SCHEMA}")
    print(f"PASS: UC schema `{CATALOG}.{SCHEMA}` exists")
except NotFound:
    print(f"`{CATALOG}.{SCHEMA}` does not exist, trying to create...")
    try:
        _ = w.schemas.create(name=SCHEMA, catalog_name=CATALOG)
        print(f"PASS: UC schema `{CATALOG}.{SCHEMA}` created")
    except PermissionDenied:
        print(f"FAIL: `{CATALOG}.{SCHEMA}` does not exist, and no permissions to create. Please provide an existing UC Schema.")
        raise ValueError(f"Unity Catalog Schema `{CATALOG}.{SCHEMA}` does not exist.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Vector Searchエンドポイントの確認・作成
# MAGIC
# MAGIC エンドポイントが存在しない場合は作成します（最大20分かかる場合があります）。

# COMMAND ----------

vector_search_endpoints = w.vector_search_endpoints.list_endpoints()
if sum([VECTOR_SEARCH_ENDPOINT == ve.name for ve in vector_search_endpoints]) == 0:
    print(f"Please wait, creating Vector Search endpoint `{VECTOR_SEARCH_ENDPOINT}`. This can take up to 20 minutes...")
    w.vector_search_endpoints.create_endpoint_and_wait(VECTOR_SEARCH_ENDPOINT, endpoint_type=EndpointType.STANDARD)

w.vector_search_endpoints.wait_get_endpoint_vector_search_endpoint_online(VECTOR_SEARCH_ENDPOINT)
print(f"PASS: Vector Search endpoint `{VECTOR_SEARCH_ENDPOINT}` exists")

# COMMAND ----------

# MAGIC %md
# MAGIC ## PDFパス取得関数の定義

# COMMAND ----------

def get_pdf_path(data_path: str = None, pdf_path: str = None) -> str:
    if pdf_path:
        if not pdf_path.lower().endswith('.pdf'):
            raise ValueError(f"Specified path is not a PDF file: {pdf_path}")
        print(f"Using specified PDF path: {pdf_path}")
        return pdf_path
    
    if data_path is None:
        notebook_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
        if "Repos" not in notebook_path:
            raise ValueError("Notebook must be run from Repos. Please specify data_path or pdf_path explicitly.")
        parts = notebook_path.split("/")
        repos_idx = parts.index("Repos")
        data_path = f"/Workspace/Repos/{parts[repos_idx + 1]}/{parts[repos_idx + 2]}"
        print(f"Auto-detected data_path: {data_path}")
    
    pdf_files = [f for f in dbutils.fs.ls(data_path) if f.name.lower().endswith('.pdf')]
    if not pdf_files:
        all_files = dbutils.fs.ls(data_path)
        print(f"Files in {data_path}:")
        for f in all_files:
            print(f"  - {f.name} (is_dir: {f.isDir})")
        raise FileNotFoundError(f"PDF file not found in: {data_path}")
    
    print(f"Found {len(pdf_files)} PDF file(s):")
    for i, pdf in enumerate(pdf_files, 1):
        print(f"  {i}. {pdf.name}")
    if len(pdf_files) > 1:
        print(f"Using first PDF: {pdf_files[0].name}")
    
    return pdf_files[0].path

# COMMAND ----------

# MAGIC %md
# MAGIC ## PDFパスの指定

# COMMAND ----------

pdf_path = get_pdf_path(pdf_path="/Workspace/Users/toshimitsu-yu@itec.hankyu-hanshin.co.jp/allowance-vector/通勤手当支給規程（2024-04-01）.pdf")
print(f"PDF path: {pdf_path}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## PDFからテキスト抽出

# COMMAND ----------

from pypdf import PdfReader
import os

if pdf_path.startswith("/Workspace/Repos/"):
    pdf_path = pdf_path.replace("/Workspace/Repos/", "/dbfs/Repos/")
elif pdf_path.startswith("/Workspace/"):
    pdf_path = pdf_path.replace("/Workspace/", "/dbfs/Workspace/")

if not os.path.exists(pdf_path):
    raise FileNotFoundError(f"PDF file not found: {pdf_path}")

with open(pdf_path, "rb") as f:
    reader = PdfReader(f)
    raw_text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])

print(f"Extracted text: {len(raw_text)} characters, {len(reader.pages)} pages")

# COMMAND ----------

# MAGIC %md
# MAGIC ## チャンキング関数の定義

# COMMAND ----------

def extract_sections(text: str) -> list:
    sections = []
    raw_sections = re.split(r'\n\s*\n+', text.strip())
    
    for section in raw_sections:
        title_match = re.match(r'^(第\s*\d+\s*条.*?)$', section, re.MULTILINE)
        if title_match:
            title = title_match.group(1).strip()
            content = section[len(title):].strip()
            sections.append({"title": title, "content": content})
        else:
            sections.append({"title": "", "content": section.strip()})
    
    return sections

def chunk_text(text: str) -> list:
    text = re.sub(r'\n+', '\n', text).strip()
    sections = extract_sections(text)
    
    chunks = []
    for sec in sections:
        if sec['content']:
            if sec['title']:
                chunks.append(f"{sec['title']}:\n{sec['content']}")
            else:
                chunks.append(sec['content'])
    
    return chunks

# COMMAND ----------

# MAGIC %md
# MAGIC ## テキストのチャンキング実行

# COMMAND ----------

chunked_texts = chunk_text(raw_text)
print(f"Total chunks: {len(chunked_texts)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## チャンクデータの準備

# COMMAND ----------

pdf_texts = [
    {
        "chunk_id": i,
        "chunked_text": chunk,
        "file_name": Path(pdf_path).name
    }
    for i, chunk in enumerate(chunked_texts)
]

# COMMAND ----------

# MAGIC %md
# MAGIC ## ベクトル化
# MAGIC
# MAGIC SentenceTransformerを使用してチャンクをベクトル化します。

# COMMAND ----------

from sentence_transformers import SentenceTransformer

model = SentenceTransformer("cl-nagoya/ruri-v3-310m")
embedding_dimension = model.get_sentence_embedding_dimension()
print(f"Embedding model: {embedding_dimension} dimensions")

embeddings = model.encode(
    [item["chunked_text"] for item in pdf_texts],
    show_progress_bar=True,
    batch_size=32
)

for i in range(len(pdf_texts)):
    pdf_texts[i]["embedding"] = embeddings[i].tolist()

print(f"Generated {len(embeddings)} embeddings")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Deltaテーブルへの保存

# COMMAND ----------

df = spark.createDataFrame(pdf_texts)
df = df.withColumn("embedding", expr("transform(embedding, x -> cast(x as float))"))

df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(DELTA_TABLE_NAME)

spark.sql(f"ALTER TABLE {DELTA_TABLE_NAME} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")

saved_count = spark.sql(f"SELECT COUNT(*) as count FROM {DELTA_TABLE_NAME}").collect()[0]['count']
print(f"Saved {saved_count} records to {DELTA_TABLE_NAME}")

if workspace_url:
    table_name_only = DELTA_TABLE_NAME.split('.')[-1]
    print(f"View Delta Table: https://{workspace_url}/explore/data/{CATALOG}/{SCHEMA}/{table_name_only}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 保存データの確認

# COMMAND ----------

display(spark.sql(f"SELECT chunk_id, file_name, LENGTH(chunked_text) as text_length, SIZE(embedding) as embedding_dim FROM {DELTA_TABLE_NAME} LIMIT 5"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## ベクトル次元数の取得

# COMMAND ----------

result = spark.sql(f"SELECT SIZE(embedding) as dim FROM {DELTA_TABLE_NAME} LIMIT 1").collect()
embedding_dim = result[0]['dim']
print(f"Vector dimension: {embedding_dim}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Vector Searchインデックスの作成
# MAGIC
# MAGIC 既存のインデックスがある場合は削除してから新規作成します（5-10分かかる場合があります）。

# COMMAND ----------

print(f"Creating Vector Search Index, this will take ~5 - 10 minutes.")
if workspace_url:
    index_name_only = VECTOR_INDEX_NAME.split('.')[-1]
    print(f"View Index Status: https://{workspace_url}/explore/data/{CATALOG}/{SCHEMA}/{index_name_only}")

try:
    vsc.delete_index(endpoint_name=VECTOR_SEARCH_ENDPOINT, index_name=VECTOR_INDEX_NAME)
    print("Existing index deleted. Waiting for deletion to complete...")
    time.sleep(15)
except Exception:
    print("Index does not exist or already deleted. Proceeding with creation...")

index = vsc.create_delta_sync_index_and_wait(
    endpoint_name=VECTOR_SEARCH_ENDPOINT,
    index_name=VECTOR_INDEX_NAME,
    primary_key="chunk_id",
    source_table_name=DELTA_TABLE_NAME,
    pipeline_type="TRIGGERED",
    embedding_vector_column="embedding",
    embedding_dimension=embedding_dim
)

print(f"PASS: Vector Search Index `{VECTOR_INDEX_NAME}` created and ONLINE")
