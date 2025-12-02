# Databricks notebook source
# MAGIC %md
# MAGIC # ベクトルデータ準備
# MAGIC 
# MAGIC PDFからテキストを抽出し、チャンキングしてベクトル化し、Deltaテーブルに保存します。

# COMMAND ----------

# MAGIC %pip install -U pypdf sentence-transformers

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import sys
import re
from pathlib import Path
from pyspark.sql import SparkSession
from pyspark.sql.functions import expr
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

spark = SparkSession.builder.getOrCreate()

# 設定
CATALOG = "main"
SCHEMA = "rag_pipeline"
DELTA_TABLE_NAME = f"{CATALOG}.{SCHEMA}.commuting_allowance_vectors"

workspace_url = SparkSession.getActiveSession().conf.get("spark.databricks.workspaceUrl", None)

print(f"Table: {DELTA_TABLE_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## PDF読み込み

# COMMAND ----------

def get_pdf_path(data_path: str = None) -> str:
    """PDFファイルのパスを取得（Repos配下のDataフォルダから自動検出）"""
    if data_path is None:
        notebook_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
        if "Repos" not in notebook_path:
            raise ValueError("Notebook must be run from Repos. Please specify data_path explicitly.")
        parts = notebook_path.split("/")
        repos_idx = parts.index("Repos")
        data_path = f"/Workspace/Repos/{parts[repos_idx + 1]}/{parts[repos_idx + 2]}/Data"
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

pdf_path = get_pdf_path()
print(f"PDF path: {pdf_path}")

# COMMAND ----------

# PDFを読み込んでテキストに変換
if pdf_path.startswith("/Workspace/"):
    pdf_path = f"/dbfs{pdf_path}"

with open(pdf_path, "rb") as f:
    reader = PdfReader(f)
    raw_text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])

print(f"Extracted text: {len(raw_text)} characters, {len(reader.pages)} pages")

# COMMAND ----------

# MAGIC %md
# MAGIC ## チャンキング

# COMMAND ----------

def extract_sections(text: str) -> list:
    """空白行で分割し、「第X条」などのタイトルを検出"""
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
    """セクションをチャンクに変換"""
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

chunked_texts = chunk_text(raw_text)
print(f"Total chunks: {len(chunked_texts)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## ベクトル化

# COMMAND ----------

pdf_texts = [
    {
        "chunk_id": i,
        "chunked_text": chunk,
        "file_name": Path(pdf_path).name
    }
    for i, chunk in enumerate(chunked_texts)
]

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
# MAGIC ## Deltaテーブルに保存

# COMMAND ----------

df = spark.createDataFrame(pdf_texts)
df = df.withColumn("embedding", expr("transform(embedding, x -> cast(x as float))"))

df.write.format("delta")\
    .mode("overwrite")\
    .option("overwriteSchema", "true")\
    .saveAsTable(DELTA_TABLE_NAME)

spark.sql(f"ALTER TABLE {DELTA_TABLE_NAME} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")

saved_count = spark.sql(f"SELECT COUNT(*) as count FROM {DELTA_TABLE_NAME}").collect()[0]['count']
print(f"Saved {saved_count} records to {DELTA_TABLE_NAME}")

display(spark.sql(f"SELECT chunk_id, file_name, LENGTH(chunked_text) as text_length, SIZE(embedding) as embedding_dim FROM {DELTA_TABLE_NAME} LIMIT 5"))

if workspace_url:
    table_name_only = DELTA_TABLE_NAME.split('.')[-1]
    print(f"\nView Delta Table: https://{workspace_url}/explore/data/{CATALOG}/{SCHEMA}/{table_name_only}")
