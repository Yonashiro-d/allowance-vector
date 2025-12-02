# Databricks notebook source
# MAGIC %md
# MAGIC # ベクトルデータ準備
# MAGIC
# MAGIC PDFからテキストを抽出し、チャンキングしてベクトル化し、Deltaテーブルに保存します。

# COMMAND ----------

# MAGIC %pip install -U pypdf sentence-transformers sentencepiece

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
# MAGIC ## PDFを読み込んでテキストに変換
# MAGIC

# COMMAND ----------

"""
PDFを相対パスで読み込むことができなかった為、その他の方法もできなかった為、絶対パスで読み込むことにしている。
"""
pdf_path = "/Workspace/Users/toshimitsu-yu@itec.hankyu-hanshin.co.jp/allowance-vector/Data/通勤手当支給規程（2024-04-01）.pdf"

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