# Databricks notebook source
# MAGIC %md
# MAGIC # チャンキング・ベクトル化

# COMMAND ----------

# MAGIC %run ./00_common/setup

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
