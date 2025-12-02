# Databricks notebook source
# MAGIC %md
# MAGIC # Vector Searchインデックス作成

# COMMAND ----------

# MAGIC %run ./00_common/setup

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
