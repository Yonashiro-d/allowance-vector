# Databricks notebook source
# MAGIC %md
# MAGIC # 環境セットアップ

# COMMAND ----------

# MAGIC %run ./00_common/setup

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
