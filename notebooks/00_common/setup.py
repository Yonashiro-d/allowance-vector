# Databricks notebook source
# MAGIC %md
# MAGIC # 共通セットアップ

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
