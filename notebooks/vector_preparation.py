# Databricks notebook source
# MAGIC %md
# MAGIC # ベクトルデータ準備
# MAGIC 
# MAGIC このノートブックでは、PDFからテキストを抽出し、チャンキングしてベクトル化する処理を行います。
# MAGIC 
# MAGIC ## 処理の流れ
# MAGIC 1. **PDF読み込み**: PDFファイルからテキストを抽出
# MAGIC 2. **チャンキング**: テキストを適切なサイズのチャンクに分割
# MAGIC 3. **ベクトル化**: 各チャンクをベクトル（embedding）に変換
# MAGIC 4. **保存**: Deltaテーブルに保存

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. 環境設定

# COMMAND ----------

import sys
from pathlib import Path

# Repos内のパス設定（簡潔版）
try:
    notebook_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
    if "Repos" in notebook_path:
        parts = notebook_path.split("/")
        repos_idx = parts.index("Repos")
        repo_root = Path("/Workspace/Repos") / parts[repos_idx + 1] / parts[repos_idx + 2]
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
except Exception:
    pass

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, expr
import re
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# SparkSessionを取得
spark = SparkSession.builder.getOrCreate()

# 設定
CATALOG = "main"
SCHEMA = "rag_pipeline"
DELTA_TABLE_NAME = f"{CATALOG}.{SCHEMA}.commuting_allowance_vectors"

# ワークスペースURL取得（確認用リンク生成のため）
workspace_url = SparkSession.getActiveSession().conf.get(
    "spark.databricks.workspaceUrl", None
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. PDF読み込みとテキスト抽出

# COMMAND ----------

def get_pdf_path(data_path: str = None) -> str:
    """
    PDFファイルのパスを取得する関数（自動検出）
    
    Args:
        data_path: データディレクトリのパス（指定しない場合は自動検出）
    
    Returns:
        PDFファイルのパス
    """
    # パスが指定されていない場合は自動検出
    if data_path is None:
        notebook_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
        if "Repos" not in notebook_path:
            raise ValueError("Notebook must be run from Repos")
        parts = notebook_path.split("/")
        repos_idx = parts.index("Repos")
        data_path = f"/Workspace/Repos/{parts[repos_idx + 1]}/{parts[repos_idx + 2]}/Data"
    
    # PDFファイルを検索
    pdf_files = [f for f in dbutils.fs.ls(data_path) if f.name.lower().endswith('.pdf')]
    if not pdf_files:
        # デバッグ情報: 見つかったファイル一覧を表示
        all_files = dbutils.fs.ls(data_path)
        print(f"Files in {data_path}:")
        for f in all_files:
            print(f"  - {f.name} (is_dir: {f.isDir})")
        raise FileNotFoundError(f"PDF file not found in: {data_path}")
    
    # 見つかったPDFファイルを表示
    print(f"Found {len(pdf_files)} PDF file(s):")
    for i, pdf in enumerate(pdf_files, 1):
        print(f"  {i}. {pdf.name}")
    if len(pdf_files) > 1:
        print(f"Using first PDF: {pdf_files[0].name}")
    
    return pdf_files[0].path

# PDFパスを取得（自動検出）
pdf_path = get_pdf_path()
print(f"PDF: {pdf_path}")

# COMMAND ----------

# PDFを読み込んでテキストに変換
# DBFSパスの場合は自動変換
if pdf_path.startswith("/Workspace/"):
    pdf_path = f"/dbfs{pdf_path}"

with open(pdf_path, "rb") as f:
    reader = PdfReader(f)
    raw_text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])

print(f"Extracted text length: {len(raw_text)} characters")
print(f"First 500 characters:\n{raw_text[:500]}...")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. チャンキング（テキスト分割）
# MAGIC 
# MAGIC テキストを適切なサイズのチャンクに分割します。
# MAGIC 規程文書の場合は、「第X条」などの構造を考慮して分割します。

# COMMAND ----------

def extract_sections(text: str) -> list:
    """
    文書を適切なセクションごとに分割する関数
    
    処理内容:
    - 空白行を境目として各セクションを分割
    - 「第X条」や「附則」などのタイトルを検出して個別のエントリとして処理
    
    Args:
        text: 分割するテキスト
    
    Returns:
        セクションのリスト（各セクションは{"title": str, "content": str}の形式）
    """
    sections = []
    
    # 空白行を境目にして分割
    raw_sections = re.split(r'\n\s*\n+', text.strip())
    
    for section in raw_sections:
        # 「第X条」や「附則」がタイトルであるか判定
        title_match = re.match(r'^(第\s*\d+\s*条.*?)$', section, re.MULTILINE)
        
        if title_match:
            title = title_match.group(1).strip()
            content = section[len(title):].strip()  # タイトル以外の本文
            sections.append({"title": title, "content": content})
        else:
            # タイトルがないセクションもそのまま格納
            sections.append({"title": "", "content": section.strip()})
    
    return sections

# COMMAND ----------

def chunk_text(text: str) -> list:
    """
    条文ごとに適切にチャンクを作成する関数
    
    処理内容:
    - 「第〇条」や「第〇項」を検出し、新しいチャンクを作成
    - 各エントリを独立したチャンクとして保存
    
    Args:
        text: チャンキングするテキスト
    
    Returns:
        チャンクのリスト（各チャンクは文字列）
    """
    # 連続する改行を1つに統一
    text = re.sub(r'\n+', '\n', text).strip()
    
    # 空白行で分割してセクションを取得
    sections = extract_sections(text)
    
    chunks = []
    for sec in sections:
        if sec['content']:
            if sec['title']:
                # タイトルがある場合は「タイトル: 内容」の形式で結合
                temp_chunk = f"{sec['title']}:\n{sec['content']}"
            else:
                # タイトルがない場合は内容のみ
                temp_chunk = sec['content']
            chunks.append(temp_chunk)
    
    return chunks

# COMMAND ----------

# テキストをチャンクに分割
chunked_texts = chunk_text(raw_text)
print(f"Total chunks created: {len(chunked_texts)}")
print(f"\nFirst chunk (first 300 chars):\n{chunked_texts[0][:300]}...")
print(f"\nLast chunk (first 300 chars):\n{chunked_texts[-1][:300]}...")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. ベクトル化（Embedding）
# MAGIC 
# MAGIC 各チャンクをベクトル（embedding）に変換します。
# MAGIC ここでは日本語対応のSentenceTransformerモデルを使用します。

# COMMAND ----------

# チャンクデータを辞書形式に変換（後でDataFrameに変換するため）
pdf_texts = [
    {
        "chunk_id": i,
        "chunked_text": chunk,
        "file_name": Path(pdf_path).name
    }
    for i, chunk in enumerate(chunked_texts)
]

print(f"Prepared {len(pdf_texts)} chunks for embedding")

# COMMAND ----------

# Embeddingモデルの読み込み
# 日本語対応のモデルを使用（768次元）
# 他のモデル例:
# - "cl-nagoya/ruri-base-v2" (768次元)
# - "/Workspace/Users/.../ruri-base-v2" (ローカルモデル)
model = SentenceTransformer("cl-nagoya/ruri-v3-310m")
print(f"Loaded embedding model: {model.get_sentence_embedding_dimension()} dimensions")

# COMMAND ----------

# データをベクトル化
# 全チャンクを一度にエンコード（バッチ処理）
print("Encoding chunks to vectors...")
embeddings = model.encode(
    [item["chunked_text"] for item in pdf_texts],
    show_progress_bar=True,
    batch_size=32  # バッチサイズを指定（メモリに応じて調整）
)

print(f"Generated {len(embeddings)} embeddings")
print(f"Embedding dimension: {len(embeddings[0])}")

# COMMAND ----------

# 各チャンクにembeddingを追加
for i in range(len(pdf_texts)):
    pdf_texts[i]["embedding"] = embeddings[i].tolist()

print(f"Added embeddings to {len(pdf_texts)} chunks")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Deltaテーブルへの保存

# COMMAND ----------

# DataFrameに変換
df = spark.createDataFrame(pdf_texts)

# double → float 配列への変換（Deltaテーブルの型要件に合わせる）
df = df.withColumn("embedding", expr("transform(embedding, x -> cast(x as float))"))

# スキーマ確認
print("DataFrame schema:")
df.printSchema()

# COMMAND ----------

# Deltaテーブルとして保存（上書き & schema更新）
df.write.format("delta")\
    .mode("overwrite")\
    .option("overwriteSchema", "true")\
    .saveAsTable(DELTA_TABLE_NAME)

print(f"Saved to Delta table: {DELTA_TABLE_NAME}")

# COMMAND ----------

# Change Data Feedを有効化（オプション）
spark.sql(
    f"ALTER TABLE {DELTA_TABLE_NAME} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)"
)

# COMMAND ----------

# 保存されたデータの確認
saved_count = spark.sql(f"SELECT COUNT(*) as count FROM {DELTA_TABLE_NAME}").collect()[0]['count']
print(f"Total records in table: {saved_count}")

# サンプルデータの表示
print("\nSample data:")
display(spark.sql(f"SELECT chunk_id, file_name, LENGTH(chunked_text) as text_length, SIZE(embedding) as embedding_dim FROM {DELTA_TABLE_NAME} LIMIT 5"))

# COMMAND ----------

# Databricks UI 上で Delta Table を確認できるリンクを表示
if workspace_url:
    table_name_only = DELTA_TABLE_NAME.split('.')[-1]
    print(
        f"\nView Delta Table at: https://{workspace_url}/explore/data/{CATALOG}/{SCHEMA}/{table_name_only}"
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## 完了
# MAGIC 
# MAGIC ベクトルデータの準備が完了しました。
# MAGIC 
# MAGIC **次のステップ:**
# MAGIC - Vector Searchインデックスの作成
# MAGIC - RAGチャットアプリの構築

