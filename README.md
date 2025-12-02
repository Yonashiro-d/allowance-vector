# Databricks PDF RAG ベクトルデータベース化

## 概要

通勤手当支給規定PDFをDatabricks Vector SearchとDatabricks BGEを使用してベクトルデータベース化し、RAGパイプラインを構築する。

## 最小要件

1. ✅ PDF読み込みとテキスト抽出
2. ✅ チャンキング（50文字単位）
3. ✅ ベクトル化（Databricks BGE、1024次元）
4. ✅ Vector Searchインデックス作成
5. ✅ 基本的なRAGクエリ

## クイックスタート

### 1. リポジトリのクローン

DatabricksワークスペースでReposにクローン:
- Git URL: `https://github.com/Yonashiro-d/allowance-vector.git`
- リポジトリは `/Workspace/Repos/{username}/allowance-vector` に配置

### 2. 環境設定

1. **クラスター作成**
   - Databricks Runtime 14.3 LTS以降
   - ライブラリ: `pdfplumber==0.10.0`, `langchain==0.1.0`

2. **Vector Searchエンドポイント作成**
   - エンドポイント名: `databricks-bge-large-en-endpoint`
   - モデル: `databricks-bge-large-en`

3. **PDFファイルの配置**
   - ローカルの `Data/` フォルダにPDFファイルを配置
   - Gitにコミットしてリポジトリに含める
   - ファイル名は任意（`.pdf`拡張子であればOK）
   - パス: `/Workspace/Repos/{username}/allowance-vector/Data/`
   - **自動検出機能**: Notebookが `Data/` フォルダから自動的にPDFファイルを検出します

4. **設定の確認**
   - `notebooks/00_common/setup.py`で設定を確認・変更

### 3. 実行順序

1. `00_common/setup.py` - 共通設定
2. `01_setup/environment_setup.py` - テーブル作成
3. `02_ingestion/pdf_ingestion.py` - PDF読み込み
4. `03_processing/chunking_embedding.py` - チャンキング・ベクトル化
5. `03_processing/vector_index_creation.py` - インデックス作成
6. `04_retrieval/rag_query.py` - RAGクエリ

## 技術スタック

- Databricks Vector Search
- Databricks BGE (Embedding)
- LangChain (チャンキング)
- Delta Lake
- Unity Catalog
