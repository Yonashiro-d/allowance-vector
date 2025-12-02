# Databricks notebook source
# MAGIC %md
# MAGIC # PDF取り込み

# COMMAND ----------

# MAGIC %run ./00_common/setup

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
