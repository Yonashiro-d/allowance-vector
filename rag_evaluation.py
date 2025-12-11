# Databricks notebook source
# MAGIC %md
# MAGIC # RAGチェーン品質評価
# MAGIC
# MAGIC このノートブックでは以下の処理を実行します：
# MAGIC 1. 評価データセットの定義
# MAGIC 2. RAGチェーンの評価実行
# MAGIC 3. 品質メトリクスの計算（関連性、接地性、正確性など）
# MAGIC 4. MLflow評価機能を使用した評価結果の記録
# MAGIC 5. 評価結果の可視化と分析
# MAGIC
# MAGIC **目的**: RAGチェーンの品質を体系的に評価し、継続的な改善を行う

# COMMAND ----------

# MAGIC %md
# MAGIC ## 依存関係のインストール

# COMMAND ----------

# MAGIC %pip install mlflow langchain langchain-core langchain-community langchain-huggingface databricks-langchain databricks-vectorsearch databricks-sdk sentence-transformers transformers tokenizers sentencepiece torch numpy pandas

# COMMAND ----------

# MAGIC %md
# MAGIC ## ライブラリの再起動

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 基本インポート

# COMMAND ----------

from typing import Any, Dict, List
import time
import pandas as pd
from pyspark.sql import SparkSession
import mlflow
from mlflow.genai import evaluate as genai_evaluate, scorer
from mlflow.genai.scorers import Correctness, RelevanceToQuery, Guidelines

spark = SparkSession.builder.getOrCreate()

# COMMAND ----------

# MAGIC %md
# MAGIC ## RAG設定の読み込み

# COMMAND ----------

from rag_config import RAGConfig

config = RAGConfig()
print(f"Catalog: {config.catalog}, Schema: {config.schema}")
print(f"Delta Table: {config.delta_table_name}")
print(f"Vector Index: {config.vector_index_name}")
print(f"LLM Endpoint: {config.llm_endpoint}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## LangChain関連のインポート

# COMMAND ----------

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from databricks_langchain import ChatDatabricks, DatabricksVectorSearch
from langchain_huggingface import HuggingFaceEmbeddings

# COMMAND ----------

# MAGIC %md
# MAGIC ## RAGチェーン構築関数の定義

# COMMAND ----------

def build_rag_chain(config: RAGConfig) -> Any:
    """RAGチェーンを構築する関数"""
    embedding_model = HuggingFaceEmbeddings(model_name=config.query_embedding_model)
    
    vector_store = DatabricksVectorSearch(
        index_name=config.vector_index_name,
        embedding=embedding_model,
        text_column="chunked_text"
    )
    
    llm = ChatDatabricks(
        endpoint=config.llm_endpoint,
        extra_params={"temperature": 0.1}
    )
    
    retriever = vector_store.as_retriever(search_kwargs={"k": config.retriever_top_k})
    
    prompt_template = """あなたは質問に答えるアシスタントです。取得したコンテキストの内容をもとに質問に答えてください。一部のコンテキストが無関係な場合、それを回答に利用しないでください。

コンテキスト:
{context}

質問: {input}"""
    prompt = ChatPromptTemplate.from_template(prompt_template)
    
    document_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, document_chain)
    
    return rag_chain

# COMMAND ----------

# MAGIC %md
# MAGIC ## 評価データセットの定義
# MAGIC
# MAGIC 通勤手当に関する質問と期待される回答を含む評価データセットを定義します。

# COMMAND ----------

eval_dataset = [
    {
        "inputs": {"question": "通勤手当はいくらまで支給されますか？"},
        "expectations": {
            "expected_facts": ["通勤手当の支給上限額に関する情報"]
        },
        "tags": {"category": "支給額", "difficulty": "easy"},
    },
    {
        "inputs": {"question": "通勤手当の支給条件を教えてください"},
        "expectations": {
            "expected_facts": ["通勤手当の支給条件に関する情報"]
        },
        "tags": {"category": "支給条件", "difficulty": "medium"},
    },
    {
        "inputs": {"question": "通勤手当はどのように計算されますか？"},
        "expectations": {
            "expected_facts": ["通勤手当の計算方法に関する情報"]
        },
        "tags": {"category": "計算方法", "difficulty": "medium"},
    },
    {
        "inputs": {"question": "通勤手当の申請方法を教えてください"},
        "expectations": {
            "expected_facts": ["通勤手当の申請方法に関する情報"]
        },
        "tags": {"category": "申請方法", "difficulty": "easy"},
    },
    {
        "inputs": {"question": "通勤手当が支給されない場合はありますか？"},
        "expectations": {
            "expected_facts": ["通勤手当が支給されない条件に関する情報"]
        },
        "tags": {"category": "支給条件", "difficulty": "hard"},
    },
]

print(f"評価データセット: {len(eval_dataset)}件の質問を定義しました")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 予測関数の定義
# MAGIC
# MAGIC MLflow評価で使用する予測関数を定義します。

# COMMAND ----------

# RAGチェーンを構築（一度だけ初期化）
rag_chain = build_rag_chain(config)
print("RAGチェーンを構築しました")

def predict_fn(question: str) -> str:
    """評価用の予測関数（文字列を返す）"""
    if not question:
        return ""
    
    try:
        result = rag_chain.invoke({"input": question})
        answer = result.get("answer", "")
        return answer
    except Exception as e:
        return f"エラーが発生しました: {str(e)}"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 品質評価スコアラーの定義
# MAGIC
# MAGIC RAGチェーンの品質を評価するためのスコアラーを定義します。
# MAGIC 組み込みのscorersとカスタムスコアラーを使用します。

# COMMAND ----------

# 組み込みのスコアラーを使用
# 正確性評価（組み込みのCorrectnessスコアラーを使用）
accuracy_scorer = Correctness(name="accuracy_evaluator")

# 関連性評価（組み込みのRelevanceToQueryスコアラーを使用）
relevance_scorer = RelevanceToQuery(name="relevance_evaluator")

# 接地性評価（カスタムGuidelinesスコアラーを使用）
grounding_scorer = Guidelines(
    name="grounding_evaluator",
    guidelines=(
        "回答が取得されたコンテキストに基づいているか（接地しているか）を評価してください。\n"
        "以下の観点から評価してください：\n"
        "1. 回答がコンテキストに基づいているか\n"
        "2. 回答がコンテキストから推論できるか\n"
        "3. 回答がコンテキストにない情報を含んでいないか\n\n"
        "評価は以下のスケールで行ってください：\n"
        "- 'excellent': 完全にコンテキストに基づいている\n"
        "- 'good': 主にコンテキストに基づいている\n"
        "- 'acceptable': 一部コンテキストに基づいている\n"
        "- 'poor': コンテキストに基づいていない"
    ),
)

print("品質評価スコアラーを定義しました")

# COMMAND ----------

# MAGIC %md
# MAGIC ## MLflow評価の実行
# MAGIC
# MAGIC 定義した評価データセットとスコアラーを使用してRAGチェーンを評価します。

# COMMAND ----------

with mlflow.start_run(run_name="yona-commuting-allowance-rag-evaluation"):
    mlflow.set_tag("task", "evaluation")
    mlflow.set_tag("evaluation_type", "rag_quality")
    
    mlflow.log_params({
        "llm_endpoint": config.llm_endpoint,
        "vector_search_index": config.vector_index_name,
        "query_embedding_model": config.query_embedding_model,
        "retriever_top_k": config.retriever_top_k,
        "eval_dataset_size": len(eval_dataset),
    })
    
    print("MLflow評価を開始します...")
    start_time = time.time()
    
    # MLflow GenAI評価を実行
    eval_results = genai_evaluate(
        data=eval_dataset,
        predict_fn=predict_fn,
        scorers=[accuracy_scorer, relevance_scorer, grounding_scorer],
    )
    
    elapsed_time = time.time() - start_time
    print(f"評価完了: {elapsed_time:.2f}秒")
    
    # 評価結果をログ
    mlflow.log_metric("evaluation_time_seconds", elapsed_time)
    
    # 評価結果のメトリクスをログ
    if hasattr(eval_results, 'metrics'):
        for metric_name, metric_value in eval_results.metrics.items():
            mlflow.log_metric(metric_name, metric_value)
        print(f"評価メトリクス: {eval_results.metrics}")
    
    # 評価結果のテーブルをログ
    if hasattr(eval_results, 'result_df'):
        # result_dfをテーブルとしてログ
        mlflow.log_table(eval_results.result_df, "eval_results_table.json")
        print("評価結果テーブルをログしました")
    elif hasattr(eval_results, 'tables'):
        for table_name, table_df in eval_results.tables.items():
            mlflow.log_table(table_df, f"{table_name}.json")
            print(f"テーブル '{table_name}' をログしました")
    
    # 評価結果全体をJSONとしても保存
    try:
        import json
        results_dict = {
            "metrics": eval_results.metrics if hasattr(eval_results, 'metrics') else {},
            "summary": str(eval_results),
        }
        mlflow.log_dict(results_dict, "evaluation_summary.json")
    except Exception as e:
        print(f"評価結果のJSON保存でエラー（スキップ）: {e}")
    
    print(f"Run ID: {mlflow.active_run().info.run_id}")
    print(f"評価結果をMLflowに記録しました")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 評価結果の分析と可視化

# COMMAND ----------

# 評価結果をDataFrameに変換
if hasattr(eval_results, 'result_df'):
    # EvaluationResultオブジェクトにはresult_df属性がある
    results_df = eval_results.result_df
    display(results_df.head())
elif hasattr(eval_results, 'tables') and 'eval_results_table' in eval_results.tables:
    # フォールバック: tables属性を使用
    results_df = eval_results.tables['eval_results_table']
    display(results_df.head())
else:
    raise ValueError("評価結果からDataFrameを取得できませんでした")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 個別の評価結果の確認

# COMMAND ----------

def extract_question(row, df_columns):
    """質問を抽出する関数"""
    # request列から取得
    if 'request' in df_columns:
        request = row['request']
        if isinstance(request, dict):
            # request.inputs.question または request.question または request.input
            if 'inputs' in request:
                inputs = request['inputs']
                if isinstance(inputs, dict):
                    return inputs.get('question') or inputs.get('input', '')
                return str(inputs)
            elif 'question' in request:
                return request['question']
            elif 'input' in request:
                return request['input']
    # inputs列から取得
    elif 'inputs' in df_columns:
        inputs = row['inputs']
        if isinstance(inputs, dict):
            return inputs.get('question') or inputs.get('input', '')
        return str(inputs)
    # question列から取得
    elif 'question' in df_columns:
        return row['question']
    return 'N/A'

def extract_answer(row, df_columns):
    """回答を抽出する関数"""
    # response列から取得
    if 'response' in df_columns:
        response = row['response']
        if isinstance(response, dict):
            # response.outputs.answer または response.answer または response.content
            if 'outputs' in response:
                outputs = response['outputs']
                if isinstance(outputs, dict):
                    return outputs.get('answer', '')
                return str(outputs)
            elif 'answer' in response:
                return response['answer']
            elif 'content' in response:
                return response['content']
    # outputs列から取得
    elif 'outputs' in df_columns:
        outputs = row['outputs']
        if isinstance(outputs, dict):
            return outputs.get('answer', '')
        return str(outputs)
    # answer列から取得
    elif 'answer' in df_columns:
        return row['answer']
    return 'N/A'

# 個別の評価結果を表示
for idx, row in results_df.iterrows():
    print(f"\n=== 質問 {idx + 1} ===")
    
    # 質問と回答を抽出
    question = extract_question(row, results_df.columns)
    answer = extract_answer(row, results_df.columns)
    
    # 回答が長い場合は切り詰め
    if isinstance(answer, str) and len(answer) > 300:
        answer = answer[:300] + "..."
    
    print(f"質問: {question}")
    print(f"回答: {answer}")
    
    # 評価スコアを取得
    accuracy_col = next((col for col in results_df.columns if 'accuracy' in col.lower() and '/value' in col), None)
    relevance_col = next((col for col in results_df.columns if 'relevance' in col.lower() and '/value' in col), None)
    grounding_col = next((col for col in results_df.columns if 'grounding' in col.lower() and '/value' in col), None)
    
    # フォールバック
    if not accuracy_col:
        accuracy_col = next((col for col in results_df.columns if 'accuracy' in col.lower()), None)
    if not relevance_col:
        relevance_col = next((col for col in results_df.columns if 'relevance' in col.lower()), None)
    if not grounding_col:
        grounding_col = next((col for col in results_df.columns if 'grounding' in col.lower()), None)
    
    accuracy = row[accuracy_col] if accuracy_col else 'N/A'
    relevance = row[relevance_col] if relevance_col else 'N/A'
    grounding = row[grounding_col] if grounding_col else 'N/A'
    
    print(f"正確性: {accuracy}")
    print(f"関連性: {relevance}")
    print(f"接地性: {grounding}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 評価結果の保存
# MAGIC
# MAGIC 評価結果をDeltaテーブルに保存して、後続の分析や比較に使用できます。

# COMMAND ----------

# 評価結果をDeltaテーブルに保存（オプション）
EVAL_RESULTS_TABLE = f"{config.catalog}.{config.schema}.yona_commuting_allowance_eval_results"

try:
    # 複合型（辞書、リストなど）を含む列を除外または文字列に変換
    results_df_clean = results_df.copy()
    
    # 複合型の列を文字列に変換
    for col in results_df_clean.columns:
        if results_df_clean[col].dtype == 'object':
            # 辞書やリストの場合は文字列に変換
            results_df_clean[col] = results_df_clean[col].apply(
                lambda x: str(x) if isinstance(x, (dict, list)) else x
            )
    
    eval_spark_df = spark.createDataFrame(results_df_clean)
    eval_spark_df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(EVAL_RESULTS_TABLE)
    print(f"評価結果を {EVAL_RESULTS_TABLE} に保存しました")
except Exception as e:
    print(f"テーブル保存エラー（スキップ）: {e}")
    print("複合型を含む列があるため、Deltaテーブルへの保存をスキップしました。")
    print("必要に応じて、結果をCSVやJSON形式で保存してください。")

# COMMAND ----------
