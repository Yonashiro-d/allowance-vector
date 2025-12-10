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

# MAGIC %pip install mlflow>=3.4.0 langchain==0.3.27 langchain-core==0.3.75 langchain-community langchain-huggingface==0.3.1 databricks-langchain==0.7.1 databricks-vectorsearch==0.57 databricks-sdk==0.65.0 sentence-transformers==5.1.0 "transformers>=4.49.0,<4.51" "tokenizers>=0.21.2,<0.22" "sentencepiece>=0.2.0,<0.3" torch==2.4.1 numpy==1.26.4 pandas==2.2.3

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
    if hasattr(eval_results, 'tables'):
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
if hasattr(eval_results, 'tables') and 'eval_results_table' in eval_results.tables:
    results_df = eval_results.tables['eval_results_table']
    print("評価結果のサマリー:")
    print(results_df.describe())
    
    # 評価結果を表示
    display(results_df)
else:
    # フォールバック: eval_resultsを直接DataFrameに変換
    try:
        results_df = pd.DataFrame(eval_results)
        print("評価結果のサマリー:")
        print(results_df.describe())
        display(results_df)
    except Exception as e:
        print(f"評価結果のDataFrame変換でエラー: {e}")
        print(f"評価結果の型: {type(eval_results)}")
        print(f"評価結果: {eval_results}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## カテゴリ別の評価結果

# COMMAND ----------

# カテゴリ別に評価結果を集計
try:
    # tagsカラムがある場合
    if "tags" in results_df.columns:
        # tagsが辞書の場合は展開
        if results_df["tags"].dtype == 'object':
            tags_df = pd.json_normalize(results_df["tags"])
            results_df_expanded = pd.concat([results_df, tags_df], axis=1)
            
            # カテゴリ別に集計
            if "category" in tags_df.columns:
                category_results = results_df_expanded.groupby("category").agg({
                    col: ["mean", "std"] 
                    for col in results_df_expanded.columns 
                    if "evaluator" in col or "judge" in col.lower()
                })
                print("\nカテゴリ別評価結果:")
                display(category_results)
    else:
        print("tagsカラムが見つかりません。評価データセットにtagsが含まれているか確認してください。")
except Exception as e:
    print(f"カテゴリ別集計でエラー: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 個別の評価結果の確認

# COMMAND ----------

try:
    for idx, row in results_df.iterrows():
        print(f"\n=== 質問 {idx + 1} ===")
        
        # 入力の取得
        inputs = row.get('inputs', {})
        if isinstance(inputs, dict):
            question = inputs.get('question', 'N/A')
        else:
            question = str(inputs)
        print(f"質問: {question}")
        
        # 出力の取得
        outputs = row.get('outputs', {})
        if isinstance(outputs, dict):
            answer = outputs.get('answer', 'N/A')
            if isinstance(answer, str) and len(answer) > 200:
                answer = answer[:200] + "..."
        else:
            answer = str(outputs)[:200] + "..." if len(str(outputs)) > 200 else str(outputs)
        print(f"回答: {answer}")
        
        # 評価スコアの取得
        accuracy = row.get('accuracy_evaluator', 'N/A')
        relevance = row.get('relevance_evaluator', 'N/A')
        grounding = row.get('grounding_evaluator', 'N/A')
        
        # スコアが辞書の場合はvalueを取得
        if isinstance(accuracy, dict):
            accuracy = accuracy.get('value', accuracy)
        if isinstance(relevance, dict):
            relevance = relevance.get('value', relevance)
        if isinstance(grounding, dict):
            grounding = grounding.get('value', grounding)
        
        print(f"正確性: {accuracy}")
        print(f"関連性: {relevance}")
        print(f"接地性: {grounding}")
except Exception as e:
    print(f"個別結果の表示でエラー: {e}")
    print("評価結果の構造を確認してください")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 評価結果の保存
# MAGIC
# MAGIC 評価結果をDeltaテーブルに保存して、後続の分析や比較に使用できます。

# COMMAND ----------

# 評価結果をDeltaテーブルに保存（オプション）
EVAL_RESULTS_TABLE = f"{config.catalog}.{config.schema}.yona_commuting_allowance_eval_results"

try:
    eval_spark_df = spark.createDataFrame(results_df)
    eval_spark_df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(EVAL_RESULTS_TABLE)
    print(f"評価結果を {EVAL_RESULTS_TABLE} に保存しました")
except Exception as e:
    print(f"テーブル保存エラー（スキップ）: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 改善提案
# MAGIC
# MAGIC 評価結果に基づいて、以下の改善点を検討してください：
# MAGIC
# MAGIC 1. **低スコアの質問を特定**: 評価スコアが低い質問を特定し、原因を分析
# MAGIC 2. **コンテキストの改善**: 関連性スコアが低い場合は、チャンキング方法やベクトル検索のパラメータを調整
# MAGIC 3. **プロンプトの改善**: 接地性スコアが低い場合は、プロンプトテンプレートを改善
# MAGIC 4. **評価データセットの拡充**: より多様な質問を追加して評価の網羅性を向上
# MAGIC
# MAGIC 参考: [Databricks生成AIアプリ開発ワークフロー - 評価と反復](https://docs.databricks.com/aws/ja/generative-ai/tutorials/ai-cookbook/genai-developer-workflow#2--評価と反復)

