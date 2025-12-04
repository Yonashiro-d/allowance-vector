from operator import itemgetter
import mlflow
import os

from databricks.vector_search.client import VectorSearchClient

#from langchain_community.chat_models import ChatDatabricks
#from langchain_community.vectorstores import DatabricksVectorSearch
from databricks_langchain import ChatDatabricks, UCFunctionToolkit, VectorSearchRetrieverTool, DatabricksVectorSearch

from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.runnables import RunnablePassthrough, RunnableBranch
from langchain_core.messages import HumanMessage, AIMessage
from databricks_langchain import DatabricksEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

print("!!!!!!1!!!!!!")
## Enable MLflow Tracing
mlflow.langchain.autolog()


### ユーザーの質問を受け取る
# ユーザーから最新のメッセージ (最新の質問) を取得
def extract_user_query_string(chat_messages_array):
    return chat_messages_array[-1]["content"]


# 最新のメッセージ (現在の質問の前の会話履歴) を除くすべての以前のメッセージを取得
def extract_chat_history(chat_messages_array):
    return chat_messages_array[:-1]


###
def refine_query(query):
    # return f"必ず通勤距離、通勤経路、通勤方法、通勤手当の額に関する内容を検索してください。{query}"
    return f"{query}"


# def __init__(self, embedding):
#     self.embedding = embedding
#     # Use self.embedding when constructing vector search or other components

# 設定ファイル（YAML）をPythonで読み込む
model_config = mlflow.models.ModelConfig(development_config="XXX.yaml")


print("!!!!!!2!!!!!!")
vs_client = VectorSearchClient(disable_notice=True)
# print(model_config.get("vector_search_index"))
# vs_index = vs_client.get_index(
#     endpoint_name=model_config.get("vector_search_endpoint_name"),
#     index_name=model_config.get("vector_search_index"),
# )

print("!!!!!!3!!!!!!")

# Example: create embedding object with your desired model
# embedding = DatabricksEmbeddings(endpoint="cl-nagoya/ruri-v3-310m")
embedding_model = HuggingFaceEmbeddings(model_name="cl-nagoya/ruri-v3-310m")


print("!!!!!!4!!!!!!")
# def vector_search_as_retriever(question):
#     print(f"!!!!!!!!!!!!{question}")
#     vector_search_as_rule = DatabricksVectorSearch(
#         vs_index,
#         embedding=embedding_model,
#         text_column="chunked_text",
#         columns=[
#             "chunk_id",
#             "chunked_text",
#             "embedding",
#         ],
#     ).as_retriever(
#         search_kwargs={"k": 6, "score_threshold": 0.99}
#     )  # 最大 6 つの結果が返される。類似性スコアが 0.7 以上の結果だけが返される
#     result = vector_search_as_rule.invoke(question)
#     print(f"!!!!!!!!!{result}")
#     return result
#         | RunnableLambda(vector_search_as_retriever)

    #vs_index,
#    endpoint_name=model_config.get("vector_search_endpoint_name"),
vector_search_as_retriever = DatabricksVectorSearch(
    index_name=model_config.get("vector_search_index"),
    embedding=embedding_model,
    text_column="chunked_text",
    columns=[
        "chunk_id",
        "chunked_text",
    ],
).as_retriever(
    search_kwargs={"k": 6, "score_threshold": 0.003}
)  # 最大 6 つの結果が返される。類似性スコアが 0.7 以上の結果だけが返される

print("!!!!!!5!!!!!!")

mlflow.models.set_retriever_schema(
    primary_key="chunk_id",
    text_column="chunked_text",
    doc_uri="url",  # Review App uses `doc_uri` to display chunks from the same document in a single view
)

print("!!!!!!6!!!!!!")

def format_context(docs):
    chunk_template = "{chunk_text}"
    chunk_contents = [
        chunk_template.format(
            chunk_text=d.page_content,
        )
        for d in docs
    ]
    return f"会社規定：{''.join(chunk_contents)}"  # まとめた文章の先頭に「会社規定：」という文字をつけて返す
    # vector_search_as_retrieverの後にこれを呼ぶと、検索した結果の先頭に会社規定：ってつく？

print("!!!!!!7!!!!!!")

prompt = ChatPromptTemplate.from_messages(
    [
        (  # System prompt contains the instructions and company policy context
            "system",
            model_config.get("llm_prompt_template") + "\n\n以下は会社規定の内容です。\n\n{context}",
        ),  #   "\n\n以下は会社規定の内容です。必ずこの情報に基づいて回答してください。\n\n{context}",
        MessagesPlaceholder(variable_name="formatted_chat_history"),
        ("user", "{question}"),
    ]
)

print("!!!!!!8!!!!!!")


# 会話履歴をAIが理解しやすい「HumanMessage」や「AIMessage」という形式に変換して、プロンプト（AIへの指示文）に使えるようにする
def format_chat_history_for_prompt(chat_messages_array):
    history = extract_chat_history(chat_messages_array)
    formatted_chat_history = []
    if len(history) > 0:
        for chat_message in history:
            if chat_message["role"] == "user":
                formatted_chat_history.append(
                    HumanMessage(content=chat_message["content"])
                )
            elif chat_message["role"] == "assistant":
                formatted_chat_history.append(
                    AIMessage(content=chat_message["content"])
                )
    return formatted_chat_history


# # これがなかったらどうなるのか→今回のサンプルでは何も変わらなかった。プロンプトが変わるからもっと難しい質問の時には意味があるのかも。
# query_rewrite_template = """Based on the chat history below, we want you to generate a query for an external data source to retrieve relevant documents so that we can better answer the question. The query should be in natural language. The external data source uses similarity search to search for relevant documents in a vector space. So the query should be similar to the relevant documents semantically. Answer with only the query. Do not add explanation.

# Chat history: {chat_history}

# Question: {question}"""


#-----------------------------------------
from langchain_core.runnables import RunnableLambda

# def add_session_id_to_output(output_dict):
#     # output_dictには "output"（AIの回答）と "session_id" が含まれる前提
#     return f"{output_dict['output']}\n\n[セッションID: {output_dict['session_id']}]"

# # # 既存のchainの後ろに合成用のRunnableLambdaを追加
# chain_with_session_id = (
#     chain
#     | (lambda output, session_id: {"output": output, "session_id": session_id})
#     | RunnableLambda(add_session_id_to_output)
# )
def prepend_session_id_to_output(output, inputs):
    # inputsにはリクエストボディ全体が渡されるので、session_idを取得
    session_id = inputs.get("session_id", "")
    print(f"session_id: {session_id}")
    return f"session_id: {session_id}"

#-----------------------------------------

print("!!!!!!9!!!!!!")

model = ChatDatabricks(
    endpoint=model_config.get("llm_model_serving_endpoint_name"),
    extra_params={"temperature": 0.01},  # AIの回答の「ランダム性（創造性）」を調整するパラメータ
)
chain = (
    {
        "question": itemgetter("messages")
        | RunnableLambda(extract_user_query_string)
        | RunnableLambda(refine_query),
        "chat_history": itemgetter("messages") | RunnableLambda(extract_chat_history),
        "formatted_chat_history": itemgetter("messages")
        | RunnableLambda(format_chat_history_for_prompt),
        "session_id": itemgetter("session_id"),
    }
    | RunnablePassthrough()
    | {
        "context": itemgetter("question")
        | vector_search_as_retriever
        | RunnableLambda(format_context),
        "formatted_chat_history": itemgetter("formatted_chat_history"),
        "question": itemgetter("question"),
        "session_id": itemgetter("session_id"),
    }
    | prompt
    | model
    | StrOutputParser()
    | RunnableLambda(prepend_session_id_to_output)
)
print(chain)
mlflow.models.set_model(model=chain)