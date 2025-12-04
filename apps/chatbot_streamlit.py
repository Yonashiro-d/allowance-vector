import streamlit as st
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from rag_config import RAGConfig

st.set_page_config(
    page_title="通勤手当規程チャットボット",
    page_icon="💬",
    layout="wide"
)

st.title("💬 通勤手当支給規程チャットボット")
st.markdown("通勤手当支給規程に関する質問にお答えします。")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "deploy_client" not in st.session_state:
    with st.spinner("エンドポイントクライアントを初期化中..."):
        try:
            import mlflow.deployments
            config = RAGConfig()
            st.session_state.config = config
            st.session_state.deploy_client = mlflow.deployments.get_deploy_client("databricks")
        except Exception as e:
            st.error(f"初期化エラー: {str(e)}")
            st.stop()

# サイドバー
with st.sidebar:
    st.header("設定")
    
    if st.button("会話履歴をクリア", type="secondary"):
        st.session_state.messages = []
        st.rerun()
    
    st.markdown("---")
    st.markdown("### 使い方")
    st.markdown("""
    1. 下のテキストボックスに質問を入力
    2. 「送信」ボタンをクリック
    3. AIが規程文書に基づいて回答します
    """)
    
    st.markdown("---")
    st.markdown("### 情報")
    if st.session_state.config:
        config = st.session_state.config
        st.markdown(f"**エンドポイント**: {config.serving_endpoint_name}")
        st.markdown(f"**カタログ**: {config.catalog}")
        st.markdown(f"**スキーマ**: {config.schema}")
        st.markdown(f"**インデックス**: {config.vector_index_name}")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("質問を入力してください"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("回答を生成中..."):
            try:
                messages = [
                    {"role": msg["role"], "content": msg["content"]}
                    for msg in st.session_state.messages
                ]
                
                response = st.session_state.deploy_client.predict(
                    endpoint=st.session_state.config.serving_endpoint_name,
                    inputs={"messages": messages}
                )
                
                if isinstance(response, dict) and "choices" in response:
                    answer = response["choices"][0]["message"]["content"]
                elif isinstance(response, dict) and "predictions" in response:
                    answer = response["predictions"][0] if isinstance(response["predictions"], list) else str(response["predictions"])
                else:
                    answer = str(response)
                
                st.markdown(answer)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer
                })
                
            except Exception as e:
                error_msg = f"エラーが発生しました: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg
                })

