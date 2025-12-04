"""
Streamlitチャットボットアプリ
Databricks上で実行可能なRAGチャットボット
"""
import streamlit as st
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from rag_client import RAGClient, RAGConfig

# ページ設定
st.set_page_config(
    page_title="通勤手当規程チャットボット",
    page_icon="💬",
    layout="wide"
)

# タイトル
st.title("💬 通勤手当支給規程チャットボット")
st.markdown("通勤手当支給規程に関する質問にお答えします。")

# セッション状態の初期化
if "messages" not in st.session_state:
    st.session_state.messages = []

if "rag_client" not in st.session_state:
    with st.spinner("RAGクライアントを初期化中..."):
        try:
            config = RAGConfig()
            st.session_state.rag_client = RAGClient(config)
            st.session_state.rag_client._initialize()
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
    if st.session_state.rag_client:
        config = st.session_state.rag_client.config
        st.markdown(f"**カタログ**: {config.catalog}")
        st.markdown(f"**スキーマ**: {config.schema}")
        st.markdown(f"**インデックス**: {config.vector_index_name}")
        st.markdown(f"**LLM**: {config.llm_endpoint}")

# チャット履歴の表示
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # ソース情報を表示
        if message["role"] == "assistant" and "sources" in message and message["sources"]:
            with st.expander(f"参照元 ({message.get('num_sources', 0)}件)"):
                for i, source in enumerate(message["sources"], 1):
                    st.markdown(f"**参照{i}**")
                    st.text(source.get("content", "")[:300])
                    if source.get("metadata"):
                        st.caption(f"メタデータ: {source['metadata']}")

# ユーザー入力
if prompt := st.chat_input("質問を入力してください"):
    # ユーザーメッセージを追加
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # ユーザーメッセージを表示
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # アシスタントの回答を生成
    with st.chat_message("assistant"):
        with st.spinner("回答を生成中..."):
            try:
                # 会話履歴を準備（最後のユーザーメッセージを除く）
                chat_history = [
                    {"role": msg["role"], "content": msg["content"]}
                    for msg in st.session_state.messages[:-1]
                ]
                
                # RAGクエリを実行
                result = st.session_state.rag_client.query(prompt, chat_history)
                
                # 回答を表示
                st.markdown(result["answer"])
                
                # ソース情報を表示
                if result.get("sources"):
                    with st.expander(f"参照元 ({result.get('num_sources', 0)}件)"):
                        for i, source in enumerate(result["sources"], 1):
                            st.markdown(f"**参照{i}**")
                            st.text(source.get("content", "")[:300])
                            if source.get("metadata"):
                                st.caption(f"メタデータ: {source['metadata']}")
                
                # アシスタントメッセージを追加
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result.get("sources", []),
                    "num_sources": result.get("num_sources", 0)
                })
                
            except Exception as e:
                error_msg = f"エラーが発生しました: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg
                })

