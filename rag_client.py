"""
RAGクライアントAPI
会話履歴対応のRAGクエリ機能を提供
"""
from typing import Dict, Any, List, Optional
from databricks_langchain import ChatDatabricks, DatabricksVectorSearch
from langchain_huggingface import HuggingFaceEmbeddings
from rag_config import RAGConfig


class RAGClient:
    """RAGクライアントクラス"""
    
    def __init__(self, config: Optional[RAGConfig] = None):
        self.config = config or RAGConfig()
        self.vector_store = None
        self.llm = None
        self._initialized = False
    
    def _initialize(self):
        """ベクトルストアとLLMを初期化"""
        if self._initialized:
            return
        
        embedding_model = HuggingFaceEmbeddings(model_name=self.config.query_embedding_model)
        
        self.vector_store = DatabricksVectorSearch(
            index_name=self.config.vector_index_name,
            embedding=embedding_model,
            text_column="chunked_text",
            columns=["chunk_id", "chunked_text"]
        )
        
        self.llm = ChatDatabricks(
            endpoint=self.config.llm_endpoint,
            extra_params={"temperature": 0.1}
        )
        
        self._initialized = True
    
    def _build_prompt(self, question: str, context: str, chat_history: Optional[List[Dict[str, str]]] = None) -> str:
        """プロンプトを構築"""
        prompt_parts = [
            "以下のコンテキスト情報を使用して、質問に答えてください。",
            "コンテキスト情報に基づいて回答し、コンテキストにない情報は推測せずに「わかりません」と答えてください。",
            "",
            "コンテキスト情報:",
            context,
            ""
        ]
        
        if chat_history:
            prompt_parts.extend([
                "過去の会話履歴:",
                "\n".join([f"{msg.get('role', 'user')}: {msg.get('content', '')}" for msg in chat_history[-5:]]),
                ""
            ])
        
        prompt_parts.extend([
            f"質問: {question}",
            "",
            "回答:"
        ])
        
        return "\n".join(prompt_parts)
    
    def query(
        self,
        question: str,
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        RAGクエリを実行
        
        Args:
            question: 質問文
            chat_history: 会話履歴 [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
        
        Returns:
            {
                "answer": str,
                "sources": List[Dict],
                "num_sources": int,
                "question": str
            }
        """
        if not self._initialized:
            self._initialize()
        
        try:
            documents = self.vector_store.similarity_search(question, k=self.config.retriever_top_k)
            context = "\n\n".join([doc.page_content for doc in documents])
            
            prompt = self._build_prompt(question, context, chat_history)
            
            response = self.llm.invoke(prompt)
            answer = response.content if hasattr(response, 'content') else str(response)
            
            sources = [
                {
                    "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                    "metadata": doc.metadata
                }
                for doc in documents
            ]
            
            return {
                "question": question,
                "answer": answer,
                "sources": sources,
                "num_sources": len(sources)
            }
        except Exception as e:
            return {
                "question": question,
                "answer": f"エラーが発生しました: {str(e)}",
                "sources": [],
                "num_sources": 0,
                "error": str(e)
            }
    
    def query_chat_completion(
        self,
        messages: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """
        チャット補完形式でRAGクエリを実行
        
        Args:
            messages: メッセージリスト [{"role": "user", "content": "..."}]
        
        Returns:
            {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": str
                    }
                }]
            }
        """
        if not messages:
            return {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": "メッセージが見つかりませんでした。"
                    }
                }]
            }
        
        last_message = messages[-1]
        question = last_message.get("content", "") if isinstance(last_message, dict) else str(last_message)
        
        if not question:
            return {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": "質問内容が空です。"
                    }
                }]
            }
        
        chat_history = messages[:-1] if len(messages) > 1 else None
        result = self.query(question, chat_history)
        
        return {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": result["answer"]
                }
            }]
        }

