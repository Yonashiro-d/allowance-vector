from typing import Any, Generator, Optional
import uuid
import re
import os

import mlflow
from mlflow.pyfunc import ChatAgent
from mlflow.types.agent import (
    ChatAgentChunk,
    ChatAgentMessage,
    ChatAgentResponse,
    ChatContext,
)
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from databricks_langchain import ChatDatabricks, DatabricksVectorSearch
from langchain_huggingface import HuggingFaceEmbeddings
from rag_config import RAGConfig


class RAGChatAgent(ChatAgent):
    def __init__(self) -> None:
        self.config = RAGConfig()
        self.rag_chain: Optional[Any] = None
        self._ensure_authentication()
    
    def _ensure_authentication(self) -> None:
        databricks_host = os.environ.get("DATABRICKS_HOST")
        databricks_client_id = os.environ.get("DATABRICKS_CLIENT_ID")
        databricks_client_secret = os.environ.get("DATABRICKS_CLIENT_SECRET")
        
        if not databricks_host:
            workspace_url = os.environ.get("DATABRICKS_WORKSPACE_URL")
            if workspace_url:
                os.environ["DATABRICKS_HOST"] = workspace_url
            else:
                raise ValueError(
                    "DATABRICKS_HOST or DATABRICKS_WORKSPACE_URL environment variable must be set. "
                    "Please configure the endpoint with DATABRICKS_HOST environment variable. "
                    "OAuth統合認証を使用する場合は、DATABRICKS_HOSTのみで認証が自動的に処理されます。"
                )
        
        if databricks_client_id and databricks_client_secret:
            os.environ["DATABRICKS_CLIENT_ID"] = databricks_client_id
            os.environ["DATABRICKS_CLIENT_SECRET"] = databricks_client_secret
        
        os.environ.pop("DATABRICKS_TOKEN", None)
    
    def _ensure_rag_chain_initialized(self) -> None:
        if self.rag_chain is None:
            self._initialize_rag_chain()
    
    def _initialize_rag_chain(self) -> None:
        embedding_model = HuggingFaceEmbeddings(model_name=self.config.query_embedding_model)
        
        vector_store = DatabricksVectorSearch(
            index_name=self.config.vector_index_name,
            embedding=embedding_model,
            text_column="chunked_text"
        )
        
        llm = ChatDatabricks(
            endpoint=self.config.llm_endpoint,
            extra_params={"temperature": 0.1}
        )
        
        retriever = vector_store.as_retriever(search_kwargs={"k": self.config.retriever_top_k})
        
        prompt_template = """あなたは質問に答えるアシスタントです。取得したコンテキストの内容をもとに質問に答えてください。一部のコンテキストが無関係な場合、それを回答に利用しないでください。

コンテキスト:
{context}

質問: {input}"""
        prompt = ChatPromptTemplate.from_template(prompt_template)
        
        document_chain = create_stuff_documents_chain(llm, prompt)
        self.rag_chain = create_retrieval_chain(retriever, document_chain)
    
    def _extract_user_message(self, messages: list[ChatAgentMessage]) -> str:
        for message in reversed(messages):
            if message.role == "user":
                return message.content
        return ""
    
    def predict(
        self,
        messages: list[ChatAgentMessage],
        context: Optional[ChatContext] = None,
        custom_inputs: Optional[dict[str, Any]] = None,
    ) -> ChatAgentResponse:
        self._ensure_rag_chain_initialized()
        
        user_message = self._extract_user_message(messages)
        if not user_message:
            return ChatAgentResponse(messages=[
                ChatAgentMessage(
                    id=str(uuid.uuid4()),
                    role="assistant",
                    content="ユーザーメッセージが見つかりませんでした。"
                )
            ])
        
        result = self.rag_chain.invoke({"input": user_message})
        answer = result.get("answer", "")
        
        return ChatAgentResponse(messages=[
            ChatAgentMessage(
                id=str(uuid.uuid4()),
                role="assistant",
                content=answer
            )
        ])
    
    def predict_stream(
        self,
        messages: list[ChatAgentMessage],
        context: Optional[ChatContext] = None,
        custom_inputs: Optional[dict[str, Any]] = None,
    ) -> Generator[ChatAgentChunk, None, None]:
        self._ensure_rag_chain_initialized()
        
        user_message = self._extract_user_message(messages)
        if not user_message:
            yield ChatAgentChunk(
                delta=ChatAgentMessage(
                    id=str(uuid.uuid4()),
                    role="assistant",
                    content="ユーザーメッセージが見つかりませんでした。"
                )
            )
            return
        
        result = self.rag_chain.invoke({"input": user_message})
        answer = result.get("answer", "")
        message_id = str(uuid.uuid4())
        
        sentences = re.split(r'([。！？\n])', answer)
        current_chunk = ""
        
        for part in sentences:
            current_chunk += part
            if len(current_chunk) > 10 or part in ['。', '！', '？', '\n']:
                if current_chunk.strip():
                    yield ChatAgentChunk(
                        delta=ChatAgentMessage(
                            id=message_id,
                            role="assistant",
                            content=current_chunk
                        )
                    )
                    current_chunk = ""
        
        if current_chunk.strip():
            yield ChatAgentChunk(
                delta=ChatAgentMessage(
                    id=message_id,
                    role="assistant",
                    content=current_chunk
                )
            )


AGENT = RAGChatAgent()
mlflow.models.set_model(AGENT)
