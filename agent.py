"""
RAG Agent for Commuting Allowance Questions

This agent uses Databricks Vector Search and LLM to answer questions
about commuting allowance policies.
"""

from typing import Any, Generator, Optional
import uuid

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
from databricks.vector_search.client import VectorSearchClient
from databricks.vector_search.utils import CredentialStrategy
from rag_config import RAGConfig


class RAGChatAgent(ChatAgent):
    """RAG Agent that uses Databricks Vector Search and LLM"""
    
    def __init__(self):
        """Initialize the RAG Agent"""
        self.config = RAGConfig()
        self.rag_chain = None
        # RAGチェーンの初期化は遅延初期化（predictメソッド内で実行）
        # これにより、モデルサービング環境での初期化タイミングの問題を回避
    
    def _ensure_rag_chain_initialized(self):
        """RAGチェーンが初期化されていない場合、初期化する（遅延初期化）"""
        if self.rag_chain is None:
            self._initialize_rag_chain()
    
    def _initialize_rag_chain(self):
        """Initialize the RAG chain"""
        # Initialize embedding model
        embedding_model = HuggingFaceEmbeddings(model_name=self.config.query_embedding_model)
        
        # Initialize VectorSearchClient with model serving user credentials
        # This ensures proper authentication in model serving environment
        vector_search_client = VectorSearchClient(
            credential_strategy=CredentialStrategy.MODEL_SERVING_USER_CREDENTIALS
        )
        
        # Initialize vector store
        vector_store = DatabricksVectorSearch(
            index_name=self.config.vector_index_name,
            embedding=embedding_model,
            text_column="chunked_text",
            columns=["chunk_id", "chunked_text"],
            vector_search_client=vector_search_client
        )
        
        # Initialize LLM
        llm = ChatDatabricks(
            endpoint=self.config.llm_endpoint,
            extra_params={"temperature": 0.1}
        )
        
        # Create retriever
        retriever = vector_store.as_retriever(search_kwargs={"k": self.config.retriever_top_k})
        
        # Create prompt
        prompt_template = """あなたは質問に答えるアシスタントです。取得したコンテキストの内容をもとに質問に答えてください。一部のコンテキストが無関係な場合、それを回答に利用しないでください。

コンテキスト:
{context}

質問: {input}"""
        prompt = ChatPromptTemplate.from_template(prompt_template)
        
        # Create document chain and RAG chain
        document_chain = create_stuff_documents_chain(llm, prompt)
        self.rag_chain = create_retrieval_chain(retriever, document_chain)
    
    def _extract_user_message(self, messages: list[ChatAgentMessage]) -> str:
        """Extract the last user message content"""
        # Get the last user message
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
        """Predict using the RAG chain and return ChatAgentResponse
        
        Args:
            messages: List of ChatAgentMessage objects
            context: Optional ChatContext
            custom_inputs: Optional custom inputs
            
        Returns:
            ChatAgentResponse with the answer
        """
        # Ensure RAG chain is initialized (lazy initialization)
        self._ensure_rag_chain_initialized()
        
        # Extract the last user message
        user_message = self._extract_user_message(messages)
        
        # Invoke the RAG chain
        result = self.rag_chain.invoke({"input": user_message})
        
        # Create ChatAgentMessage with the answer
        response_message = ChatAgentMessage(
            id=str(uuid.uuid4()),
            role="assistant",
            content=result.get("answer", "")
        )
        
        return ChatAgentResponse(messages=[response_message])
    
    def predict_stream(
        self,
        messages: list[ChatAgentMessage],
        context: Optional[ChatContext] = None,
        custom_inputs: Optional[dict[str, Any]] = None,
    ) -> Generator[ChatAgentChunk, None, None]:
        """Stream predictions using the RAG chain
        
        Args:
            messages: List of ChatAgentMessage objects
            context: Optional ChatContext
            custom_inputs: Optional custom inputs
            
        Yields:
            ChatAgentChunk objects
        """
        # Ensure RAG chain is initialized (lazy initialization)
        self._ensure_rag_chain_initialized()
        
        # Extract the last user message
        user_message = self._extract_user_message(messages)
        
        # Invoke the RAG chain
        result = self.rag_chain.invoke({"input": user_message})
        
        # Stream the answer word by word (simple implementation)
        answer = result.get("answer", "")
        message_id = str(uuid.uuid4())
        
        # Split answer into chunks (by sentences for better streaming)
        import re
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


# Create the agent object, and specify it as the agent object to use when
# loading the agent back for inference via mlflow.models.set_model()
AGENT = RAGChatAgent()
mlflow.models.set_model(AGENT)
