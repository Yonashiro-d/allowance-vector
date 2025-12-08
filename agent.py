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
from rag_config import RAGConfig


class RAGChatAgent(ChatAgent):
    """RAG Agent that uses Databricks Vector Search and LLM"""
    
    def __init__(self):
        """Initialize the RAG Agent"""
        super().__init__()
        self.config = None
        self.rag_chain = None
    
    def _initialize_rag_chain(self):
        """Initialize the RAG chain (lazy initialization)"""
        if self.rag_chain is not None:
            return
        
        # Initialize config if not already done
        if self.config is None:
            self.config = RAGConfig()
        
        # Initialize embedding model
        embedding_model = HuggingFaceEmbeddings(model_name=self.config.query_embedding_model)
        
        # Initialize vector store
        vector_store = DatabricksVectorSearch(
            index_name=self.config.vector_index_name,
            embedding=embedding_model,
            text_column="chunked_text",
            columns=["chunk_id", "chunked_text"]
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
        # Lazy initialization of RAG chain
        self._initialize_rag_chain()
        
        # Extract the last user message
        user_message = self._extract_user_message(messages)
        
        if not user_message:
            # Return empty response if no user message found
            return ChatAgentResponse(messages=[
                ChatAgentMessage(
                    id=str(uuid.uuid4()),
                    role="assistant",
                    content="質問を入力してください。"
                )
            ])
        
        # Invoke the RAG chain
        try:
            result = self.rag_chain.invoke({"input": user_message})
            answer = result.get("answer", "")
        except Exception as e:
            # Return error message if RAG chain fails
            answer = f"エラーが発生しました: {str(e)}"
        
        # Create ChatAgentMessage with the answer
        response_message = ChatAgentMessage(
            id=str(uuid.uuid4()),
            role="assistant",
            content=answer
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
        # Lazy initialization of RAG chain
        self._initialize_rag_chain()
        
        # Extract the last user message
        user_message = self._extract_user_message(messages)
        
        if not user_message:
            # Return empty response if no user message found
            yield ChatAgentChunk(
                delta=ChatAgentMessage(
                    id=str(uuid.uuid4()),
                    role="assistant",
                    content="質問を入力してください。"
                )
            )
            return
        
        # Invoke the RAG chain
        try:
            result = self.rag_chain.invoke({"input": user_message})
            answer = result.get("answer", "")
        except Exception as e:
            # Return error message if RAG chain fails
            answer = f"エラーが発生しました: {str(e)}"
        
        # Stream the answer word by word (simple implementation)
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
        
        # Yield remaining chunk if any
        if current_chunk.strip():
            yield ChatAgentChunk(
                delta=ChatAgentMessage(
                    id=message_id,
                    role="assistant",
                    content=current_chunk
                )
            )


# Create the agent object, and specify it as the agent object to use when
# loading the agent back for inference via mlflow.models.set_model()
AGENT = RAGChatAgent()
mlflow.models.set_model(AGENT)
