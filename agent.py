"""
RAG Agent for Commuting Allowance Questions

This agent uses Databricks Vector Search and LLM to answer questions
about commuting allowance policies.
"""

from typing import Any, Dict, List
import mlflow
from mlflow.pyfunc import PythonModel, PythonModelContext
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from databricks_langchain import ChatDatabricks, DatabricksVectorSearch
from langchain_huggingface import HuggingFaceEmbeddings
from rag_config import RAGConfig


class RAGAgent(PythonModel):
    """RAG Agent that uses Databricks Vector Search and LLM"""
    
    def __init__(self):
        """Initialize the RAG Agent"""
        self.config = RAGConfig()
        self.rag_chain = None
    
    def load_context(self, context: PythonModelContext):
        """Load the RAG chain when the model is loaded"""
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
        rag_chain = create_retrieval_chain(retriever, document_chain)
        
        # Wrap with message format converter
        self.rag_chain = RunnableLambda(self._messages_to_rag_input) | rag_chain
    
    def _messages_to_rag_input(self, messages: Dict[str, Any]) -> Dict[str, str]:
        """Convert agent messages format to RAG chain input format"""
        if isinstance(messages, dict) and "messages" in messages:
            messages_list = messages["messages"]
        elif isinstance(messages, list):
            messages_list = messages
        else:
            raise ValueError(f"Unexpected input format: {type(messages)}")
        
        # Get the last user message
        last_message = messages_list[-1] if messages_list else None
        if last_message:
            if isinstance(last_message, dict):
                content = last_message.get("content", "")
            elif hasattr(last_message, "content"):
                content = last_message.content
            else:
                content = str(last_message)
            return {"input": content}
        return {"input": ""}
    
    def predict(self, context: PythonModelContext, model_input: Dict[str, Any]) -> Dict[str, Any]:
        if self.rag_chain is None:
            self.load_context(context)
        
        # Invoke the RAG chain
        result = self.rag_chain.invoke(model_input)
        
        # Return the answer in a format compatible with agent framework
        # Keep the original messages and append the assistant's response
        response_messages = model_input.get("messages", []) + [
            {
                "role": "assistant",
                "content": result.get("answer", "")
            }
        ]
        
        return {
            "messages": response_messages
        }


# Initialize and set the agent
AGENT = RAGAgent()
mlflow.models.set_model(AGENT)
