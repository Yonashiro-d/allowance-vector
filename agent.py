"""
RAG Agent for Commuting Allowance Questions

This agent uses Databricks Vector Search and LLM to answer questions
about commuting allowance policies.
"""

from typing import Any, Dict, List
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from databricks_langchain import ChatDatabricks, DatabricksVectorSearch
from langchain_huggingface import HuggingFaceEmbeddings
from rag_config import RAGConfig


# Configuration
config = RAGConfig()
LLM_ENDPOINT_NAME = config.llm_endpoint
VECTOR_SEARCH_INDEX = config.vector_index_name
QUERY_EMBEDDING_MODEL = config.query_embedding_model
RETRIEVER_TOP_K = config.retriever_top_k

# Prompt template
PROMPT_TEMPLATE = """あなたは質問に答えるアシスタントです。取得したコンテキストの内容をもとに質問に答えてください。一部のコンテキストが無関係な場合、それを回答に利用しないでください。

コンテキスト:
{context}

質問: {input}"""


def messages_to_rag_input(messages: Dict[str, Any]) -> Dict[str, str]:
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


def build_rag_chain():
    """Build RAG chain with Vector Search and LLM"""
    # Initialize embedding model
    embedding_model = HuggingFaceEmbeddings(model_name=QUERY_EMBEDDING_MODEL)
    
    # Initialize vector store
    vector_store = DatabricksVectorSearch(
        index_name=VECTOR_SEARCH_INDEX,
        embedding=embedding_model,
        text_column="chunked_text",
        columns=["chunk_id", "chunked_text"]
    )
    
    # Initialize LLM
    llm = ChatDatabricks(
        endpoint=LLM_ENDPOINT_NAME,
        extra_params={"temperature": 0.1}
    )
    
    # Create retriever
    retriever = vector_store.as_retriever(search_kwargs={"k": RETRIEVER_TOP_K})
    
    # Create prompt
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    
    # Create document chain and RAG chain
    document_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, document_chain)
    
    # Wrap with message format converter
    agent_chain = RunnableLambda(messages_to_rag_input) | rag_chain
    
    return agent_chain


# Initialize agent
AGENT = build_rag_chain()

