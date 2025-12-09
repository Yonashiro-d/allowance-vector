from typing import Optional
import os


class RAGConfig:
    def __init__(
        self,
        catalog: Optional[str] = None,
        schema: Optional[str] = None,
        vector_index_name: Optional[str] = None,
        query_embedding_model: Optional[str] = None,
        llm_endpoint: Optional[str] = None,
        serving_endpoint_name: Optional[str] = None,
        retriever_top_k: int = 5
    ):
        self.catalog = catalog or os.environ.get("CATALOG", "hhhd_demo_itec")
        self.schema = schema or os.environ.get("SCHEMA", "allowance_payment_rules")
        self.vector_index_name = vector_index_name or os.environ.get(
            "VECTOR_INDEX_NAME",
            f"{self.catalog}.{self.schema}.commuting_allowance_index"
        )
        self.query_embedding_model = query_embedding_model or os.environ.get(
            "QUERY_EMBEDDING_MODEL",
            "cl-nagoya/ruri-v3-310m"
        )
        # LLMエンドポイント（Foundation Model APIエンドポイント）
        # 環境変数LLM_ENDPOINTが設定されていない場合は、通常のエンドポイントを使用
        self.llm_endpoint = llm_endpoint or os.environ.get(
            "LLM_ENDPOINT",
            "databricks-llama-4-maverick"  # 通常のFoundation Model APIエンドポイント
        )
        self.serving_endpoint_name = serving_endpoint_name or os.environ.get(
            "SERVING_ENDPOINT_NAME",
            "commuting-allowance-rag-endpoint"
        )
        self.retriever_top_k = retriever_top_k if retriever_top_k else int(
            os.environ.get("RETRIEVER_TOP_K", "5")
        )
    
    def to_dict(self) -> dict:
        return {
            "CATALOG": self.catalog,
            "SCHEMA": self.schema,
            "VECTOR_INDEX_NAME": self.vector_index_name,
            "QUERY_EMBEDDING_MODEL": self.query_embedding_model,
            "LLM_ENDPOINT": self.llm_endpoint,
            "SERVING_ENDPOINT_NAME": self.serving_endpoint_name,
            "RETRIEVER_TOP_K": str(self.retriever_top_k)
        }
    
    def __repr__(self):
        return f"RAGConfig(catalog={self.catalog}, schema={self.schema}, index={self.vector_index_name})"

