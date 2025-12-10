from typing import Optional
import os


class RAGConfig:
    def __init__(
        self,
        catalog: Optional[str] = None,
        schema: Optional[str] = None,
        delta_table_name: Optional[str] = None,
        vector_index_name: Optional[str] = None,
        vector_search_endpoint: Optional[str] = None,
        query_embedding_model: Optional[str] = None,
        llm_endpoint: Optional[str] = None,
        serving_endpoint_name: Optional[str] = None,
        retriever_top_k: int = 5
    ):
        # Unity Catalog設定（vector_preparation.pyと統一）
        self.catalog = catalog or os.environ.get("CATALOG", "hhhd_demo_itec")
        self.schema = schema or os.environ.get("SCHEMA", "yona_allowance_payment_rules")
        
        # Deltaテーブル名（vector_preparation.pyと統一）
        self.delta_table_name = delta_table_name or os.environ.get(
            "DELTA_TABLE_NAME",
            f"{self.catalog}.{self.schema}.yona_commuting_allowance_vectors"
        )
        
        # Vector Search設定（vector_preparation.pyと統一）
        self.vector_index_name = vector_index_name or os.environ.get(
            "VECTOR_INDEX_NAME",
            f"{self.catalog}.{self.schema}.yona_commuting_allowance_index"
        )
        self.vector_search_endpoint = vector_search_endpoint or os.environ.get(
            "VECTOR_SEARCH_ENDPOINT",
            "yona_commuting_vector_search"
        )
        
        # Embeddingモデル（vector_preparation.pyと統一）
        self.query_embedding_model = query_embedding_model or os.environ.get(
            "QUERY_EMBEDDING_MODEL",
            "cl-nagoya/ruri-v3-310m"
        )
        
        # LLMエンドポイント（Foundation Model APIエンドポイント）
        # 環境変数LLM_ENDPOINTが設定されていない場合は、通常のエンドポイントを使用
        self.llm_endpoint = llm_endpoint or os.environ.get(
            "LLM_ENDPOINT",
            "databricks-llama-4-maverick"
        )
        
        # サービングエンドポイント名
        self.serving_endpoint_name = serving_endpoint_name or os.environ.get(
            "SERVING_ENDPOINT_NAME",
            "yona-commuting-allowance-rag-endpoint"
        )
        
        # Retriever設定
        self.retriever_top_k = retriever_top_k if retriever_top_k else int(
            os.environ.get("RETRIEVER_TOP_K", "5")
        )
    
    def to_dict(self) -> dict:
        return {
            "CATALOG": self.catalog,
            "SCHEMA": self.schema,
            "DELTA_TABLE_NAME": self.delta_table_name,
            "VECTOR_INDEX_NAME": self.vector_index_name,
            "VECTOR_SEARCH_ENDPOINT": self.vector_search_endpoint,
            "QUERY_EMBEDDING_MODEL": self.query_embedding_model,
            "LLM_ENDPOINT": self.llm_endpoint,
            "SERVING_ENDPOINT_NAME": self.serving_endpoint_name,
            "RETRIEVER_TOP_K": str(self.retriever_top_k)
        }
    
    def __repr__(self):
        return f"RAGConfig(catalog={self.catalog}, schema={self.schema}, table={self.delta_table_name}, index={self.vector_index_name})"

