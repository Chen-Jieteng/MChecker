"""
RAG 检索器模块
支持多种高级检索策略
"""

from typing import Dict, Any, List
import numpy as np
from sentence_transformers import SentenceTransformer


class RAGRetriever:
    def __init__(self):
        self.embedding_model = SentenceTransformer('BAAI/bge-m3')
        self.knowledge_base = {}
        self.embeddings = {}
    
    def index_data(self, source_name: str, data: Dict[str, Any]):
        """索引数据到知识库"""
        self.knowledge_base[source_name] = data
        
        if isinstance(data, dict):
            text_content = self._extract_text_from_dict(data)
        else:
            text_content = str(data)
        
        embedding = self.embedding_model.encode(text_content)
        self.embeddings[source_name] = embedding
        
        print(f"✅ 已索引数据源: {source_name}")
    
    def retrieve(self, query: str, limit: int = 5) -> str:
        """检索相关信息"""
        query_embedding = self.embedding_model.encode(query)
        
        similarities = {}
        for source_name, embedding in self.embeddings.items():
            similarity = np.dot(query_embedding, embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(embedding)
            )
            similarities[source_name] = similarity
        
        sorted_sources = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        
        result_content = []
        for source_name, score in sorted_sources[:limit]:
            if score > 0.5:  # 相似度阈值
                content = self.knowledge_base[source_name]
                result_content.append(f"**{source_name}** (相似度: {score:.3f})\n{content}")
        
        return "\n\n".join(result_content)
    
    def get_structured_data(self, source_name: str) -> List[Dict[str, Any]]:
        """获取结构化数据"""
        data = self.knowledge_base.get(source_name, {})
        
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            return [data]
        else:
            return []
    
    def _extract_text_from_dict(self, data: Dict[str, Any]) -> str:
        """从字典中提取文本内容"""
        text_parts = []
        
        def extract_recursive(obj, prefix=""):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    extract_recursive(value, f"{prefix}{key}: ")
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    extract_recursive(item, f"{prefix}[{i}] ")
            else:
                text_parts.append(f"{prefix}{str(obj)}")
        
        extract_recursive(data)
        return " ".join(text_parts)
