#!/usr/bin/env python3
"""
文档自动化系统快速设置脚本
一键配置完整的 RAG 驱动的文档生成流水线
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path


def create_directory_structure():
    """创建目录结构"""
    directories = [
        'doc_automation/core',
        'doc_automation/rag',
        'doc_automation/data',
        'doc_automation/templates',
        'doc_automation/output',
        'doc_automation/config',
        'doc_automation/tests',
        'docusaurus_site/docs',
        'docusaurus_site/static',
        'dagster_storage'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f" 创建目录: {directory}")


def create_core_modules():
    """创建核心模块"""
    
    doc_generator_code = '''"""
文档生成器核心模块
基于 RAG 和 DocSpec 规范生成各类文档
"""

import yaml
import json
from typing import Dict, Any, List
from jinja2 import Environment, FileSystemLoader
from .rag_retriever import RAGRetriever


class DocumentGenerator:
    def __init__(self, rag_retriever: RAGRetriever):
        self.rag_retriever = rag_retriever
        self.jinja_env = Environment(
            loader=FileSystemLoader('templates'),
            autoescape=True
        )
    
    def generate(self, docspec: Dict[str, Any], output_format: str) -> str:
        """根据文档规格生成文档"""
        doc_type = docspec['type']
        sections = docspec['sections']
        
        content_parts = []
        
        metadata = self._generate_metadata(docspec['metadata'])
        content_parts.append(metadata)
        
        for section in sections:
            section_content = self._generate_section(section)
            content_parts.append(section_content)
        
        full_content = '\\n\\n'.join(content_parts)
        
        if output_format == 'markdown':
            return full_content
        elif output_format == 'docx':
            return self._convert_to_docx(full_content)
        elif output_format == 'html':
            return self._convert_to_html(full_content)
        else:
            return full_content
    
    def _generate_metadata(self, metadata: Dict[str, Any]) -> str:
        """生成文档元数据"""
        title = metadata.get('title', 'Untitled Document')
        author = metadata.get('author', 'System Generated')
        
        return f"""# {title}

**作者:** {author}  
**生成时间:** {metadata.get('created_at', 'Unknown')}  
**版本:** {metadata.get('version', '1.0.0')}

---
"""
    
    def _generate_section(self, section: Dict[str, Any]) -> str:
        """生成文档章节"""
        section_id = section['id']
        section_title = section['title']
        section_type = section['type']
        
        content = f"## {section_title}\\n\\n"
        
        if section_type == 'text':
            content += self._generate_text_content(section)
        elif section_type == 'table':
            content += self._generate_table_content(section)
        elif section_type == 'chart':
            content += self._generate_chart_content(section)
        elif section_type == 'flowchart':
            content += self._generate_flowchart_content(section)
        
        return content
    
    def _generate_text_content(self, section: Dict[str, Any]) -> str:
        """生成文本内容"""
        prompts = section.get('prompts', [])
        data_sources = section.get('data_sources', [])
        
        context = ""
        for data_source in data_sources:
            relevant_data = self.rag_retriever.retrieve(data_source, limit=5)
            context += f"\\n{relevant_data}"
        
        if prompts:
            generated_content = self._call_llm_for_generation(prompts[0], context)
            return generated_content
        
        return "内容待生成..."
    
    def _generate_table_content(self, section: Dict[str, Any]) -> str:
        """生成表格内容"""
        schema = section.get('table_schema', {})
        columns = schema.get('columns', [])
        
        header = "| " + " | ".join([col['name'] for col in columns]) + " |"
        separator = "| " + " | ".join(['---'] * len(columns)) + " |"
        
        table_content = f"{header}\\n{separator}\\n"
        
        data_sources = section.get('data_sources', [])
        for data_source in data_sources:
            table_data = self.rag_retriever.get_structured_data(data_source)
            for row in table_data[:10]:  # 限制行数
                row_content = "| " + " | ".join([str(row.get(col['name'], '')) for col in columns]) + " |"
                table_content += f"{row_content}\\n"
        
        return table_content
    
    def _generate_chart_content(self, section: Dict[str, Any]) -> str:
        """生成图表内容"""
        chart_config = section.get('chart_config', {})
        chart_type = chart_config.get('type', 'bar')
        
        chart_description = f"### {chart_type.title()} 图表\\n\\n"
        chart_description += "```json\\n"
        chart_description += json.dumps(chart_config, indent=2, ensure_ascii=False)
        chart_description += "\\n```\\n\\n"
        
        return chart_description
    
    def _generate_flowchart_content(self, section: Dict[str, Any]) -> str:
        """生成流程图内容"""
        mermaid_template = section.get('mermaid_template', '')
        
        if mermaid_template:
            return f"```mermaid\\n{mermaid_template}\\n```\\n\\n"
        
        return "流程图待生成..."
    
    def _call_llm_for_generation(self, prompt: str, context: str) -> str:
        """调用 LLM 生成内容"""
        return f"基于提示 '{prompt}' 和上下文生成的内容..."
    
    def _convert_to_docx(self, markdown_content: str) -> str:
        """转换为 DOCX 格式"""
        return f"[DOCX] {markdown_content}"
    
    def _convert_to_html(self, markdown_content: str) -> str:
        """转换为 HTML 格式"""
        return f"<html><body>{markdown_content}</body></html>"
'''
    
    with open('doc_automation/core/doc_generator.py', 'w', encoding='utf-8') as f:
        f.write(doc_generator_code)
    
    rag_retriever_code = '''"""
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
        
        print(f" 已索引数据源: {source_name}")
    
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
                result_content.append(f"**{source_name}** (相似度: {score:.3f})\\n{content}")
        
        return "\\n\\n".join(result_content)
    
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
'''
    
    with open('doc_automation/core/rag_retriever.py', 'w', encoding='utf-8') as f:
        f.write(rag_retriever_code)
    
    data_fetcher_code = '''"""
数据获取器模块
从各种数据源获取审核相关数据
"""

import random
from datetime import datetime, timedelta
from typing import Dict, Any


class DataFetcher:
    def __init__(self):
        self.mock_data = True  # 生产环境中应连接真实数据源
    
    def get_audit_metrics(self) -> Dict[str, Any]:
        """获取审核性能指标"""
        if self.mock_data:
            return {
                'total_processed': random.randint(10000, 50000),
                'accuracy_rate': round(random.uniform(0.92, 0.98), 3),
                'false_positive_rate': round(random.uniform(0.01, 0.03), 3),
                'false_negative_rate': round(random.uniform(0.005, 0.015), 3),
                'avg_processing_time': f"{random.randint(15, 45)}s",
                'escalation_ratio': round(random.uniform(0.02, 0.08), 3),
                'daily_trends': [
                    {'date': (datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d'),
                     'volume': random.randint(1000, 3000),
                     'accuracy': round(random.uniform(0.90, 0.98), 3)}
                    for i in range(7)
                ]
            }
        
        return {}
    
    def get_experiment_data(self) -> Dict[str, Any]:
        """获取实验数据"""
        if self.mock_data:
            return {
                'ab_tests': [
                    {
                        'experiment_id': 'exp_001',
                        'name': 'Prompt优化实验',
                        'start_date': '2024-01-01',
                        'end_date': '2024-01-15',
                        'groups': {
                            'control': {'accuracy': 0.92, 'latency': 45},
                            'treatment': {'accuracy': 0.95, 'latency': 42}
                        },
                        'statistical_significance': 0.95
                    }
                ],
                'model_versions': [
                    {
                        'version': 'v2.1.0',
                        'release_date': '2024-01-10',
                        'improvements': ['提升准确率3%', '降低延迟10%'],
                        'performance_delta': {'accuracy': 0.03, 'latency': -0.1}
                    }
                ]
            }
        
        return {}
    
    def get_policy_configs(self) -> Dict[str, Any]:
        """获取策略配置数据"""
        if self.mock_data:
            return {
                'risk_categories': [
                    {'name': '暴力内容', 'threshold': 0.8, 'action': 'block'},
                    {'name': '色情内容', 'threshold': 0.7, 'action': 'review'},
                    {'name': '虚假信息', 'threshold': 0.6, 'action': 'flag'}
                ],
                'content_types': [
                    {'type': '短视频', 'daily_volume': 50000, 'auto_rate': 0.85},
                    {'type': '图文', 'daily_volume': 30000, 'auto_rate': 0.90},
                    {'type': '直播', 'daily_volume': 5000, 'auto_rate': 0.70}
                ],
                'escalation_rules': [
                    {'condition': 'confidence < 0.5', 'action': 'human_review'},
                    {'condition': 'risk_score > 0.9', 'action': 'expert_review'}
                ]
            }
        
        return {}
    
    def get_user_feedback(self) -> Dict[str, Any]:
        """获取用户反馈数据"""
        if self.mock_data:
            return {
                'satisfaction_scores': [
                    {'department': '内容安全', 'score': 4.2, 'responses': 156},
                    {'department': '产品运营', 'score': 3.8, 'responses': 89},
                    {'department': '算法团队', 'score': 4.5, 'responses': 67}
                ],
                'feature_requests': [
                    {'feature': '批量审核', 'votes': 23, 'priority': 'high'},
                    {'feature': '自定义规则', 'votes': 18, 'priority': 'medium'},
                    {'feature': '移动端支持', 'votes': 12, 'priority': 'low'}
                ],
                'pain_points': [
                    '审核界面响应慢',
                    '误报率偏高',
                    '培训材料不足'
                ]
            }
        
        return {}
'''
    
    with open('doc_automation/core/data_fetcher.py', 'w', encoding='utf-8') as f:
        f.write(data_fetcher_code)
    
    print(" 核心模块创建完成")


def create_dagster_config():
    """创建 Dagster 配置文件"""
    dagster_yaml = '''# Dagster 工作空间配置
load_from:
  - python_package: doc_automation.pipelines

telemetry:
  enabled: false

storage:
  filesystem:
    base_dir: dagster_storage

run_launcher:
  module: dagster._core.launcher.sync_in_memory_run_launcher
  class: SyncInMemoryRunLauncher

compute_logs:
  module: dagster._core.storage.local_compute_log_manager
  class: LocalComputeLogManager
  config:
    base_dir: dagster_storage/compute_logs
'''
    
    with open('doc_automation/workspace.yaml', 'w', encoding='utf-8') as f:
        f.write(dagster_yaml)
    
    print(" Dagster 配置创建完成")


def create_api_server():
    """创建 FastAPI 服务器"""
    api_code = '''"""
文档生成 API 服务器
提供 REST API 接口触发文档生成
"""

from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from typing import List, Dict, Any
import uvicorn
import asyncio
from dagster import execute_job_sync
from .pipelines.doc_generation_pipeline import doc_generation_job


app = FastAPI(title="文档自动化 API", version="1.0.0")


class DocGenerationRequest(BaseModel):
    doc_type: str
    output_formats: List[str] = ["markdown", "docx"]
    auto_publish: bool = True
    context: Dict[str, Any] = {}


class DocGenerationResponse(BaseModel):
    job_id: str
    status: str
    message: str


@app.post("/api/doc-automation/generate", response_model=DocGenerationResponse)
async def generate_document(
    request: DocGenerationRequest,
    background_tasks: BackgroundTasks
):
    """触发文档生成任务"""
    
    job_id = f"doc_gen_{request.doc_type}_{int(asyncio.get_event_loop().time())}"
    
    run_config = {
        "ops": {
            "generate_documents": {
                "config": {
                    "doc_type": request.doc_type,
                    "output_formats": request.output_formats,
                    "auto_publish": request.auto_publish
                }
            }
        }
    }
    
    background_tasks.add_task(run_dagster_job, run_config)
    
    return DocGenerationResponse(
        job_id=job_id,
        status="started",
        message=f"文档生成任务已启动: {request.doc_type}"
    )


async def run_dagster_job(run_config: Dict[str, Any]):
    """在后台运行 Dagster 作业"""
    try:
        result = execute_job_sync(doc_generation_job, run_config=run_config)
        print(f" Dagster 作业执行完成: {result.run_id}")
    except Exception as e:
        print(f" Dagster 作业执行失败: {e}")


@app.get("/api/doc-automation/health")
async def health_check():
    """健康检查"""
    return {"status": "ok", "message": "文档自动化服务运行正常"}


@app.get("/api/doc-automation/docs")
async def list_documents():
    """列出已生成的文档"""
    return {"documents": []}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
'''
    
    with open('doc_automation/api_server.py', 'w', encoding='utf-8') as f:
        f.write(api_code)
    
    print(" API 服务器创建完成")


def create_start_script():
    """创建启动脚本"""
    start_script = '''#!/bin/bash

echo " 启动文档自动化系统..."

echo " 检查依赖..."
pip install -r requirements.txt

echo " 初始化 Dagster..."
export DAGSTER_HOME=$(pwd)/dagster_storage
dagster instance migrate

echo " 启动 Dagster UI..."
nohup dagster dev --host 0.0.0.0 --port 3000 > dagster.log 2>&1 &

echo " 启动 API 服务器..."
nohup python -m doc_automation.api_server > api.log 2>&1 &

echo " 系统启动完成！"
echo ""
echo " Dagster UI: http://localhost:3000"
echo " API 接口: http://localhost:8000"
echo " API 文档: http://localhost:8000/docs"
echo ""
echo "日志文件:"
echo "  - Dagster: dagster.log"
echo "  - API: api.log"
echo ""
echo "停止服务: ./stop.sh"
'''
    
    with open('doc_automation/start.sh', 'w', encoding='utf-8') as f:
        f.write(start_script)
    os.chmod('doc_automation/start.sh', 0o755)
    
    stop_script = '''#!/bin/bash

echo " 停止文档自动化系统..."

pkill -f "dagster dev"
pkill -f "api_server"

echo " 系统已停止"
'''
    
    with open('doc_automation/stop.sh', 'w', encoding='utf-8') as f:
        f.write(stop_script)
    os.chmod('doc_automation/stop.sh', 0o755)
    
    print(" 启动脚本创建完成")


def main():
    """主函数"""
    print(" 开始设置文档自动化系统...")
    
    try:
        create_directory_structure()
        create_core_modules()
        create_dagster_config()
        create_api_server()
        create_start_script()
        
        print("\n 文档自动化系统设置完成！")
        print("\n下一步操作:")
        print("1. 安装依赖: pip install -r requirements.txt")
        print("2. 启动系统: ./doc_automation/start.sh")
        print("3. 访问 Dagster UI: http://localhost:3000")
        print("4. 测试 API: http://localhost:8000/docs")
        print("\n 前端集成:")
        print("- 在审核界面点击文档生成按钮")
        print("- 调用 POST /api/doc-automation/generate")
        
    except Exception as e:
        print(f" 设置失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
