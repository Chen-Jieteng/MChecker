"""
文档生成流水线
支持策略文档、实验报告、数据周报、培训材料和PRD的自动化生成
"""

import os
import yaml
from datetime import datetime
from typing import Dict, Any, List
from dagster import asset, job, op, Config, In, Out, DynamicOut, DynamicOutput
from pydantic import BaseModel

from ..core.doc_generator import DocumentGenerator
from ..core.rag_retriever import RAGRetriever
from ..core.data_fetcher import DataFetcher
from ..publishers.confluence_publisher import ConfluencePublisher
from ..publishers.docusaurus_publisher import DocusaurusPublisher


class DocGenerationConfig(Config):
    """文档生成配置"""
    doc_type: str  # strategy_policy, experiment_report, data_weekly, training_materials, prd_document
    template_name: str = None  # 可选的自定义模板
    output_formats: List[str] = ["markdown", "docx"]
    auto_publish: bool = True
    recipients: List[str] = []


@asset(group_name="data_sources")
def audit_performance_metrics() -> Dict[str, Any]:
    """获取审核性能指标数据"""
    data_fetcher = DataFetcher()
    return data_fetcher.get_audit_metrics()


@asset(group_name="data_sources")
def experiment_results() -> Dict[str, Any]:
    """获取实验结果数据"""
    data_fetcher = DataFetcher()
    return data_fetcher.get_experiment_data()


@asset(group_name="data_sources")
def policy_configurations() -> Dict[str, Any]:
    """获取策略配置数据"""
    data_fetcher = DataFetcher()
    return data_fetcher.get_policy_configs()


@asset(group_name="data_sources")
def user_feedback_data() -> Dict[str, Any]:
    """获取用户反馈数据"""
    data_fetcher = DataFetcher()
    return data_fetcher.get_user_feedback()


@asset(group_name="knowledge_base")
def rag_knowledge_base(
    audit_performance_metrics: Dict[str, Any],
    experiment_results: Dict[str, Any],
    policy_configurations: Dict[str, Any],
    user_feedback_data: Dict[str, Any]
) -> RAGRetriever:
    """构建RAG知识库"""
    rag_retriever = RAGRetriever()
    
    # 索引各类数据源
    rag_retriever.index_data("audit_metrics", audit_performance_metrics)
    rag_retriever.index_data("experiments", experiment_results)
    rag_retriever.index_data("policies", policy_configurations)
    rag_retriever.index_data("feedback", user_feedback_data)
    
    return rag_retriever


@op(
    config_schema=DocGenerationConfig,
    out=DynamicOut(str)
)
def generate_documents(context, rag_knowledge_base: RAGRetriever):
    """生成文档的动态操作"""
    config = context.op_config
    doc_type = config["doc_type"]
    
    # 加载文档规格
    docspec_path = f"docspecs/{doc_type}.yaml"
    with open(docspec_path, 'r', encoding='utf-8') as f:
        docspec = yaml.safe_load(f)
    
    # 生成文档
    generator = DocumentGenerator(rag_knowledge_base)
    
    # 根据配置的输出格式生成多种格式
    for output_format in config["output_formats"]:
        document = generator.generate(docspec, output_format)
        
        # 保存文档
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{doc_type}_{timestamp}.{output_format}"
        output_path = f"output/{filename}"
        
        os.makedirs("output", exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(document)
        
        context.log.info(f"Generated {output_format} document: {output_path}")
        yield DynamicOutput(output_path, mapping_key=output_format)


@op(
    ins={"document_path": In(str)},
    out=Out(bool)
)
def publish_to_confluence(context, document_path: str) -> bool:
    """发布到Confluence"""
    try:
        publisher = ConfluencePublisher()
        result = publisher.publish(document_path)
        context.log.info(f"Published to Confluence: {document_path}")
        return result
    except Exception as e:
        context.log.error(f"Failed to publish to Confluence: {e}")
        return False


@op(
    ins={"document_path": In(str)},
    out=Out(bool)
)
def publish_to_docusaurus(context, document_path: str) -> bool:
    """发布到Docusaurus"""
    try:
        publisher = DocusaurusPublisher()
        result = publisher.publish(document_path)
        context.log.info(f"Published to Docusaurus: {document_path}")
        return result
    except Exception as e:
        context.log.error(f"Failed to publish to Docusaurus: {e}")
        return False


@op(
    ins={"document_path": In(str)},
    out=Out(bool)
)
def send_notification(context, document_path: str) -> bool:
    """发送通知"""
    try:
        # 这里可以集成邮件、钉钉、企微等通知渠道
        context.log.info(f"Notification sent for document: {document_path}")
        return True
    except Exception as e:
        context.log.error(f"Failed to send notification: {e}")
        return False


@job
def doc_generation_job():
    """文档生成作业流水线"""
    # 构建知识库
    knowledge_base = rag_knowledge_base(
        audit_performance_metrics(),
        experiment_results(),
        policy_configurations(),
        user_feedback_data()
    )
    
    # 动态生成文档
    documents = generate_documents(knowledge_base)
    
    # 发布文档
    confluence_results = documents.map(publish_to_confluence)
    docusaurus_results = documents.map(publish_to_docusaurus)
    
    # 发送通知
    notifications = documents.map(send_notification)


# 资产组定义
doc_generation_assets = [
    audit_performance_metrics,
    experiment_results,
    policy_configurations,
    user_feedback_data,
    rag_knowledge_base
]


# 预定义的文档生成作业
@job(
    config={
        "ops": {
            "generate_documents": {
                "config": {
                    "doc_type": "data_weekly",
                    "output_formats": ["html", "pdf"],
                    "auto_publish": True
                }
            }
        }
    }
)
def weekly_data_report_job():
    """每周数据报告自动生成"""
    doc_generation_job()


@job(
    config={
        "ops": {
            "generate_documents": {
                "config": {
                    "doc_type": "experiment_report",
                    "output_formats": ["markdown", "pptx"],
                    "auto_publish": False
                }
            }
        }
    }
)
def experiment_report_job():
    """实验报告生成"""
    doc_generation_job()


@job(
    config={
        "ops": {
            "generate_documents": {
                "config": {
                    "doc_type": "strategy_policy",
                    "output_formats": ["docx", "confluence"],
                    "auto_publish": True
                }
            }
        }
    }
)
def strategy_policy_job():
    """策略文档生成"""
    doc_generation_job()
