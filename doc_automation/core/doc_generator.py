"""
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
        
        # 构建文档内容
        content_parts = []
        
        # 生成元数据头部
        metadata = self._generate_metadata(docspec['metadata'])
        content_parts.append(metadata)
        
        # 生成各个章节
        for section in sections:
            section_content = self._generate_section(section)
            content_parts.append(section_content)
        
        # 合并内容
        full_content = '\n\n'.join(content_parts)
        
        # 根据输出格式转换
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
        
        content = f"## {section_title}\n\n"
        
        if section_type == 'text':
            # 生成文本内容
            content += self._generate_text_content(section)
        elif section_type == 'table':
            # 生成表格
            content += self._generate_table_content(section)
        elif section_type == 'chart':
            # 生成图表
            content += self._generate_chart_content(section)
        elif section_type == 'flowchart':
            # 生成流程图
            content += self._generate_flowchart_content(section)
        
        return content
    
    def _generate_text_content(self, section: Dict[str, Any]) -> str:
        """生成文本内容"""
        prompts = section.get('prompts', [])
        data_sources = section.get('data_sources', [])
        
        # 使用 RAG 检索相关信息
        context = ""
        for data_source in data_sources:
            relevant_data = self.rag_retriever.retrieve(data_source, limit=5)
            context += f"\n{relevant_data}"
        
        # 基于上下文生成内容
        if prompts:
            # 这里应该调用 LLM 生成内容
            generated_content = self._call_llm_for_generation(prompts[0], context)
            return generated_content
        
        return "内容待生成..."
    
    def _generate_table_content(self, section: Dict[str, Any]) -> str:
        """生成表格内容"""
        schema = section.get('table_schema', {})
        columns = schema.get('columns', [])
        
        # 构建 Markdown 表格
        header = "| " + " | ".join([col['name'] for col in columns]) + " |"
        separator = "| " + " | ".join(['---'] * len(columns)) + " |"
        
        table_content = f"{header}\n{separator}\n"
        
        # 获取数据并填充表格
        data_sources = section.get('data_sources', [])
        for data_source in data_sources:
            table_data = self.rag_retriever.get_structured_data(data_source)
            for row in table_data[:10]:  # 限制行数
                row_content = "| " + " | ".join([str(row.get(col['name'], '')) for col in columns]) + " |"
                table_content += f"{row_content}\n"
        
        return table_content
    
    def _generate_chart_content(self, section: Dict[str, Any]) -> str:
        """生成图表内容"""
        chart_config = section.get('chart_config', {})
        chart_type = chart_config.get('type', 'bar')
        
        # 生成图表描述和配置
        chart_description = f"### {chart_type.title()} 图表\n\n"
        chart_description += "```json\n"
        chart_description += json.dumps(chart_config, indent=2, ensure_ascii=False)
        chart_description += "\n```\n\n"
        
        return chart_description
    
    def _generate_flowchart_content(self, section: Dict[str, Any]) -> str:
        """生成流程图内容"""
        mermaid_template = section.get('mermaid_template', '')
        
        if mermaid_template:
            return f"```mermaid\n{mermaid_template}\n```\n\n"
        
        return "流程图待生成..."
    
    def _call_llm_for_generation(self, prompt: str, context: str) -> str:
        """调用 LLM 生成内容"""
        # 这里应该集成实际的 LLM API 调用
        # 比如 DashScope、OpenAI 等
        return f"基于提示 '{prompt}' 和上下文生成的内容..."
    
    def _convert_to_docx(self, markdown_content: str) -> str:
        """转换为 DOCX 格式"""
        # 使用 python-docx 或 pandoc 转换
        return f"[DOCX] {markdown_content}"
    
    def _convert_to_html(self, markdown_content: str) -> str:
        """转换为 HTML 格式"""
        # 使用 markdown 库转换
        return f"<html><body>{markdown_content}</body></html>"
