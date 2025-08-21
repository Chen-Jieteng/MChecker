"""
Confluence 发布器
支持将生成的文档发布到 Confluence 空间
"""

import os
import requests
import base64
from typing import Dict, Any, Optional
from confluence import Confluence


class ConfluencePublisher:
    """Confluence 文档发布器"""
    
    def __init__(self):
        self.base_url = os.getenv('CONFLUENCE_BASE_URL')
        self.username = os.getenv('CONFLUENCE_USERNAME') 
        self.api_token = os.getenv('CONFLUENCE_API_TOKEN')
        self.space_key = os.getenv('CONFLUENCE_SPACE_KEY', 'AUDIT')
        
        if not all([self.base_url, self.username, self.api_token]):
            raise ValueError("Confluence credentials not configured")
            
        self.confluence = Confluence(
            url=self.base_url,
            username=self.username,
            password=self.api_token
        )
    
    def publish(self, document_path: str, page_config: Optional[Dict[str, Any]] = None) -> bool:
        """发布文档到Confluence"""
        try:
            with open(document_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            title, body = self._parse_document(content)
            
            if page_config:
                title = page_config.get('title', title)
                parent_id = page_config.get('parent_id')
                space_key = page_config.get('space_key', self.space_key)
            else:
                parent_id = None
                space_key = self.space_key
            
            existing_page = self._find_page(title, space_key)
            
            if existing_page:
                page_id = existing_page['id']
                result = self.confluence.update_page(
                    page_id=page_id,
                    title=title,
                    body=self._convert_to_confluence_format(body)
                )
            else:
                result = self.confluence.create_page(
                    space=space_key,
                    title=title,
                    body=self._convert_to_confluence_format(body),
                    parent_id=parent_id
                )
            
            if result:
                print(f"Successfully published '{title}' to Confluence")
                return True
            else:
                print(f"Failed to publish '{title}' to Confluence")
                return False
                
        except Exception as e:
            print(f"Error publishing to Confluence: {e}")
            return False
    
    def _parse_document(self, content: str) -> tuple[str, str]:
        """解析文档，提取标题和正文"""
        lines = content.split('\n')
        
        title = "Untitled Document"
        body_start_index = 0
        
        for i, line in enumerate(lines):
            if line.startswith('# '):
                title = line[2:].strip()
                body_start_index = i + 1
                break
        
        body = '\n'.join(lines[body_start_index:])
        return title, body
    
    def _convert_to_confluence_format(self, markdown_content: str) -> str:
        """将Markdown内容转换为Confluence格式"""
        content = markdown_content
        
        content = content.replace('## ', 'h2. ')
        content = content.replace('### ', 'h3. ')
        content = content.replace('#### ', 'h4. ')
        
        lines = content.split('\n')
        converted_lines = []
        
        for line in lines:
            if line.strip().startswith('- '):
                converted_lines.append(line.replace('- ', '* '))
            elif line.strip().startswith('1. '):
                converted_lines.append(line.replace('1. ', '# '))
            else:
                converted_lines.append(line)
        
        return '\n'.join(converted_lines)
    
    def _find_page(self, title: str, space_key: str) -> Optional[Dict[str, Any]]:
        """查找现有页面"""
        try:
            pages = self.confluence.get_all_pages_from_space(
                space=space_key,
                start=0,
                limit=100
            )
            
            for page in pages:
                if page['title'] == title:
                    return page
                    
            return None
        except Exception:
            return None
    
    def create_page_tree(self, pages_config: Dict[str, Any]) -> bool:
        """创建页面树结构"""
        try:
            parent_page = self.confluence.create_page(
                space=self.space_key,
                title=pages_config['title'],
                body=pages_config.get('body', ''),
            )
            
            if parent_page and 'children' in pages_config:
                parent_id = parent_page['id']
                
                for child_config in pages_config['children']:
                    self.confluence.create_page(
                        space=self.space_key,
                        title=child_config['title'],
                        body=child_config.get('body', ''),
                        parent_id=parent_id
                    )
            
            return True
        except Exception as e:
            print(f"Error creating page tree: {e}")
            return False
    
    def add_attachment(self, page_id: str, file_path: str) -> bool:
        """添加附件到页面"""
        try:
            with open(file_path, 'rb') as f:
                result = self.confluence.attach_file(
                    filename=os.path.basename(file_path),
                    data=f.read(),
                    page_id=page_id
                )
            return bool(result)
        except Exception as e:
            print(f"Error adding attachment: {e}")
            return False
    
    def update_page_labels(self, page_id: str, labels: list[str]) -> bool:
        """更新页面标签"""
        try:
            for label in labels:
                self.confluence.set_page_label(page_id, label)
            return True
        except Exception as e:
            print(f"Error updating labels: {e}")
            return False


CONFLUENCE_CONFIG = {
    'audit_reports': {
        'space_key': 'AUDIT',
        'parent_page': '审核报告',
        'labels': ['audit', 'report', 'weekly']
    },
    'strategy_docs': {
        'space_key': 'POLICY',
        'parent_page': '策略文档',
        'labels': ['strategy', 'policy', 'guidelines']
    },
    'training_materials': {
        'space_key': 'TRAINING',
        'parent_page': '培训材料',
        'labels': ['training', 'onboarding', 'documentation']
    }
}
