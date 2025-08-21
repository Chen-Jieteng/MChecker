"""
Docusaurus 发布器
支持将生成的文档发布到 Docusaurus 站点
"""

import os
import shutil
import yaml
import json
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path


class DocusaurusPublisher:
    """Docusaurus 文档发布器"""
    
    def __init__(self, site_path: Optional[str] = None):
        self.site_path = site_path or os.getenv('DOCUSAURUS_SITE_PATH', './docusaurus_site')
        self.docs_path = os.path.join(self.site_path, 'docs')
        self.static_path = os.path.join(self.site_path, 'static')
        
        os.makedirs(self.docs_path, exist_ok=True)
        os.makedirs(self.static_path, exist_ok=True)
    
    def publish(self, document_path: str, config: Optional[Dict[str, Any]] = None) -> bool:
        """发布文档到Docusaurus"""
        try:
            with open(document_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            frontmatter, body = self._parse_frontmatter(content)
            
            if config:
                frontmatter.update(config.get('frontmatter', {}))
                category = config.get('category')
                filename = config.get('filename')
            else:
                category = frontmatter.get('category', 'general')
                filename = None
            
            if not filename:
                title = frontmatter.get('title', 'untitled')
                timestamp = datetime.now().strftime('%Y-%m-%d')
                filename = f"{timestamp}-{self._slugify(title)}.md"
            
            if category:
                target_dir = os.path.join(self.docs_path, category)
                os.makedirs(target_dir, exist_ok=True)
                target_path = os.path.join(target_dir, filename)
            else:
                target_path = os.path.join(self.docs_path, filename)
            
            full_content = self._generate_content_with_frontmatter(frontmatter, body)
            
            with open(target_path, 'w', encoding='utf-8') as f:
                f.write(full_content)
            
            print(f"Successfully published to Docusaurus: {target_path}")
            
            self._update_sidebar_config(category, filename, frontmatter)
            
            return True
            
        except Exception as e:
            print(f"Error publishing to Docusaurus: {e}")
            return False
    
    def _parse_frontmatter(self, content: str) -> tuple[Dict[str, Any], str]:
        """解析frontmatter和正文"""
        if content.startswith('---\n'):
            parts = content.split('---\n', 2)
            if len(parts) >= 3:
                try:
                    frontmatter = yaml.safe_load(parts[1])
                    body = parts[2]
                    return frontmatter or {}, body
                except yaml.YAMLError:
                    pass
        
        lines = content.split('\n')
        title = "Untitled"
        body = content
        
        for line in lines:
            if line.startswith('# '):
                title = line[2:].strip()
                break
        
        frontmatter = {
            'title': title,
            'date': datetime.now().strftime('%Y-%m-%d'),
            'authors': ['system']
        }
        
        return frontmatter, body
    
    def _generate_content_with_frontmatter(self, frontmatter: Dict[str, Any], body: str) -> str:
        """生成带frontmatter的完整内容"""
        if 'id' not in frontmatter:
            frontmatter['id'] = self._slugify(frontmatter.get('title', 'untitled'))
        
        if 'date' not in frontmatter:
            frontmatter['date'] = datetime.now().strftime('%Y-%m-%d')
        
        yaml_content = yaml.dump(frontmatter, default_flow_style=False, allow_unicode=True)
        
        return f"---\n{yaml_content}---\n\n{body}"
    
    def _slugify(self, text: str) -> str:
        """将文本转换为URL友好的slug"""
        import re
        text = text.lower()
        text = re.sub(r'[^\w\s-]', '', text)
        text = re.sub(r'[-\s]+', '-', text)
        return text.strip('-')
    
    def _update_sidebar_config(self, category: str, filename: str, frontmatter: Dict[str, Any]):
        """更新侧边栏配置"""
        try:
            sidebar_path = os.path.join(self.site_path, 'sidebars.js')
            
            if not os.path.exists(sidebar_path):
                self._create_initial_sidebar_config(sidebar_path)
            
            print(f"Sidebar updated for category: {category}")
            
        except Exception as e:
            print(f"Warning: Failed to update sidebar config: {e}")
    
    def _create_initial_sidebar_config(self, sidebar_path: str):
        """创建初始侧边栏配置"""
        config_content = """
module.exports = {
  docs: [
    'intro',
    {
      type: 'category',
      label: '策略文档',
      items: [],
    },
    {
      type: 'category',
      label: '实验报告',
      items: [],
    },
    {
      type: 'category',
      label: '数据分析',
      items: [],
    },
    {
      type: 'category',
      label: '培训材料',
      items: [],
    },
    {
      type: 'category',
      label: 'PRD文档',
      items: [],
    },
  ],
};
"""
        with open(sidebar_path, 'w', encoding='utf-8') as f:
            f.write(config_content)
    
    def publish_assets(self, assets_dir: str, target_subdir: str = 'generated') -> bool:
        """发布静态资源（图片、图表等）"""
        try:
            target_dir = os.path.join(self.static_path, target_subdir)
            
            if os.path.exists(assets_dir):
                if os.path.exists(target_dir):
                    shutil.rmtree(target_dir)
                shutil.copytree(assets_dir, target_dir)
                print(f"Assets published to: {target_dir}")
                return True
            else:
                print(f"Assets directory not found: {assets_dir}")
                return False
                
        except Exception as e:
            print(f"Error publishing assets: {e}")
            return False
    
    def create_category_index(self, category: str, config: Dict[str, Any]) -> bool:
        """为分类创建索引页面"""
        try:
            category_dir = os.path.join(self.docs_path, category)
            os.makedirs(category_dir, exist_ok=True)
            
            index_path = os.path.join(category_dir, '_category_.json')
            
            category_config = {
                'label': config.get('label', category.title()),
                'position': config.get('position', 1),
                'collapsible': config.get('collapsible', True),
                'collapsed': config.get('collapsed', False),
            }
            
            with open(index_path, 'w', encoding='utf-8') as f:
                json.dump(category_config, f, indent=2, ensure_ascii=False)
            
            intro_content = f"""# {category_config['label']}

{config.get('description', f'{category_config["label"]}相关文档。')}


此分类包含以下类型的文档：

{config.get('content_types', '- 相关文档')}
"""
            
            intro_path = os.path.join(category_dir, 'intro.md')
            with open(intro_path, 'w', encoding='utf-8') as f:
                f.write(intro_content)
            
            return True
            
        except Exception as e:
            print(f"Error creating category index: {e}")
            return False
    
    def build_site(self, build_command: Optional[str] = None) -> bool:
        """构建Docusaurus站点"""
        try:
            import subprocess
            
            if not build_command:
                build_command = "npm run build"
            
            result = subprocess.run(
                build_command.split(),
                cwd=self.site_path,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                print("Docusaurus site built successfully")
                return True
            else:
                print(f"Build failed: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"Error building site: {e}")
            return False
    
    def deploy_site(self, deploy_command: Optional[str] = None) -> bool:
        """部署Docusaurus站点"""
        try:
            import subprocess
            
            if not deploy_command:
                deploy_command = "npm run deploy"
            
            result = subprocess.run(
                deploy_command.split(),
                cwd=self.site_path,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                print("Docusaurus site deployed successfully")
                return True
            else:
                print(f"Deploy failed: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"Error deploying site: {e}")
            return False


DOCUSAURUS_CONFIG = {
    'categories': {
        'strategy': {
            'label': '策略文档',
            'position': 1,
            'description': '内容审核策略和规范文档',
            'content_types': '- 审核策略\n- 风险分类\n- 处置措施'
        },
        'experiments': {
            'label': '实验报告',
            'position': 2,
            'description': 'A/B测试和模型实验报告',
            'content_types': '- A/B测试报告\n- Prompt实验\n- 性能分析'
        },
        'data-analysis': {
            'label': '数据分析',
            'position': 3,
            'description': '数据分析和监控报告',
            'content_types': '- 周报月报\n- 效能分析\n- 趋势预测'
        },
        'training': {
            'label': '培训材料',
            'position': 4,
            'description': '培训文档和操作手册',
            'content_types': '- 操作手册\n- 培训PPT\n- 最佳实践'
        },
        'prd': {
            'label': 'PRD文档',
            'position': 5,
            'description': '产品需求文档',
            'content_types': '- 功能需求\n- 技术规范\n- 用户故事'
        }
    }
}
