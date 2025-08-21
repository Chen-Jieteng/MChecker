# 文档自动化生成系统

## 概述
基于 RAG 技术的智能文档生成系统，支持策略文档、实验报告、数据分析、培训材料和 PRD 的一键生成。

## 架构
```
doc_automation/
├── docspecs/           # 文档规格模板
├── pipelines/          # Dagster 流水线
├── publishers/         # 发布脚本
├── templates/          # 模板文件
├── data/              # 数据源配置
├── rag/               # RAG 检索增强
└── config/            # 配置文件
```

## 技术栈
- **RAG**: Milvus + bge-m3 + bge-reranker-v2
- **流水线**: Dagster + dbt
- **模型**: DashScope Qwen 系列
- **发布**: Docusaurus + Confluence API
- **图表**: ECharts + Mermaid + Plotly

## 快速开始
```bash
# 安装依赖
pip install -r requirements.txt

# 初始化配置
python setup.py init

# 启动 Dagster UI
dagster dev
```

## 支持的文档类型
1. **策略文档**: 审核规范、风险分类、判定阈值
2. **实验报告**: A/B测试、Prompt实验、性能对比
3. **数据周报**: 效能指标、趋势监控、KPI仪表板
4. **培训材料**: PPT、操作手册、沟通文档
5. **PRD文档**: 需求规格、接口设计、用户场景
