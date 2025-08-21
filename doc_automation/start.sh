#!/bin/bash

echo "🚀 启动文档自动化系统..."

# 检查 Python 依赖
echo "📦 检查依赖..."
pip install -r requirements.txt

# 初始化 Dagster
echo "⚙️ 初始化 Dagster..."
export DAGSTER_HOME=$(pwd)/dagster_storage
dagster instance migrate

# 启动 Dagster UI (后台)
echo "🎯 启动 Dagster UI..."
nohup dagster dev --host 0.0.0.0 --port 3000 > dagster.log 2>&1 &

# 启动 API 服务器 (后台)
echo "🌐 启动 API 服务器..."
nohup python -m doc_automation.api_server > api.log 2>&1 &

echo "✅ 系统启动完成！"
echo ""
echo "📊 Dagster UI: http://localhost:3000"
echo "🔌 API 接口: http://localhost:8000"
echo "📖 API 文档: http://localhost:8000/docs"
echo ""
echo "日志文件:"
echo "  - Dagster: dagster.log"
echo "  - API: api.log"
echo ""
echo "停止服务: ./stop.sh"
