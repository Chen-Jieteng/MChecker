#!/bin/bash

echo "🛑 停止文档自动化系统..."

# 停止所有相关进程
pkill -f "dagster dev"
pkill -f "api_server"

echo "✅ 系统已停止"
