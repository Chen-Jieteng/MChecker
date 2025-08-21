#!/usr/bin/env python3
"""
简化版文档生成API服务器
不依赖Dagster，直接提供文档生成功能
"""

import os
import json
import time
import uuid
import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from professional_document_generator import ProfessionalDocumentGenerator

app = FastAPI(
    title="文档自动化 API",
    description="智能文档生成系统API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

task_storage: Dict[str, Dict] = {}

generation_contexts: Dict[str, Dict] = {}

doc_generator = ProfessionalDocumentGenerator()

class DocGenerationRequest(BaseModel):
    doc_type: str
    output_formats: List[str] = ["markdown", "docx"]
    auto_publish: bool = True
    context: Dict[str, Any] = {}

class DocGenerationResponse(BaseModel):
    job_id: str
    status: str
    message: str

class TaskStatus(BaseModel):
    job_id: str
    status: str  # started, processing, completed, failed
    progress: int = 0
    message: str = ""
    documents: List[Dict[str, Any]] = []
    created_at: str
    updated_at: str

@app.post("/api/doc-automation/generate", response_model=DocGenerationResponse)
async def generate_document(
    request: DocGenerationRequest,
    background_tasks: BackgroundTasks
):
    """触发文档生成任务"""
    
    job_id = f"doc_{request.doc_type}_{uuid.uuid4().hex[:8]}"
    
    task_storage[job_id] = {
        "job_id": job_id,
        "status": "started",
        "progress": 0,
        "message": f"开始生成{request.doc_type}文档",
        "documents": [],
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
        "request": request.dict()
    }
    
    background_tasks.add_task(process_document_generation, job_id, request)
    
    return DocGenerationResponse(
        job_id=job_id,
        status="started",
        message=f"文档生成任务已启动: {request.doc_type}"
    )

@app.get("/api/doc-automation/status/{job_id}")
async def get_generation_status(job_id: str):
    """查询文档生成状态"""
    
    if job_id not in task_storage:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    task = task_storage[job_id]
    return TaskStatus(**task)

@app.get("/api/doc-automation/download/{document_id}")
async def download_document(document_id: str, format: str = "markdown"):
    """下载生成的文档"""
    
    job_id = None
    doc_type = "strategy"  # 默认类型
    
    if "_" in document_id:
        parts = document_id.split("_")
        if len(parts) >= 3:
            job_id = "_".join(parts[:-1])  # 移除最后的format部分
            
            if len(parts) >= 4 and parts[1] in ["prompt", "performance", "ab"]:
                doc_type = f"{parts[1]}_{parts[2]}"
            else:
                doc_type = parts[1]  # 例如: doc_strategy_abc123 -> strategy
    
    context = {}
    if job_id and job_id in task_storage:
        stored_request = task_storage[job_id].get("request", {})
        context = stored_request.get("context", {})
        print(f"📋 找到存储的请求上下文: job_id={job_id}, doc_type={doc_type}")
        
        if "novel_config" in context:
            novel_config = context["novel_config"]
            print(f"📖 小说配置: mode={novel_config.get('mode')}, target_length={novel_config.get('target_length')}, style={novel_config.get('style')}")
    else:
        print(f"⚠️  未找到存储的上下文，使用默认配置: job_id={job_id}")
        context = {
            "doc_type": doc_type,
            "user_context": {
                "reviewer": "系统",
                "department": "内容安全团队",
                "timestamp": datetime.now().isoformat()
            }
        }
    
    doc_content = generate_real_document(document_id, format, doc_type, context)
    
    from fastapi.responses import Response
    
    content_type_map = {
        "markdown": "text/markdown",
        "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "html": "text/html",
        "pdf": "application/pdf"
    }
    
    return Response(
        content=doc_content,
        media_type=content_type_map.get(format, "text/plain"),
        headers={
            "Content-Disposition": f"attachment; filename=document_{document_id}.{format}"
        }
    )

@app.get("/api/doc-automation/health")
async def health_check():
    """健康检查"""
    return {
        "status": "ok",
        "message": "文档自动化服务运行正常",
        "timestamp": datetime.now().isoformat(),
        "active_tasks": len(task_storage)
    }

@app.get("/api/doc-automation/docs")
async def list_documents(limit: int = 10):
    """列出已生成的文档"""
    
    recent_tasks = list(task_storage.values())[-limit:]
    
    documents = []
    for task in recent_tasks:
        if task["status"] == "completed" and task["documents"]:
            documents.extend(task["documents"])
    
    return {"documents": documents}

async def process_document_generation(job_id: str, request: DocGenerationRequest):
    """处理文档生成任务"""
    
    try:
        update_task_status(job_id, "processing", 10, "开始数据收集...")
        await asyncio.sleep(2)
        
        update_task_status(job_id, "processing", 30, "收集审核数据...")
        await asyncio.sleep(3)
        
        update_task_status(job_id, "processing", 50, "检索相关知识...")
        await asyncio.sleep(2)
        
        update_task_status(job_id, "processing", 70, "AI内容生成中...")
        await asyncio.sleep(4)
        
        update_task_status(job_id, "processing", 90, "格式转换中...")
        await asyncio.sleep(2)
        
        documents = []
        for format_type in request.output_formats:
            doc_id = f"{job_id}_{format_type}"
            document = {
                "id": doc_id,
                "format": format_type,
                "filename": f"{get_doc_type_name(request.doc_type)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{get_file_extension(format_type)}",
                "size": get_estimated_size(format_type),
                "created_at": datetime.now().isoformat(),
                "download_url": f"/api/doc-automation/download/{doc_id}?format={format_type}"
            }
            documents.append(document)
        
        task_storage[job_id]["documents"] = documents
        update_task_status(job_id, "completed", 100, f"文档生成完成！共生成{len(documents)}个文件")
        
    except Exception as e:
        update_task_status(job_id, "failed", 0, f"生成失败: {str(e)}")
        print(f"❌ 文档生成失败 {job_id}: {e}")

def update_task_status(job_id: str, status: str, progress: int, message: str):
    """更新任务状态"""
    if job_id in task_storage:
        task_storage[job_id].update({
            "status": status,
            "progress": progress,
            "message": message,
            "updated_at": datetime.now().isoformat()
        })

def get_doc_type_name(doc_type: str) -> str:
    """获取文档类型中文名"""
    names = {
        "strategy": "策略文档",
        "ab_test": "A/B测试报告",
        "prompt_experiment": "Prompt实验报告", 
        "performance_test": "性能对比报告",
        "novel": "小说生成",
        "experiment": "实验报告",  # 兼容旧版本
        "data": "数据周报",
        "training": "培训材料",
        "prd": "PRD文档"
    }
    return names.get(doc_type, doc_type)

def get_file_extension(format_type: str) -> str:
    """获取文件扩展名"""
    extensions = {
        "markdown": "md",
        "docx": "docx",
        "html": "html",
        "pdf": "pdf",
        "pptx": "pptx"
    }
    return extensions.get(format_type, "txt")

def get_estimated_size(format_type: str) -> int:
    """获取预估文件大小（字节）"""
    sizes = {
        "markdown": 15000,   # ~15KB
        "docx": 45000,       # ~45KB
        "html": 25000,       # ~25KB
        "pdf": 85000,        # ~85KB
        "pptx": 120000       # ~120KB
    }
    return sizes.get(format_type, 20000)

def generate_real_document(document_id: str, format_type: str, doc_type: str, context: Dict) -> bytes:
    """生成真实文档内容"""
    
    try:
        if format_type == "docx":
            try:
                return doc_generator.generate_document_docx(doc_type, context)
            except Exception as e:
                print(f"⚠️ DOCX生成失败，回退为Markdown: {e}")
                real_content_fallback = doc_generator.generate_document(doc_type, context)
                return real_content_fallback.encode('utf-8')
        
        real_content = doc_generator.generate_document(doc_type, context)
        
        if format_type == "html":
            html_content = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>智能文档生成</title>
    <style>
        @font-face {{
            font-family: 'SimSun';
            src: local('SimSun'), local('宋体');
        }}
        body {{ 
            font-family: 'SimSun', '宋体', 'NSimSun', 'FangSong', '仿宋', 'STSong', 'Times New Roman', serif !important; 
            line-height: 1.6; 
            margin: 0;
            padding: 40px;
            background: #f8f9fa;
        }}
        * {{
            font-family: 'SimSun', '宋体', 'NSimSun', 'FangSong', '仿宋', 'STSong', 'Times New Roman', serif !important;
        }}
        .container {{ 
            max-width: 1000px; 
            margin: 0 auto; 
            background: white; 
            padding: 40px; 
            border-radius: 8px; 
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{ color: #007acc; border-bottom: 3px solid #007acc; padding-bottom: 10px; }}
        h2 {{ color: #333; border-bottom: 1px solid #eee; padding-bottom: 5px; }}
        h3 {{ color: #555; }}
        .highlight {{ 
            background: #f0f8ff; 
            padding: 15px; 
            border-left: 4px solid #007acc; 
            margin: 20px 0;
        }}
        code {{ 
            background: #f5f5f5; 
            padding: 2px 4px; 
            border-radius: 3px; 
        }}
        pre {{ 
            background: #f5f5f5; 
            padding: 15px; 
            border-radius: 5px; 
            overflow-x: auto; 
        }}
        table {{ 
            width: 100%; 
            border-collapse: collapse; 
            margin: 20px 0; 
        }}
        th, td {{ 
            border: 1px solid #ddd; 
            padding: 12px; 
            text-align: left; 
        }}
        th {{ background: #f5f5f5; }}
    </style>
</head>
<body>
    <div class="container">
        <div style="white-space: pre-wrap;">{real_content}</div>
    </div>
</body>
</html>"""
            return html_content.encode('utf-8')
        else:
            return real_content.encode('utf-8')
            
    except Exception as e:
        error_content = f"""# 文档生成错误

**文档ID**: {document_id}  
**格式**: {format_type}  
**错误时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  

{str(e)}

请联系技术支持或稍后重试。

---
*错误报告由系统自动生成*"""
        return error_content.encode('utf-8')

if __name__ == "__main__":
    print("🚀 启动文档自动化API服务器...")
    print("📍 服务地址: http://localhost:8000")
    print("📚 API文档: http://localhost:8000/docs")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
