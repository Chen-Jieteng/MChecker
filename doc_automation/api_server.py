"""
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
        print(f"✅ Dagster 作业执行完成: {result.run_id}")
    except Exception as e:
        print(f"❌ Dagster 作业执行失败: {e}")


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
