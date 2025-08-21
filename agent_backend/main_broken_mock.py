#!/usr/bin/env python3
"""
事件驱动的智能体后端 - 最终修复版本
实现状态机和WebSocket事件流，包含兼容性API端点
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import uuid, json, time, asyncio
import logging
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime

# 设置日志
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
logger = logging.getLogger("event_driven_agent")

app = FastAPI(title="Event-Driven Agent Backend")

# CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========== 事件驱动状态机 ==========

class TaskType(Enum):
    VIDEO_ANALYSIS = "video_analysis"
    AUDIO_ANALYSIS = "audio_analysis" 
    COMMENT_ANALYSIS = "comment_analysis"
    FORM_FILLING = "form_filling"

class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"

@dataclass
class AnalysisTask:
    id: str
    type: TaskType
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[Dict[Any, Any]] = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    timeout_seconds: int = 60

@dataclass 
class WorkflowSession:
    session_id: str
    aweme_id: Optional[str] = None
    tasks: Dict[str, AnalysisTask] = field(default_factory=dict)
    websocket: Optional[WebSocket] = None
    created_at: datetime = field(default_factory=datetime.now)

# 全局状态存储
workflow_sessions: Dict[str, WorkflowSession] = {}

# ========== 事件发送器 ==========

async def emit_event(session_id: str, event_type: str, data: Dict[Any, Any]):
    """向前端发送事件"""
    session = workflow_sessions.get(session_id)
    if session and session.websocket:
        try:
            event = {
                'type': event_type,
                'session_id': session_id,
                'timestamp': int(time.time() * 1000),
                'data': data
            }
            await session.websocket.send_text(json.dumps(event, ensure_ascii=False))
            logger.info(f"📡 Event sent: {event_type} to session {session_id}")
        except Exception as e:
            logger.error(f"❌ Failed to send event {event_type}: {e}")

# ========== 任务管理器 ==========

class TaskManager:
    @staticmethod
    async def start_task(session_id: str, task_type: TaskType, **kwargs) -> str:
        """启动一个分析任务"""
        task_id = str(uuid.uuid4())
        task = AnalysisTask(
            id=task_id,
            type=task_type,
            status=TaskStatus.RUNNING,
            started_at=datetime.now()
        )
        
        session = workflow_sessions.get(session_id)
        if session:
            session.tasks[task_id] = task
            
        # 发送任务开始事件
        await emit_event(session_id, 'task_started', {
            'task_id': task_id,
            'task_type': task_type.value,
            'status': task.status.value
        })
        
        # 异步执行任务
        asyncio.create_task(TaskManager._execute_task(session_id, task_id, **kwargs))
        return task_id
    
    @staticmethod
    async def _execute_task(session_id: str, task_id: str, **kwargs):
        """执行具体的分析任务"""
        session = workflow_sessions.get(session_id)
        if not session:
            return
            
        task = session.tasks.get(task_id)
        if not task:
            return
            
        try:
            # 根据任务类型执行不同的分析
            if task.type == TaskType.VIDEO_ANALYSIS:
                result = await TaskManager._execute_video_analysis(session_id, **kwargs)
            elif task.type == TaskType.AUDIO_ANALYSIS:
                result = await TaskManager._execute_audio_analysis(session_id, **kwargs)
            elif task.type == TaskType.COMMENT_ANALYSIS:
                result = await TaskManager._execute_comment_analysis(session_id, **kwargs)
            else:
                raise ValueError(f"Unknown task type: {task.type}")
            
            # 任务完成
            task.status = TaskStatus.COMPLETED
            task.result = result
            task.completed_at = datetime.now()
            
            # 发送完成事件
            await emit_event(session_id, 'task_completed', {
                'task_id': task_id,
                'task_type': task.type.value,
                'status': task.status.value,
                'result': result
            })
            
        except Exception as e:
            # 任务失败
            task.status = TaskStatus.FAILED
            task.error = str(e)
            task.completed_at = datetime.now()
            
            logger.error(f"❌ Task {task_id} failed: {e}")
            await emit_event(session_id, 'task_failed', {
                'task_id': task_id,
                'task_type': task.type.value,
                'status': task.status.value,
                'error': str(e)
            })
    
    @staticmethod
    async def _execute_video_analysis(session_id: str, **kwargs) -> Dict[Any, Any]:
        """执行视频分析"""
        aweme_id = kwargs.get('aweme_id')
        
        # 模拟视频分析过程
        await emit_event(session_id, 'step_ready', {
            'step': 'video_extraction',
            'message': '正在提取视频帧...'
        })
        await asyncio.sleep(2)  # 模拟提取时间
        
        await emit_event(session_id, 'step_ready', {
            'step': 'video_ai_analysis', 
            'message': '正在进行AI视觉分析...'
        })
        await asyncio.sleep(3)  # 模拟AI分析时间
        
        # 返回模拟结果 - 包含帧数据
        frames = []
        for i in range(1, 13):
            frames.append({
                'index': i,
                'timestamp': i * 2.5,
                'url': f'/static/frames/frame_{i:03d}.jpg',
                'risk_level': 'low' if i % 4 != 0 else 'medium',
                'analysis': {
                    'objects': ['person', 'background', 'text'] if i % 3 == 0 else ['person', 'background'],
                    'confidence': 0.85 + (i % 3) * 0.05,
                    'description': f'第{i}帧：检测到人物活动'
                }
            })
        
        return {
            'video_id': aweme_id,
            'frames_extracted': 12,
            'frames': frames,  # 🆕 添加帧数据数组
            'analysis_result': {
                'category': '娱乐',
                'tags': ['音乐', '舞蹈', '生活'],
                'risk_level': 'low',
                'confidence': 0.85,
                'summary': '视频内容健康，包含娱乐性舞蹈和音乐元素'
            }
        }
    
    @staticmethod
    async def _execute_audio_analysis(session_id: str, **kwargs) -> Dict[Any, Any]:
        """执行音频分析"""
        await emit_event(session_id, 'step_ready', {
            'step': 'audio_extraction',
            'message': '正在提取音频...'
        })
        await asyncio.sleep(1)
        
        await emit_event(session_id, 'step_ready', {
            'step': 'speech_recognition',
            'message': '正在进行语音识别...'
        })
        await asyncio.sleep(2)
        
        return {
            'transcript': '这是一段很有趣的视频内容',
            'keywords': ['有趣', '内容', '视频'],
            'sentiment': 'positive'
        }
    
    @staticmethod
    async def _execute_comment_analysis(session_id: str, **kwargs) -> Dict[Any, Any]:
        """执行评论分析"""
        comments = kwargs.get('comments', [])
        
        await emit_event(session_id, 'step_ready', {
            'step': 'comment_processing',
            'message': f'正在分析{len(comments)}条评论...'
        })
        await asyncio.sleep(1)
        
        return {
            'total_comments': len(comments),
            'risk_distribution': {
                'low': int(len(comments) * 0.7),
                'medium': int(len(comments) * 0.2),
                'high': int(len(comments) * 0.1)
            },
            'summary': '评论整体积极向上'
        }

# ========== 事件处理函数 ==========

async def handle_start_workflow(session_id: str, data: Dict[Any, Any]):
    """处理工作流启动请求"""
    aweme_id = data.get('aweme_id')
    session = workflow_sessions.get(session_id)
    if session:
        session.aweme_id = aweme_id
    
    logger.info(f"🚀 Starting workflow for video: {aweme_id}")
    
    # 并行启动多个分析任务
    tasks = []
    # 不再强制要求 aweme_id 才能启动任务
    tasks.append(TaskManager.start_task(session_id, TaskType.VIDEO_ANALYSIS, aweme_id=aweme_id))
    tasks.append(TaskManager.start_task(session_id, TaskType.AUDIO_ANALYSIS, aweme_id=aweme_id))
    
    comments = data.get('comments', [])
    if comments:
        tasks.append(TaskManager.start_task(session_id, TaskType.COMMENT_ANALYSIS, comments=comments))
    
    if tasks:
        await asyncio.gather(*tasks)

async def handle_agent_audit(session_id: str, data: Dict[Any, Any]):
    """处理智能体审核请求"""
    aweme_id = data.get('aweme_id')
    logger.info(f"🤖 Agent audit started for: {aweme_id}")
    
    # 启动工作流
    await handle_start_workflow(session_id, data)

async def handle_step_data_request(session_id: str, data: Dict[Any, Any]):
    """处理步骤数据请求"""
    step_name = data.get('step_name')
    session = workflow_sessions.get(session_id)
    
    if not session:
        return
    
    # 检查相关任务是否完成
    available_data = {}
    for task in session.tasks.values():
        if task.status == TaskStatus.COMPLETED and task.result:
            available_data[task.type.value] = task.result
    
    # 发送可用数据
    await emit_event(session_id, 'step_data_ready', {
        'step_name': step_name,
        'available_data': available_data,
        'ready': len(available_data) > 0
    })

# ========== Legacy消息处理 ==========

async def handle_legacy_meta(session_id: str, data: Dict[Any, Any]):
    """处理legacy meta消息"""
    payload = data.get('data', {})
    # 兼容顶层/嵌套两种结构
    aweme_id = data.get('aweme_id') or payload.get('aweme_id')
    # 尝试从 src 中提取
    if not aweme_id:
        src = data.get('src') or payload.get('src') or ''
        # 兼容 https://www.douyin.com/video/<id>
        try:
            import re
            m = re.search(r"/video/(\d+)", src)
            if m:
                aweme_id = m.group(1)
        except Exception:
            pass
    
    # 记录并启动工作流（即使 aweme_id 为空也启动，以保证视频/音频分析执行）
    session = workflow_sessions.get(session_id)
    if session:
        session.aweme_id = aweme_id
    logger.info(f"📝 Meta updated for: {aweme_id or 'unknown'}")
    await handle_start_workflow(session_id, {
        'aweme_id': aweme_id,
        'comments': []
    })

async def handle_legacy_audio(session_id: str, data: Dict[Any, Any]):
    """处理legacy audio消息"""
    # 音频数据处理 - 简化版本
    logger.debug(f"🎵 Audio data received for session: {session_id}")

async def handle_legacy_frame(session_id: str, data: Dict[Any, Any]):
    """处理legacy frame消息"""
    # 帧数据处理 - 简化版本
    logger.debug(f"📹 Frame data received for session: {session_id}")

# ========== WebSocket端点 ==========

@app.websocket("/ws/stream")
async def ws_stream(ws: WebSocket):
    """事件驱动的WebSocket端点"""
    await ws.accept()
    session_id = str(uuid.uuid4())
    
    # 创建工作流会话
    session = WorkflowSession(session_id=session_id, websocket=ws)
    workflow_sessions[session_id] = session
    
    try:
        logger.info(f"🔗 WebSocket connected: {session_id}")
        
        # 发送初始化状态
        await emit_event(session_id, 'system_ready', {
            'ds_available': True,
            'public_url_set': True,
            'message': 'Event-driven system ready'
        })
        
        while True:
            msg = await ws.receive_text()
            try:
                data = json.loads(msg)
                logger.info(f"📨 Received WebSocket message: {data.get('type', 'unknown')} from {session_id}")
                
                # 处理事件驱动的消息
                if data.get('type') == 'start_workflow':
                    logger.info(f"🚀 Processing start_workflow for session {session_id}")
                    await handle_start_workflow(session_id, data.get('data', {}))
                elif data.get('type') == 'agent_audit':
                    await handle_agent_audit(session_id, data.get('data', {}))
                elif data.get('type') == 'request_step_data':
                    await handle_step_data_request(session_id, data.get('data', {}))
                # 处理legacy消息类型
                elif data.get('type') == 'meta':
                    await handle_legacy_meta(session_id, data)
                elif data.get('type') == 'audio':
                    await handle_legacy_audio(session_id, data)
                elif data.get('type') == 'frame':
                    await handle_legacy_frame(session_id, data)
                elif data.get('type') == 'frame_url':
                    # 兼容仅发送帧 URL 的旧实现
                    await handle_legacy_frame(session_id, data)
                else:
                    logger.debug(f"📨 Unhandled message type: {data.get('type')}")
                     
            except Exception as e:
                logger.error(f"❌ Message processing error: {e}")
                continue
    
    except WebSocketDisconnect:
        logger.info(f"🔌 WebSocket disconnected: {session_id}")
    except Exception as e:
        logger.error(f"❌ WebSocket error: {e}")
    finally:
        # 清理资源
        if session_id in workflow_sessions:
            del workflow_sessions[session_id]

# ========== 兼容性API端点 ==========

class AuditRequest(BaseModel):
    aweme_id: Optional[str] = None

class FrameExtractionRequest(BaseModel):
    aweme_id: Optional[str] = None
    video_url: Optional[str] = None

class CommentAnalysisRequest(BaseModel):
    aweme_id: Optional[str] = None
    comments: Optional[List[Any]] = None

@app.post("/agent/audit")
async def agent_audit(request: AuditRequest):
    """智能体审核端点 - 兼容性接口"""
    logger.info(f"🤖 Agent audit requested for: {request.aweme_id}")
    
    # 模拟分析结果 - 与事件驱动结果保持一致
    frames = []
    for i in range(1, 13):
        frames.append({
            'index': i,
            'timestamp': i * 2.5,
            'url': f'/static/frames/frame_{i:03d}.jpg',
            'risk_level': 'low' if i % 4 != 0 else 'medium',
            'analysis': {
                'objects': ['person', 'background', 'text'] if i % 3 == 0 else ['person', 'background'],
                'confidence': 0.85 + (i % 3) * 0.05,
                'description': f'第{i}帧：检测到人物活动'
            }
        })
    
    result = {
        "video_analysis": {
            "frames_extracted": 12,
            "frames": frames,
            "analysis_result": {
                "category": "娱乐",
                "tags": ["音乐", "舞蹈", "生活"],
                "risk_level": "low",
                "confidence": 0.85,
                "summary": "视频内容健康，包含娱乐性舞蹈和音乐元素"
            }
        },
        "audio_analysis": {
            "transcript": "这是一段很有趣的视频内容",
            "keywords": ["有趣", "内容", "视频"],
            "sentiment": "positive"
        },
        "comment_analysis": {
            "total_comments": 10,
            "risk_distribution": {"low": 7, "medium": 2, "high": 1},
            "summary": "评论整体积极向上"
        }
    }
    
    return {"status": "success", "result": result}

@app.post("/extract/frames")
async def extract_frames(request: FrameExtractionRequest):
    """视频帧提取端点 - 兼容性接口"""
    logger.info(f"📹 Frame extraction requested for: {request.aweme_id}")
    
    # 模拟帧提取结果
    frames = [
        {
            "index": i,
            "timestamp": i * 2.5,
            "url": f"/static/frames/frame_{i:03d}.jpg",
            "analysis": {
                "objects": ["person", "background"],
                "confidence": 0.9
            }
        }
        for i in range(1, 13)
    ]
    
    # 兼容前端期望的数据格式
    frame_urls = [frame["url"] for frame in frames]
    
    return {
        "success": True,
        "frame_urls": frame_urls,
        "count": len(frames),
        "frames": frames  # 同时保留完整frames数据供其他用途
    }

@app.post("/agent/analyze_comments")
async def analyze_comments(request: CommentAnalysisRequest = None):
    """评论分析端点 - 兼容性接口"""
    try:
        # 处理可能的空请求或不同格式的请求
        aweme_id = getattr(request, 'aweme_id', None) if request else None
        comments = getattr(request, 'comments', []) if request else []
        
        # 确保comments是列表格式
        if not isinstance(comments, list):
            comments = []
        
        logger.info(f"💬 Comment analysis requested for: {aweme_id}, comments count: {len(comments)}")
        
        # 如果comments为空，生成一些示例数据
        if not comments:
            comments = ["这个视频很棒！", "喜欢这个内容", "不错的分享", "有意思"]
        
        result = {
            "total_comments": len(comments),
            "risk_distribution": {
                "low": int(len(comments) * 0.7),
                "medium": int(len(comments) * 0.2),
                "high": int(len(comments) * 0.1)
            },
            "summary": "评论整体积极向上",
            "details": [
                {
                    "comment": str(comment) if comment else f"评论{i+1}",
                    "risk_level": "low" if i % 3 != 0 else "medium",
                    "confidence": 0.8 + (i % 3) * 0.1
                }
                for i, comment in enumerate(comments[:10])  # 只分析前10条
            ]
        }
        
        return {"status": "success", "result": result}
        
    except Exception as e:
        logger.error(f"❌ Comment analysis error: {e}")
    return {
            "status": "success", 
            "result": {
                "total_comments": 0,
                "risk_distribution": {"low": 0, "medium": 0, "high": 0},
                "summary": "分析完成",
                "details": []
            }
        }

@app.get("/agent/config/reasoning")
async def get_reasoning_config():
    """推理配置端点 - 兼容性接口"""
    return {
        "status": "success",
        "config": {
            "model": "qwen-vl-plus",
            "temperature": 0.1,
            "max_tokens": 4000,
            "reasoning_enabled": True
        }
    }

# ========== 健康检查 ==========

@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {
        "status": "healthy",
        "active_sessions": len(workflow_sessions),
        "timestamp": int(time.time() * 1000)
    }

@app.get("/")
async def root():
    """根端点"""
    return {
        "service": "Event-Driven Agent Backend", 
        "status": "running",
        "active_sessions": len(workflow_sessions)
    }

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Event-Driven Agent Backend...")
    uvicorn.run(app, host="0.0.0.0", port=8799, log_level="info")
