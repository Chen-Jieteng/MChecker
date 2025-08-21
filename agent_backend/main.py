from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Response
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import uuid, tempfile, subprocess, os, json, time, base64
import random
import logging
import httpx
import asyncio

try:
    os.environ['DASHSCOPE_VL_MODEL'] = 'qwen-vl-max'
    print(" 已强制设置视觉模型: qwen-vl-max")
    from dashscope_client import DSClient, VisionDSClient
    from vision_api_coordinator import get_vision_coordinator
    test_ds = DSClient()
    vision_ds = VisionDSClient()
    print(f" DSClient initialized successfully at startup")
    print(f" VisionDSClient initialized successfully")
except Exception as e:
    print(f" DSClient initialization failed: {e}")
    DSClient = None
    VisionDSClient = None

app = FastAPI(title="MChecker Agent Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源，生产环境应该限制
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有HTTP方法
    allow_headers=["*"],  # 允许所有请求头
)

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
logger = logging.getLogger("agent")

STATIC_ROOT = os.path.abspath(os.getenv("AGENT_STATIC_ROOT", os.path.join("agent_backend", "static")))
os.makedirs(STATIC_ROOT, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_ROOT), name="static")

PUBLIC_BASE_URL = (os.getenv("PUBLIC_BASE_URL", "").strip().rstrip('/'))

class AuditRequest(BaseModel):
    aweme_id: Optional[str] = None
    video_url: Optional[str] = None
    audio_url: Optional[str] = None
    title: Optional[str] = None
    desc: Optional[str] = None

class CommentAnalysisRequest(BaseModel):
    aweme_id: Optional[str] = None
    comments: List[Dict[str, Any]] = []

@app.get("/health")
async def health():
    return {"ok": True}

class ReasoningConfig(BaseModel):
    step_limit: int = 15  # 增加步骤数以支持更全面的视频分析
    tick_seconds: float = 2.0
    use_asr: bool = True
    temperature: Optional[float] = None
    top_p: Optional[float] = None

_REASONING_CFG = ReasoningConfig(step_limit=15, tick_seconds=2.0, use_asr=True, temperature=0.3, top_p=0.9)

@app.post("/agent/config/reasoning")
async def set_reasoning_config(cfg: ReasoningConfig):
    global _REASONING_CFG
    _REASONING_CFG = cfg
    if cfg.temperature is not None:
        os.environ['DASHSCOPE_TEMPERATURE'] = str(cfg.temperature)
    if cfg.top_p is not None:
        os.environ['DASHSCOPE_TOP_P'] = str(cfg.top_p)
    return {"ok": True, "config": _REASONING_CFG.dict()}

@app.get("/agent/config/reasoning")
async def get_reasoning_config():
    return _REASONING_CFG.dict()

@app.get("/agent/api/status")
async def get_api_status():
    """获取API协调器状态"""
    coordinator = get_vision_coordinator()
    if coordinator:
        status = coordinator.get_status()
        status['coordinator_enabled'] = True
    else:
        status = {
            'coordinator_enabled': False,
            'message': '未配置视觉API协调器，使用默认轮换机制',
            'total_keys': 1,
            'total_concurrent_requests': 0,
            'total_available_slots': 'unlimited'
        }
    
    return status

@app.get("/api/frames/info")
async def get_frames_info():
    """获取当前提取的帧信息"""
    import os
    import glob
    import re
    
    frames_dir = "agent_backend/static/extracted_frames"
    if not os.path.exists(frames_dir):
        return {"frame_count": 0, "max_frame_number": 0, "frames": []}
    
    frame_files = glob.glob(os.path.join(frames_dir, "frame_*.jpg"))
    frame_files.sort()
    
    frames = []
    max_frame_number = 0
    for frame_file in frame_files:
        filename = os.path.basename(frame_file)
        match = re.match(r'frame_(\d+)\.jpg', filename)
        if match:
            frame_number = int(match.group(1))
            max_frame_number = max(max_frame_number, frame_number)
            frames.append({
                "number": frame_number,
                "filename": filename,
                "url": f"http://127.0.0.1:8799/static/extracted_frames/{filename}"
            })
    
    return {
        "frame_count": len(frames),
        "max_frame_number": max_frame_number,
        "frames": frames
    }

@app.options("/proxy/video")
async def proxy_video_options():
    return Response(status_code=204, headers={
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
        "Access-Control-Allow-Headers": "*",
        "Content-Length": "0",
    })

@app.get("/proxy/video")
async def proxy_video(url: str):
    """代理视频请求，解决跨域问题"""
    if not url:
        raise HTTPException(status_code=400, detail="Missing URL parameter")
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            async with client.stream("GET", url) as response:
                if response.status_code != 200:
                    raise HTTPException(status_code=response.status_code, detail="Failed to fetch video")
                
                content_type = response.headers.get("content-type", "video/mp4")
                content_length = response.headers.get("content-length")
                
                headers = {
                    "Content-Type": content_type,
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
                    "Access-Control-Allow-Headers": "*",
                }
                if content_length:
                    headers["Content-Length"] = content_length
                
                async def generate():
                    async for chunk in response.aiter_bytes(chunk_size=8192):
                        yield chunk
                
                return StreamingResponse(generate(), headers=headers)
    except httpx.TimeoutException:
        raise HTTPException(status_code=408, detail="Request timeout")
    except Exception as e:
        logger.error(f"Video proxy error: {e}")
        raise HTTPException(status_code=500, detail=f"Proxy error: {str(e)}")

@app.post("/fix/frames")
async def fix_frame_extraction(request: dict):
    """彻底修复帧提取 - 绕过所有缓存"""
    video_url = request.get("video_url", "https://www.douyin.com/aweme/v1/play/?video_id=v0200fa50000bqv2ovedm15352jvv5vg&line=0&file_id=efac24de9d2548228975fc8429e5bdcb&sign=3b7c4acc3b831e92448d6909510074c0&is_play_url=1&source=PackSourceEnum_PUBLISH")
    count = request.get("count", 30)
    
    import tempfile
    import subprocess
    import os
    
    tmpdir = tempfile.mkdtemp(prefix="fix_frames_")
    outpat = os.path.join(tmpdir, "frame_%02d.jpg")
    
    cmd = ["ffmpeg", "-y", "-i", video_url, "-vf", "fps=1", "-vframes", str(count), outpat]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        frame_files = []
        for i in range(1, count + 1):
            frame_path = outpat.replace("%02d", f"{i:02d}")
            if os.path.exists(frame_path):
                frame_files.append(frame_path)
        
        frame_urls = []
        for path in frame_files:
            filename = os.path.basename(path)
            static_path = os.path.join(STATIC_ROOT, "extracted_frames", filename)
            os.makedirs(os.path.dirname(static_path), exist_ok=True)
            
            if os.path.exists(path):
                import shutil
                shutil.move(path, static_path)
                rel_url = f"/static/extracted_frames/{filename}"
                public_url = f"http://127.0.0.1:8799{rel_url}"
                frame_urls.append(public_url)
        
        return {
            "success": True,
            "frame_urls": frame_urls,
            "count": len(frame_urls),
            "debug": f"Generated {len(frame_files)} frames, moved {len(frame_urls)} to static"
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "debug": f"Command failed: {' '.join(cmd)}"
        }

@app.post("/extract/frames")
async def extract_frames_from_url(request: dict):
    """从视频URL直接提取帧，绕过前端跨域限制"""
    video_url = request.get("video_url")
    if not video_url:
        raise HTTPException(status_code=400, detail="Missing video_url")
    
    try:
        logger.info(f"开始提取30帧: {video_url}")
        frame_paths = _extract_frames(video_url, fps=0, count=30)
        logger.info(f"实际提取帧数: {len(frame_paths)}")
        if not frame_paths:
            return {"success": False, "error": "No frames extracted"}
        
        frame_urls = []
        for path in frame_paths:
            filename = os.path.basename(path)
            static_path = os.path.join(STATIC_ROOT, "extracted_frames", filename)
            os.makedirs(os.path.dirname(static_path), exist_ok=True)
            
            if os.path.exists(path):
                import shutil
                shutil.move(path, static_path)
                
                rel_url = f"/static/extracted_frames/{filename}"
                if PUBLIC_BASE_URL:
                    public_url = f"{PUBLIC_BASE_URL}{rel_url}"
                else:
                    public_url = f"http://127.0.0.1:8799{rel_url}"
                frame_urls.append(public_url)
        
        return {
            "success": True,
            "frame_urls": frame_urls,
            "count": len(frame_urls)
        }
    except Exception as e:
        logger.error(f"Frame extraction error: {e}")
        return {"success": False, "error": str(e)}


def _extract_frames(video_url: str, fps: int = 0, count: int = 30) -> List[str]:
    """提取视频帧，强制确保返回指定数量的帧"""
    paths: List[str] = []
    if not video_url:
        return paths
    
    tmpdir = tempfile.mkdtemp(prefix="mchecker_frames_")
    outpat = os.path.join(tmpdir, "frame_%02d.jpg")
    
    try:
        if fps == 0:
            try:
                probe_cmd = ["ffprobe", "-v", "quiet", 
                            "-user_agent", "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
                            "-headers", "Referer: https://www.douyin.com/",
                            "-show_entries", "format=duration", "-of", "csv=p=0", video_url]
                probe_result = subprocess.run(probe_cmd, capture_output=True, text=True)
                video_duration = float(probe_result.stdout.strip()) if probe_result.returncode == 0 else 30.0
                
                interval = max(1.0, video_duration / count)
                target_fps = 1.0 / interval
                
                cmd = ["ffmpeg", "-y", 
                       "-user_agent", "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
                       "-headers", "Referer: https://www.douyin.com/",
                       "-reconnect", "1", "-reconnect_streamed", "1", "-reconnect_delay_max", "5",
                       "-timeout", "30000000",
                       "-i", video_url, "-vf", f"fps={target_fps}", "-vframes", str(count), outpat]
                print(f" 视频时长{video_duration:.1f}s，使用fps={target_fps:.3f}策略提取{count}帧")
            except:
                cmd = ["ffmpeg", "-y", 
                       "-user_agent", "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
                       "-headers", "Referer: https://www.douyin.com/",
                       "-reconnect", "1", "-reconnect_streamed", "1", "-reconnect_delay_max", "5",
                       "-timeout", "30000000",
                       "-i", video_url, "-vf", "fps=0.2", "-vframes", str(count), outpat]
                print(f" 使用回退策略fps=0.2提取{count}帧")
        else:
            cmd = ["ffmpeg", "-y", 
                   "-user_agent", "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
                   "-headers", "Referer: https://www.douyin.com/",
                   "-reconnect", "1", "-reconnect_streamed", "1", "-reconnect_delay_max", "5",
                   "-timeout", "30000000",
                   "-i", video_url, "-vf", f"fps={fps}", "-vframes", str(count), outpat]
        
        print(f"DEBUG: Executing command for {count} frames: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"ERROR: ffmpeg failed with code {result.returncode}")
            print(f"STDERR: {result.stderr}")
            return paths
        
        for i in range(1, count + 1):
            frame_path = outpat.replace("%02d", f"{i:02d}")
            if os.path.exists(frame_path):
                paths.append(frame_path)
        
        print(f"DEBUG: Successfully extracted {len(paths)} frames out of {count} requested")
        return paths
        
    except Exception as e:
        print(f"EXCEPTION in _extract_frames: {e}")
        import traceback
        traceback.print_exc()
        return []

async def _asr(ds: DSClient, audio_url: Optional[str]) -> str:
    if not audio_url:
        return ""
    try:
        r = await ds.asr_paraformer(audio_url)
        text = r.get("output",{}).get("text") or r.get("result",{}).get("text") or ""
        return text
    except Exception:
        return ""

async def _send_trace(ws: WebSocket, role: str, text: str, stage: str, payload: Optional[Dict[str, Any]] = None, kind: Optional[str] = None):
    try:
        if hasattr(ws, 'client_state') and ws.client_state.value != 1:  # WebSocketState.CONNECTED = 1
            print(f" WebSocket连接已断开，跳过发送trace")
            return False
            
        data = {'role': role, 'text': text, 'stage': stage, 'ts': int(time.time() * 1000)}
        if payload is not None:
            data['payload'] = payload
        if kind is not None:
            data['kind'] = kind
        await ws.send_text(json.dumps({'type': 'trace', 'data': data}, ensure_ascii=False))
        return True
    except Exception as e:
        print(f" WebSocket发送trace失败: {e}")
        return False  # 不再抛出异常，避免程序崩溃


async def _send_tool(ws: WebSocket, name: str, payload: Optional[Dict[str, Any]] = None):
    try:
        await ws.send_text(json.dumps({'type': 'tool', 'name': name, 'payload': payload or {}, 'ts': int(time.time() * 1000)}, ensure_ascii=False))
    except Exception:
        pass

async def _vision_describe(ds: DSClient, buf: 'SessionBuf') -> Dict[str, Any]:
    """分段视频分析：按时间窗口分析视频帧，识别高风险片段"""
    import base64
    
    total_frames = len(buf.frames)
    if total_frames == 0:
        return {'text': '', 'segments': [], 'risk_segments': []}
    
    current_window_start = max(0, total_frames - 4)
    current_frames = buf.frames[current_window_start:]
    
    img_inputs: List[str] = []
    frame_timestamps = []
    
    for i, frame_path in enumerate(current_frames):
        timestamp = (current_window_start + i) * 0.5
        frame_timestamps.append(timestamp)
        
        if isinstance(frame_path, str) and frame_path.startswith('http://127.0.0.1:8799/static/'):
            local_path = frame_path.replace('http://127.0.0.1:8799/static/', 'agent_backend/static/')
            logger.info(f" 处理图片: {frame_path} -> {local_path}")
            try:
                if os.path.exists(local_path):
                    with open(local_path, 'rb') as f:
                        img_data = base64.b64encode(f.read()).decode('utf-8')
                        img_inputs.append(f"data:image/jpeg;base64,{img_data}")
                        logger.info(f" 成功编码图片: {local_path} ({len(img_data)} chars)")
                else:
                    logger.error(f" 图片文件不存在: {local_path}")
            except Exception as e:
                logger.error(f" 图片编码失败 {local_path}: {e}")
                continue
        elif isinstance(frame_path, str) and frame_path.startswith(('http://', 'https://')):
            img_inputs.append(frame_path)
        elif isinstance(frame_path, str) and os.path.exists(frame_path):
            try:
                with open(frame_path, 'rb') as f:
                    img_data = base64.b64encode(f.read()).decode('utf-8')
                    img_inputs.append(f"data:image/jpeg;base64,{img_data}")
            except Exception as e:
                logger.error(f"Failed to encode local image {frame_path}: {e}")
                continue
    
    if not img_inputs:
        logger.warning(f" 无可用图片输入: frames={len(buf.frames)}, current_frames={len(current_frames)}")
        return {'text': '', 'segments': [], 'risk_segments': []}
    
    logger.info(f" 准备调用视觉模型: {len(img_inputs)}张图片")
    
    enhanced_prompt = f"""
分析这{len(img_inputs)}帧连续视频画面（时间窗口: {frame_timestamps[0]:.1f}s - {frame_timestamps[-1]:.1f}s）：

1. **内容描述**：简要描述人物、场景、动作、文字、物品
2. **风险识别**：识别任何可能的违规内容（暴力、色情、政治敏感、虚假信息、违法等）
3. **风险等级**：如发现风险，评估等级（低风险/中风险/高风险/违禁）
4. **关键时刻**：标注关键动作或风险点出现的具体时间

格式：[内容描述] | [风险等级：无/低/中/高/禁] | [关键时刻：X.Xs]
    """
    
    r_vl = await ds.qwen_vl(img_inputs, prompt=enhanced_prompt)
    logger.info(f" 视觉模型调用完成: 收到响应 {type(r_vl)}")
    
    text = ''
    out = r_vl.get('output') if isinstance(r_vl, dict) else {}
    
    if isinstance(out, dict):
        if 'text' in out:
            text = out.get('text', '')
            logger.info(f" 使用直接text结构提取: {len(text)}字符")
        
        elif 'choices' in out:
            choices = out.get('choices', [])
            if choices and isinstance(choices[0], dict):
                msg = choices[0].get('message', {})
                content = msg.get('content', [])
                if content and isinstance(content[0], dict):
                    text = content[0].get('text', '')
                    logger.info(f" 使用choices结构提取: {len(text)}字符")
    
    if not text:
        text = json.dumps(r_vl, ensure_ascii=False)[:500]
        logger.warning(f" 使用原始响应作为文本: {len(text)}字符")
    
    risk_segments = []
    segments_info = []
    
    if text and '|' in text:
        parts = text.split('|')
        if len(parts) >= 3:
            content_desc = parts[0].strip()
            risk_part = parts[1].strip()
            time_part = parts[2].strip()
            
            risk_level = 'low'
            if '高风险' in risk_part or '高' in risk_part:
                risk_level = 'high'
            elif '中风险' in risk_part or '中' in risk_part:
                risk_level = 'medium'  
            elif '违禁' in risk_part or '禁' in risk_part:
                risk_level = 'ban'
            elif '低风险' in risk_part or '低' in risk_part:
                risk_level = 'low'
            elif '无' in risk_part:
                risk_level = 'none'
            
            import re
            time_match = re.search(r'(\d+\.?\d*)s', time_part)
            timestamp = float(time_match.group(1)) if time_match else frame_timestamps[0]
            
            segment = {
                'start_time': frame_timestamps[0],
                'end_time': frame_timestamps[-1],
                'risk_level': risk_level,
                'description': content_desc,
                'key_timestamp': timestamp,
                'frames_analyzed': len(img_inputs)
            }
            
            segments_info.append(segment)
            
            if risk_level in ['high', 'ban']:
                risk_segments.append({
                    'timestamp': timestamp,
                    'duration': frame_timestamps[-1] - frame_timestamps[0],
                    'risk_level': risk_level,
                    'description': content_desc,
                    'frames_count': len(img_inputs)
                })
    
    return {
        'text': text, 
        'segments': segments_info,
        'risk_segments': risk_segments,
        'window_start': frame_timestamps[0] if frame_timestamps else 0,
        'window_end': frame_timestamps[-1] if frame_timestamps else 0,
        'total_frames_analyzed': total_frames
    }

async def _analyze_single_segment(segment: Dict, video_duration: float, buf: 'SessionBuf', segment_index: int) -> Dict:
    """分析单个片段 - 并行调用的独立函数"""
    import time
    try:
        start_time = time.time()
        print(f" 开始并行分析片段{segment_index}: {segment['start_time']:.1f}s-{segment['end_time']:.1f}s (时刻: {start_time:.3f})")
        segment_summary = await _generate_time_based_segment_summary(
            {}, segment, video_duration, buf
        )
        end_time = time.time()
        print(f" 并行完成片段{segment_index}分析: {segment['start_time']:.1f}s-{segment['end_time']:.1f}s (耗时: {end_time - start_time:.3f}s)")
        return segment_summary
    except Exception as e:
        print(f" 并行分析片段{segment_index}失败: {e}")
        return {
            'segment_index': segment_index,
            'time_range': f'{segment["start_time"]:.2f}s-{segment["end_time"]:.2f}s',
            'duration': f'{segment["duration"]:.2f}秒',
            'content': f'片段{segment_index}分析失败: {str(e)}',
            'risk_level': 'low',
            'key_findings': []
        }

async def _preload_segment_analysis(time_segments: List[Dict], video_duration: float, buf: 'SessionBuf', ws: WebSocket, cache: Dict) -> None:
    """后台预加载所有片段分析 - 使用API协调器实现智能并行"""
    try:
        coordinator = get_vision_coordinator()
        
        if coordinator:
            print(f" 开始智能并行预加载片段分析: {len(time_segments)}个片段（使用API协调器）")
            print(f" 协调器状态: {coordinator.get_status()}")
            
            await _send_trace(ws, 'system', 
                f' 启动{len(time_segments)}个片段的智能并行分析（无锁协调器）...', 
                'smart_parallel_analysis_start', {
                    'total_segments': len(time_segments),
                    'parallel_mode': True,
                    'lockless_coordinator': True,
                    'coordinator_status': coordinator.get_status()
                })
            
            max_concurrency = int(os.getenv('VL_MAX_CONCURRENCY', '6'))
            print(f" 智能并行发起分析（限流并发={max_concurrency}）...")

            semaphore = asyncio.Semaphore(max_concurrency)

            async def run_with_retry(seg, idx):
                backoff = 0.5
                max_backoff = 8.0
                attempts = 0
                while True:
                    attempts += 1
                    async with semaphore:
                        try:
                            return await _analyze_single_segment(seg, video_duration, buf, idx)
                        except Exception as e:
                            msg = str(e)
                            if '429' in msg or 'Too Many Requests' in msg:
                                sleep_s = min(max_backoff, backoff * (2 ** (attempts - 1))) * (1 + 0.2 * random.random())
                                print(f"⏳ 片段{idx}命中限流429，退避{sleep_s:.2f}s后重试(第{attempts}次)")
                                await asyncio.sleep(sleep_s)
                                continue
                            else:
                                raise

            start_time = time.time()
            segment_results = []
            batch = []
            for i, segment in enumerate(time_segments):
                batch.append(run_with_retry(segment, i + 1))
                if len(batch) >= max_concurrency:
                    segment_results.extend(await asyncio.gather(*batch, return_exceptions=True))
                    batch = []
            if batch:
                segment_results.extend(await asyncio.gather(*batch, return_exceptions=True))
            end_time = time.time()

            print(f" 智能并行分析完成！总耗时: {end_time - start_time:.2f}秒，完成片段={len(segment_results)}")
            
        else:
            print(f" 未配置API协调器，回退到顺序处理: {len(time_segments)}个片段")
            
            await _send_trace(ws, 'system', 
                f' API协调器未配置，启动{len(time_segments)}个片段的顺序分析...', 
                'fallback_sequential_analysis_start', {
                    'total_segments': len(time_segments),
                    'parallel_mode': False,
                    'reason': 'no_coordinator_configured'
                })
            
            start_time = time.time()
            segment_results = []
            
            for i, segment in enumerate(time_segments):
                try:
                    print(f" 顺序分析片段{i+1}/{len(time_segments)}: {segment['start_time']:.1f}s-{segment['end_time']:.1f}s")
                    
                    result = await _analyze_single_segment(segment, video_duration, buf, i + 1)
                    segment_results.append(result)
                    
                    print(f" 片段{i+1}分析完成，进度: {((i+1)/len(time_segments)*100):.1f}%")
                    
                    try:
                        await _send_trace(ws, 'system', 
                            f' 片段{i+1}/{len(time_segments)}分析完成 ({((i+1)/len(time_segments)*100):.1f}%)', 
                            'segment_progress', {
                                'completed': i + 1,
                                'total': len(time_segments),
                                'progress': (i + 1) / len(time_segments) * 100
                            })
                    except:
                        pass
                    
                    if i < len(time_segments) - 1:
                        await asyncio.sleep(0.5)
                        
                except Exception as e:
                    print(f" 片段{i+1}分析异常: {e}")
                    fallback_segment = {
                        'segment_index': i + 1,
                        'time_range': f'{segment["start_time"]:.1f}s-{segment["end_time"]:.1f}s',
                        'duration': f'{segment["duration"]:.1f}秒',
                        'content': f'片段{i+1}分析异常: {str(e)}',
                        'risk_level': 'low',
                        'key_findings': []
                    }
                    segment_results.append(fallback_segment)
            
            end_time = time.time()
            print(f" 顺序分析完成！总耗时: {end_time - start_time:.2f}秒")
        
        successful_segments = []
        failed_count = 0
        
        for i, result in enumerate(segment_results):
            if isinstance(result, Exception):
                print(f" 片段{i+1}分析异常: {result}")
                fallback_segment = {
                    'segment_index': i + 1,
                    'time_range': f'{time_segments[i]["start_time"]:.1f}s-{time_segments[i]["end_time"]:.1f}s',
                    'duration': f'{time_segments[i]["duration"]:.1f}秒',
                    'content': f'片段{i+1}分析异常: {str(result)}',
                    'risk_level': 'low',
                    'key_findings': []
                }
                successful_segments.append(fallback_segment)
                failed_count += 1
            else:
                successful_segments.append(result)
                print(f" 片段{i+1}智能并行分析成功")
                
                try:
                    await _send_trace(ws, 'assistant', 
                        f' 并行片段总结 ({result.get("time_range", "未知")}): {result.get("content", "")[:100]}...', 
                        'segment_summary', result)
                    print(f" 已发送片段{i+1}的segment_summary到前端")
                except Exception as send_e:
                    print(f" 发送片段{i+1}的segment_summary失败: {send_e}")
        
        cache['completed_segments'] = successful_segments
        cache['analysis_complete'] = len([s for s in successful_segments if isinstance(s, dict) and s.get('content')]) > 0
        try:
            buf.segment_analysis_completed = cache['analysis_complete']
        except Exception:
            pass
        cache['analysis_progress'] = 100 if cache['analysis_complete'] else 0
        
        print(f" 审核数据已全部到达！成功分析{len(successful_segments) - failed_count}个片段，失败{failed_count}个片段")
        print(f" 顺序分析统计: 成功{len(successful_segments) - failed_count}个, 失败{failed_count}个, 总耗时{end_time - start_time:.2f}秒")
        
        if cache['analysis_complete']:
            await _send_trace(ws, 'system', 
                f' 顺序分析完成: {len(successful_segments)}个片段 (耗时{end_time - start_time:.2f}s)', 
                'sequential_analysis_complete', {
                    'total_segments': len(successful_segments),
                    'successful': len(successful_segments) - failed_count,
                    'failed': failed_count,
                    'duration': end_time - start_time,
                    'analysis_complete': True
                })
        else:
            await _send_trace(ws, 'system', 
                ' 触发限流，暂未获得有效片段，总结未完成；将继续退避重试', 
                'sequential_analysis_pending', {
                    'total_segments': len(successful_segments),
                    'successful': len(successful_segments) - failed_count,
                    'failed': failed_count,
                    'analysis_complete': False
                })
            
    except Exception as e:
        print(f" 并行预加载分析失败: {e}")
        cache['analysis_complete'] = False
        try:
            buf.segment_analysis_completed = False
        except Exception:
            pass
        completed_segments = cache.get('completed_segments', [])
        print(f" 异常中断，已分析{len(completed_segments)}个片段；未完成，将等待重试")

async def _generate_time_based_segment_summary(memory: Dict, segment_info: Dict, video_duration: float, buf: 'SessionBuf' = None) -> Dict[str, Any]:
    """生成基于时间的视频片段总结"""
    try:
        start_time = segment_info['start_time']
        end_time = segment_info['end_time']
        segment_index = segment_info['index']
        
        segment_content = ""
        segment_risk_level = "low"
        
        buf_frames = buf.frames if buf else []
        total_frames = len(buf_frames)
        
        if total_frames > 0 and video_duration > 0:
            start_frame_idx = int((start_time / video_duration) * total_frames)
            end_frame_idx = int((end_time / video_duration) * total_frames)
            
            start_frame_idx = max(0, min(start_frame_idx, total_frames - 1))
            end_frame_idx = max(start_frame_idx, min(end_frame_idx, total_frames - 1))
            
            segment_frames = buf_frames[start_frame_idx:end_frame_idx + 1]
            
            if not segment_frames and total_frames > 0:
                segment_frames = [buf_frames[min(start_frame_idx, total_frames - 1)]]
                
            print(f" 时间段{start_time:.1f}s-{end_time:.1f}s: 选择帧{start_frame_idx}-{end_frame_idx} (共{len(segment_frames)}帧)")
            
            if segment_frames and VisionDSClient:
                try:
                    vision_client = VisionDSClient()
                    coordinator = get_vision_coordinator()
                    
                    img_inputs = []
                    for frame_path in segment_frames:
                        if isinstance(frame_path, str) and frame_path.startswith('http://127.0.0.1:8799/static/'):
                            local_path = frame_path.replace('http://127.0.0.1:8799/static/', 'agent_backend/static/')
                            print(f" 处理时间段图片: {frame_path} -> {local_path}")
                            try:
                                if os.path.exists(local_path):
                                    with open(local_path, 'rb') as f:
                                        img_data = base64.b64encode(f.read()).decode('utf-8')
                                        img_inputs.append(f"data:image/jpeg;base64,{img_data}")
                                        print(f" 成功编码时间段图片: {local_path} ({len(img_data)} chars)")
                                else:
                                    print(f" 时间段图片文件不存在: {local_path}")
                            except Exception as e:
                                print(f" 时间段图片编码失败 {local_path}: {e}")
                                continue
                        elif isinstance(frame_path, str) and os.path.exists(frame_path):
                            try:
                                with open(frame_path, 'rb') as f:
                                    img_data = base64.b64encode(f.read()).decode('utf-8')
                                    img_inputs.append(f"data:image/jpeg;base64,{img_data}")
                            except Exception as e:
                                print(f" 时间段本地图片编码失败 {frame_path}: {e}")
                                continue
                    
                    if not img_inputs:
                        print(f" 时间段{start_time:.1f}s-{end_time:.1f}s: 无可用图片输入")
                        segment_content = f"第{segment_index}个时间段({start_time:.1f}s-{end_time:.1f}s)的视觉内容分析：无可用图片数据"
                    else:
                        time_prompt = f"""

- **主要人物/物体的动作和状态**：
- **场景环境的变化**：
- **任何值得注意的细节**：

请用一句自然流畅的话描述这个时间段的核心内容，例如："视频中展示了一位女性在室内介绍产品的场景"

- **人物表现**：
- **背景环境**：
- **文字信息**：

- **风险等级**：低风险
  - 理由：简要说明判定为该等级的原因

可选等级说明：
- 低风险：正常健康内容，无明显违规点
- 中风险：存在争议但不严重的内容
- 高风险：明确违反社区规范的内容
- 禁：严重违法违规，需要立即封禁

- 置信度：0.9

在这{end_time - start_time:.1f}秒钟的视频片段中，...
"""
                        
                        print(f" 时间段{start_time:.1f}s-{end_time:.1f}s准备调用视觉模型: {len(img_inputs)}张图片")
                        
                        if coordinator:
                            vl_result = await vision_client.qwen_vl_with_coordinator(
                                img_inputs, time_prompt, coordinator
                            )
                        else:
                            ds = DSClient()
                            vl_result = await ds.qwen_vl(img_inputs, prompt=time_prompt)
                    
                    print(f" 时间段{start_time:.1f}s-{end_time:.1f}s DashScope返回结果调试:")
                    print(f"   vl_result存在: {vl_result is not None}")
                    if vl_result:
                        print(f"   vl_result类型: {type(vl_result)}")
                        print(f"   vl_result键: {list(vl_result.keys()) if isinstance(vl_result, dict) else 'not dict'}")
                        if 'output' in vl_result:
                            output = vl_result['output']
                            print(f"   output类型: {type(output)}")
                            print(f"   output键: {list(output.keys()) if isinstance(output, dict) else 'not dict'}")
                            
                            if isinstance(output, dict):
                                if 'text' in output:
                                    text_content = output.get('text', '')
                                    print(f"   直接text长度: {len(text_content)}")
                                    print(f"   直接text前100字符: {repr(text_content[:100])}")
                                
                                if 'choices' in output:
                                    choices = output.get('choices', [])
                                    print(f"   choices长度: {len(choices)}")
                                    if choices:
                                        print(f"   choices[0]键: {list(choices[0].keys()) if isinstance(choices[0], dict) else 'not dict'}")
                                        if isinstance(choices[0], dict):
                                            message = choices[0].get('message', {})
                                            print(f"   message键: {list(message.keys()) if isinstance(message, dict) else 'not dict'}")
                        
                        print(f"   完整结果前500字符: {str(vl_result)[:500]}")
                    else:
                        print(f"   vl_result为None或False")
                    
                    if vl_result and vl_result.get('output'):
                        try:
                            output = vl_result['output']
                            output_text = ''
                            
                            if isinstance(output, dict) and 'text' in output:
                                output_text = output.get('text', '')
                                print(f" 使用直接text结构获取: {len(output_text)}字符")
                            
                            elif isinstance(output, dict) and 'choices' in output:
                                choices = output.get('choices', [])
                                if choices and len(choices) > 0:
                                    message = choices[0].get('message', {})
                                    content_list = message.get('content', [])
                                    if content_list and len(content_list) > 0:
                                        output_text = content_list[0].get('text', '')
                                        print(f" 使用choices结构获取: {len(output_text)}字符")
                            
                            elif isinstance(output, str):
                                output_text = output
                                print(f" 使用字符串结构获取: {len(output_text)}字符")
                            
                            if not output_text and isinstance(output, dict):
                                def find_text_in_dict(d, path=""):
                                    if isinstance(d, dict):
                                        for k, v in d.items():
                                            current_path = f"{path}.{k}" if path else k
                                            if k == 'text' and isinstance(v, str) and v.strip():
                                                return v, current_path
                                            elif isinstance(v, (dict, list)):
                                                result = find_text_in_dict(v, current_path)
                                                if result[0]:
                                                    return result
                                    elif isinstance(d, list):
                                        for i, item in enumerate(d):
                                            current_path = f"{path}[{i}]" if path else f"[{i}]"
                                            result = find_text_in_dict(item, current_path)
                                            if result[0]:
                                                return result
                                    return '', ''
                                
                                found_text, found_path = find_text_in_dict(output)
                                if found_text:
                                    output_text = found_text
                                    print(f" 在路径 {found_path} 找到text: {len(output_text)}字符")
                            
                            if output_text:
                                segment_content = output_text[:800]  # 增加长度限制
                                
                                segment_risk_level = "low"  # 默认低风险
                                
                                risk_patterns = [
                                    (['禁', '违禁', 'ban'], 'ban'),
                                    (['高风险', '高', 'high', '严重', '违规'], 'high'),  
                                    (['中风险', '中', 'medium', '警告', '注意'], 'medium'),
                                    (['低风险', '低', 'low'], 'low'),
                                    (['无风险', '无', 'none', '正常'], 'low')
                                ]
                                
                                risk_section = ""
                                if "风险等级" in output_text:
                                    risk_start = output_text.find("风险等级")
                                    if risk_start != -1:
                                        risk_section = output_text[risk_start:risk_start+50].lower()
                                elif "risk" in output_text.lower():
                                    risk_start = output_text.lower().find("risk")
                                    if risk_start != -1:
                                        risk_section = output_text[risk_start:risk_start+50].lower()
                                
                                if not risk_section:
                                    risk_section = output_text.lower()
                                
                                for keywords, level in risk_patterns:
                                    if any(keyword in risk_section for keyword in keywords):
                                        segment_risk_level = level
                                        break
                                    
                                segment_confidence = None
                                try:
                                    import re
                                    m = re.search(r"置信度[：: ]\s*([0-9]+(?:\.[0-9]+)?)%?", output_text)
                                    if m:
                                        val = float(m.group(1))
                                        segment_confidence = val/100.0 if val > 1 else val
                                except Exception:
                                    segment_confidence = None

                                print(f" 时间段{start_time:.1f}s-{end_time:.1f}s视觉分析完成: {len(segment_content)}字符，confidence={segment_confidence}")
                                print(f" 分析内容预览: {segment_content[:200]}...")
                            else:
                                print(f" 时间段{start_time:.1f}s-{end_time:.1f}s: 所有解析方法都未找到文本内容")
                                print(f"   output结构: {type(output)}")
                                print(f"   output内容预览: {str(output)[:200]}")
                                segment_content = f"第{segment_index}个时间段({start_time:.1f}s-{end_time:.1f}s)的视觉内容分析：响应解析失败"
                        except Exception as parse_error:
                            print(f" 时间段{start_time:.1f}s-{end_time:.1f}s: 响应解析错误: {parse_error}")
                            segment_content = f"第{segment_index}个时间段({start_time:.1f}s-{end_time:.1f}s)的视觉内容分析：解析错误"
                    else:
                        print(f" 时间段{start_time:.1f}s-{end_time:.1f}s: DashScope返回结构异常")
                        segment_content = f"第{segment_index}个时间段({start_time:.1f}s-{end_time:.1f}s)的视觉内容分析失败：DashScope返回异常"
                        
                except Exception as e:
                    error_message = str(e) if str(e) else f"{type(e).__name__}: {repr(e)}"
                    print(f" 时间段{start_time:.1f}s-{end_time:.1f}s视觉分析失败: {error_message}")
                    if '429' in error_message or 'Too Many Requests' in error_message:
                        raise
                    segment_content = f"第{segment_index}个时间段({start_time:.1f}s-{end_time:.1f}s)的视觉内容分析失败: {error_message}"
            else:
                segment_content = f"第{segment_index}个时间段({start_time:.1f}s-{end_time:.1f}s)的视觉内容分析：帧数据不可用"
        else:
            segment_content = f"第{segment_index}个时间段({start_time:.1f}s-{end_time:.1f}s)的视觉内容分析：视频数据不可用"
        
        time_based_segment = {
            'segment_index': segment_index,
            'time_range': f'{start_time:.2f}s-{end_time:.2f}s',
            'duration': f'{end_time - start_time:.2f}秒',
            'progress_range': f'{(start_time/video_duration)*100:.1f}%-{(end_time/video_duration)*100:.1f}%' if video_duration > 0 else f'片段{segment_index}',
            'content': segment_content,
            'risk_level': segment_risk_level,
            'confidence': segment_confidence if 'segment_confidence' in locals() else None,
            'key_findings': [f"时间{start_time:.2f}s-{end_time:.2f}s: 基于实际帧内容的分析结果"]
        }
        
        return time_based_segment
        
    except Exception as e:
        print(f"生成时间片段总结失败: {e}")
        return None

async def _generate_segment_summary(memory: Dict, segment_start: int, segment_end: int, 
                                   progress_start: float, progress_end: float) -> Dict[str, Any]:
    """生成视频片段总结"""
    try:
        recent_vision = memory.get('vision', '')[-500:] if memory.get('vision') else ''
        recent_annotations = memory.get('stream_annotations', [])[-3:] if memory.get('stream_annotations') else []
        
        segment_info = {
            'time_range': f'{progress_start:.1f}%-{progress_end:.1f}%',
            'frame_range': f'{segment_start}-{segment_end}帧',
            'content': recent_vision[:200] if recent_vision else '暂无视觉内容',
            'risk_level': 'low',
            'key_findings': []
        }
        
        for ann in recent_annotations:
            if ann.get('content'):
                content = ann.get('content', '')[:100]
                if any(keyword in content.lower() for keyword in ['风险', 'risk', '违规', '异常']):
                    segment_info['risk_level'] = 'medium'
                    segment_info['key_findings'].append(content)
        
        if recent_vision:
            if '风险' in recent_vision or '违规' in recent_vision:
                segment_info['risk_level'] = 'high'
            summary = recent_vision.split('。')[0][:100] + '...' if len(recent_vision) > 100 else recent_vision
            segment_info['content'] = summary
        
        return segment_info
        
    except Exception as e:
        return {
            'time_range': f'{progress_start:.1f}%-{progress_end:.1f}%',
            'frame_range': f'{segment_start}-{segment_end}帧',
            'content': f'片段分析完成 (帧{segment_start}-{segment_end})',
            'risk_level': 'low',
            'key_findings': []
        }

async def _asr_transcribe(ds: DSClient, buf: 'SessionBuf') -> Dict[str, Any]:
    """Transcribe last audio chunk if present; returns {'text': str, 'audio': url_or_path}"""
    if not buf.audios:
        return {'text': '', 'audio': ''}
    last = buf.audios[-1]
    url = last if last.startswith(('http://','https://')) else buf.public_url(last)
    text = await _asr(ds, url)
    return {'text': text, 'audio': url}

async def _assess_accumulated_risk(stream_annotations: List[Dict[str, Any]]) -> Dict[str, Any]:
    """基于流式标注评估累积风险"""
    if not stream_annotations:
        return {'level': 'low', 'score': 0.1, 'reasoning': '无足够数据'}
    
    risk_indicators = []
    total_confidence = 0.0
    
    for annotation in stream_annotations:
        content = annotation.get('content', '')
        if annotation.get('type') == 'vision':
            if any(word in content for word in ['裸露', '暴力', '血腥', '武器']):
                risk_indicators.append(('high', 0.8, '视觉违规内容'))
            elif any(word in content for word in ['争议', '政治', '敏感']):
                risk_indicators.append(('medium', 0.6, '敏感视觉内容'))
        elif annotation.get('type') == 'audio':
            if any(word in content for word in ['骂人', '诈骗', '赌博', '威胁']):
                risk_indicators.append(('high', 0.9, '音频违规内容'))
            elif any(word in content for word in ['广告', '推销', '联系方式']):
                risk_indicators.append(('medium', 0.5, '商业导流内容'))
    
    if not risk_indicators:
        return {'level': 'low', 'score': 0.2, 'reasoning': '未发现明显风险'}
    
    high_count = sum(1 for r in risk_indicators if r[0] == 'high')
    medium_count = sum(1 for r in risk_indicators if r[0] == 'medium')
    
    if high_count > 0:
        level = 'high'
        score = min(0.95, 0.7 + high_count * 0.1)
    elif medium_count >= 2:
        level = 'medium'
        score = min(0.8, 0.4 + medium_count * 0.15)
    else:
        level = 'low'
        score = min(0.6, 0.2 + medium_count * 0.1)
    
    reasoning = f"发现{len(risk_indicators)}个风险指标: " + "; ".join([r[2] for r in risk_indicators[:3]])
    
    return {'level': level, 'score': score, 'reasoning': reasoning}

def _rules_retrieve(vision_text: str, asr_text: str) -> List[Dict[str, Any]]:
    """Heuristic rules retrieval based on keywords; returns list of rules."""
    rules: List[Dict[str, Any]] = []
    def hit(name: str, weight: float):
        rules.append({'name': name, 'weight': weight})
    vt = (vision_text or '').lower()
    at = (asr_text or '').lower()
    if any(k in vt for k in ['血', '暴力', '武器', 'knife', 'gun']):
        hit('暴力/血腥-图像', 0.9)
    if any(k in vt for k in ['裸', '敏感部位', '色情']):
        hit('低俗/擦边-图像', 0.8)
    if any(k in at for k in ['赌', '博彩', '下注', '诈骗', '收款', '加群']):
        hit('赌博/诈骗-语音', 1.0)
    if any(k in at for k in ['联系方式', 'vx', '威信', '加我']):
        hit('导流-联系方式-语音', 0.7)
    return rules[:6]

async def run_cot_react(ds: DSClient, buf: 'SessionBuf', ws: WebSocket) -> Dict[str, Any]:
    """Enhanced streaming CoT+ReAct: 边看视频边做数据标注边做人工审批"""
    memory: Dict[str, Any] = {
        'vision': '', 
        'asr': '', 
        'rules': [],
        'stream_annotations': [],  # 流式标注记录
        'approval_signals': [],    # 审批信号累积
        'watch_progress': 0        # 观看进度模拟
    }
    step_limit = int(getattr(_REASONING_CFG, 'step_limit', 15) or 15)
    tick_seconds = float(getattr(_REASONING_CFG, 'tick_seconds', 2.0) or 2.0)
    use_asr = bool(getattr(_REASONING_CFG, 'use_asr', False))
    
    await _send_trace(ws, 'assistant', '开始流式CoT+ReAct：边看边标注边审批', 'reasoning', {
        'step_limit': step_limit, 
        'streaming_mode': True,
        'tick_seconds': tick_seconds,
        'use_asr': use_asr
    }, 'start')
    
    segment_summaries = []
    
    if not getattr(buf, 'segment_analysis_cache', None):
        buf.segment_analysis_cache = {
            'completed_segments': [],
            'total_segments_expected': 0,
            'analysis_complete': False,
            'analysis_progress': 0,
            'analysis_started': False,
        }
    segment_analysis_cache = buf.segment_analysis_cache
    
    video_duration = float(buf.meta.get('duration', 0))  # 视频总时长（秒）
    segment_duration = 5.0  # 每个片段5秒
    total_segments = max(1, int(video_duration / segment_duration)) if video_duration > 0 else 6
    
    segment_analysis_cache['total_segments_expected'] = total_segments
    
    print(f" 视频片段分析配置: 总时长={video_duration:.1f}s, 片段时长={segment_duration}s, 总片段数={total_segments}")
    
    time_segments = []
    for i in range(total_segments):
        start_time = i * segment_duration
        end_time = min((i + 1) * segment_duration, video_duration)
        time_segments.append({
            'index': i + 1,
            'start_time': start_time,
            'end_time': end_time,
            'duration': end_time - start_time
        })
    
    current_segment_index = 0
    
    print(f" 片段分析等待帧数据准备，总共需要分析{len(time_segments)}个片段")
    
    for step in range(1, step_limit+1):
        print(f" Step {step}: buf.frames={len(buf.frames)}, buf.audios={len(buf.audios)}")
        
        try:
            if 'watch_progress' in buf.meta and isinstance(buf.meta.get('watch_progress'), (int, float)):
                memory['watch_progress'] = max(0.0, min(100.0, float(buf.meta.get('watch_progress') or 0.0)))
            else:
                total_frames_expected = 30.0
                frames_analyzed = float(len(buf.frames))
                memory['watch_progress'] = min(100.0, (frames_analyzed / total_frames_expected) * 100.0)
        except Exception:
            memory['watch_progress'] = 0.0
        observations = {
            'frames_cnt': len(buf.frames),
            'audio_cnt': len(buf.audios),
            'have_vision': bool(memory['vision']),
            'have_asr': bool(memory['asr']),
            'rules_cnt': len(memory['rules']),
            'watch_progress': memory['watch_progress'],
            'stream_annotations_cnt': len(memory['stream_annotations']),
            'approval_signals_cnt': len(memory['approval_signals'])
        }
        sys_prompt = (
            f'你是流式审核智能体，正在边看视频边标注边审批（当前观看进度: {memory["watch_progress"]:.0f}%）。\n'
            '采用CoT+ReAct流程：Thought → Action → Observation，可以并行处理多个任务。\n'
            '工具（可并行使用）: \n'
            '- vision_describe: 分析当前帧内容（实时视觉理解），args={}\n'
            '- asr_transcribe: 转录当前音频片段（实时音频理解），args={}\n'
            '- rules_retrieve: 检索相关审核规则，args={}\n'
            '- annotation.update: 实时更新标注（基于当前发现），args={{"category":"string","severity":"low|medium|high|ban","progress":0-100,"evidence":"string","timestamp":number}}\n'
            '- ui.highlight: 高亮关注区域，args={{"selector":"CSS选择器","reason":"高亮原因"}}\n'
            '- approval.suggest: 基于累积证据给出审批建议，args={{"recommendation":"approve|reject|review","confidence":0-100,"reasoning":"原因"}}\n'
            '流式策略：\n'
            '- 早期步骤：专注vision_describe和asr_transcribe收集信息\n'
            '- 中期步骤：开始annotation.update填充标注\n'
            '- 后期步骤：approval.suggest给出审批建议\n'
            '- 可以在同一步骤执行多个相关工具\n'
            '输出格式(JSON)：\n'
            '{{"type":"Thought","text":"基于当前分析阶段，我现在应该..."}} 或\n'
            '{{"type":"Action","tool":"工具名","args":{{...}}}} 或\n'
            '{{"type":"Final","result":{{"risk_level":"low|medium|high|ban","counters":{{"low":n,"medium":n,"high":n,"ban":n}},"summary":"基于流式分析的综合总结"}}}}\n'
            '仅输出一个JSON，思考简洁。'
        )
        user_ctx = {
            'observations': observations,
            'vision_excerpt': memory['vision'][-200:] if memory['vision'] else '',  # 最新视觉信息
            'asr_excerpt': memory['asr'][-200:] if memory['asr'] else '',          # 最新音频信息  
            'rules': memory['rules'][:4],
            'recent_annotations': memory['stream_annotations'][-2:],               # 最近标注
            'recent_approvals': memory['approval_signals'][-2:]                   # 最近审批信号
        }
        try:
            resp = await ds.qwen_text(sys_prompt + "\n上下文:" + json.dumps(user_ctx, ensure_ascii=False))
            out = resp.get('output') if isinstance(resp, dict) else {}
            action_obj: Dict[str, Any] = {}
            if isinstance(out, dict) and out:
                action_obj = out
            else:
                text = out.get('text') or resp.get('output_text') or json.dumps(out, ensure_ascii=False)
                try:
                    import re
                    m = re.search(r"\{[\s\S]*\}", text)
                    action_obj = json.loads(m.group(0)) if m else {}
                except Exception:
                    action_obj = {}
        except Exception as e:
            print(f" Step {step} 规划异常: {e}")
            await _send_trace(ws, 'system', f'规划失败: {e}', 'reasoning', None, 'error')
            break

        typ = (action_obj.get('type') or '').lower()
        print(f" Step {step} AI返回类型: {typ}, 内容: {str(action_obj)[:200]}")
        if typ == 'thought':
            thought_text = action_obj.get('text','')
            await _send_trace(ws, 'assistant', f' [{memory["watch_progress"]:.0f}%] {thought_text}', 'reasoning', {
                'step': step, 
                'progress': memory['watch_progress'],
                'streaming_mode': True
            }, 'thought')
            continue
        if typ == 'action':
            tool = (action_obj.get('tool') or '').lower()
            args = action_obj.get('args') or {}
            await _send_trace(ws, 'assistant', f'[{memory["watch_progress"]:.0f}%] 执行: {tool}', 'reasoning', {
                'step': step, 
                'tool': tool,
                'args': args,
                'progress': memory['watch_progress']
            }, 'action')
            if tool == 'vision_describe':
                obs = await _vision_describe(ds, buf)
                vision_result = obs.get('text','')
                memory['vision'] = (memory['vision'] + '\n' + vision_result).strip()
                
                risk_segments = obs.get('risk_segments', [])
                segments_info = obs.get('segments', [])
                
                memory['stream_annotations'].append({
                    'type': 'vision',
                    'progress': memory['watch_progress'], 
                    'content': vision_result,
                    'timestamp': step,
                    'segments': segments_info,
                    'risk_segments': risk_segments,
                    'window_analyzed': f"{obs.get('window_start', 0):.1f}s-{obs.get('window_end', 0):.1f}s"
                })
                
                observation_msg = f'视觉发现: {vision_result[:200]}'
                if risk_segments:
                    risk_count = len(risk_segments)
                    high_risk = len([r for r in risk_segments if r.get('risk_level') in ['high', 'ban']])
                    observation_msg += f' | 发现{risk_count}个风险片段({high_risk}个高风险)'
                
                await _send_trace(ws, 'assistant', observation_msg, 'reasoning', {
                    'images': obs.get('images',[]),
                    'progress': memory['watch_progress'],
                    'risk_segments': risk_segments,
                    'total_frames_analyzed': obs.get('total_frames_analyzed', 0)
                }, 'observation')
            elif tool == 'asr_transcribe' and use_asr:
                obs = await _asr_transcribe(ds, buf)
                asr_result = obs.get('text','')
                memory['asr'] = (memory['asr'] + '\n' + asr_result).strip()
                memory['stream_annotations'].append({
                    'type': 'audio',
                    'progress': memory['watch_progress'],
                    'content': asr_result,
                    'timestamp': step
                })
                await _send_trace(ws, 'assistant', f' 音频内容: {asr_result[:200]}', 'reasoning', {
                    'audio': obs.get('audio',''),
                    'progress': memory['watch_progress']
                }, 'observation')
            elif tool == 'rules_retrieve':
                rules = _rules_retrieve(memory['vision'], memory['asr'])
                memory['rules'] = rules
                await _send_trace(ws, 'assistant', f" 检索到{len(rules)}条潜在规则", 'reasoning', {
                    'rules': rules,
                    'progress': memory['watch_progress']
                }, 'observation')
            elif tool == 'annotation.update':
                enhanced_args = {
                    **args,
                    'streaming_progress': memory['watch_progress'],
                    'evidence_count': len(memory['stream_annotations']),
                    'timestamp': step
                }
                memory['stream_annotations'].append({
                    'type': 'annotation',
                    'progress': memory['watch_progress'],
                    'data': enhanced_args,
                    'timestamp': step
                })
                await _send_tool(ws, 'annotation.update', enhanced_args)
                await _send_trace(ws, 'assistant', f'标注更新: 基于{memory["watch_progress"]:.0f}%进度', 'reasoning', {
                    'fields': list(args.keys()),
                    'progress': memory['watch_progress']
                }, 'observation')
            elif tool == 'ui.highlight':
                enhanced_args = {
                    **args,
                    'progress': memory['watch_progress'],
                    'context': f'{memory["watch_progress"]:.0f}%观看进度'
                }
                await _send_tool(ws, 'ui.highlight', enhanced_args)
                await _send_trace(ws, 'assistant', f'界面高亮: {args.get("selector","")}', 'reasoning', {
                    'target': args.get('selector',''),
                    'progress': memory['watch_progress']
                }, 'observation')
            elif tool == 'approval.suggest':
                enhanced_args = {
                    **args,
                    'evidence_count': len(memory['stream_annotations']),
                    'watch_progress': memory['watch_progress'],
                    'vision_analyzed': bool(memory['vision']),
                    'audio_analyzed': bool(memory['asr']),
                    'rules_checked': len(memory['rules']) > 0
                }
                memory['approval_signals'].append({
                    'progress': memory['watch_progress'],
                    'recommendation': args.get('recommendation',''),
                    'confidence': args.get('confidence', 0),
                    'reasoning': args.get('reasoning',''),
                    'timestamp': step
                })
                await _send_tool(ws, 'approval.suggest', enhanced_args)
                await _send_trace(ws, 'assistant', f'审批建议: {args.get("recommendation","")} (置信度: {args.get("confidence",0)}%)', 'reasoning', {
                    'recommendation': enhanced_args,
                    'progress': memory['watch_progress']
                }, 'observation')
            else:
                await _send_trace(ws, 'assistant', f'未知工具：{tool}', 'reasoning', {'tool': tool}, 'error')
            
            await asyncio.sleep(tick_seconds)
            continue
        if typ == 'final':
            print(f" Step {step} AI提前返回final结果: {action_obj}")
            res = action_obj.get('result') or {}
            if {'risk_level','counters','summary'} <= set(res.keys()):
                print(f" Final结果完整，直接返回")
                return res
            print(f" Final结果不完整，fallback到默认判定")
            break
        print(f" Step {step} 检查强制执行: vision={bool(memory['vision'])}, frames={len(buf.frames)}")
        if not memory['vision'] and len(buf.frames) > 0:
            await _send_trace(ws, 'assistant', f'[步骤{step}] 强制执行: vision_describe (LLM未主动调用)', 'reasoning', {
                'step': step, 
                'tool': 'vision_describe',
                'args': {},
                'progress': memory['watch_progress']
            }, 'action')
            
            print(f" Step {step} 开始强制执行vision_describe")
            obs = await _vision_describe(ds, buf)
            vision_result = obs.get('text','')
            memory['vision'] = (memory['vision'] + '\n' + vision_result).strip()
            print(f" Step {step} 强制执行vision_describe完成: {len(vision_result)}字符")
            
            risk_segments = obs.get('risk_segments', [])
            segments_info = obs.get('segments', [])
            
            memory['stream_annotations'].append({
                'type': 'vision',
                'progress': memory['watch_progress'], 
                'content': vision_result,
                'timestamp': step,
                'segments': segments_info,
                'risk_segments': risk_segments,
                'window_analyzed': f"{obs.get('window_start', 0):.1f}s-{obs.get('window_end', 0):.1f}s"
            })
            
            observation_msg = f' 视觉发现: {vision_result[:200]}'
            if risk_segments:
                risk_count = len(risk_segments)
                high_risk = len([r for r in risk_segments if r.get('risk_level') in ['high', 'ban']])
                observation_msg += f' |  发现{risk_count}个风险片段({high_risk}个高风险)'
            
            await _send_trace(ws, 'assistant', observation_msg, 'reasoning', {
                'images': obs.get('images',[]),
                'progress': memory['watch_progress'],
                'risk_segments': risk_segments,
                'total_frames_analyzed': obs.get('total_frames_analyzed', 0)
            }, 'observation')
            
            min_frames_needed = 1
            analysis_started = segment_analysis_cache.get('analysis_started', False)
            frames_count = len(buf.frames)
            
            print(f" 阈值检查: frames={frames_count}, 阈值={min_frames_needed}, 已启动={analysis_started}, 总片段={total_segments}")
            
            if (not getattr(buf, 'segment_analysis_started', False)) and (not analysis_started) and frames_count >= min_frames_needed:
                print(f" 帧数据准备充足({frames_count}帧，阈值{min_frames_needed})，现在启动智能并行片段分析（API协调器管理）")
                segment_analysis_cache['analysis_started'] = True
                buf.segment_analysis_started = True
                if not getattr(buf, 'segment_analysis_task', None) or buf.segment_analysis_task.done():
                    buf.segment_analysis_task = asyncio.create_task(
                        _preload_segment_analysis(time_segments, video_duration, buf, ws, segment_analysis_cache)
                    )
            
            if current_segment_index < len(time_segments):
                current_segment = time_segments[current_segment_index]
                segment_progress_threshold = (current_segment['end_time'] / video_duration) * 100 if video_duration > 0 else (current_segment_index + 1) * (100 / total_segments)
                
                if memory['watch_progress'] >= segment_progress_threshold - 5:  # 留5%缓冲
                    completed_segments = len(segment_analysis_cache.get('completed_segments', []))
                    await _send_trace(ws, 'system', 
                        f' 当前播放到片段{current_segment_index + 1}，后台分析进度: {completed_segments}/{total_segments}', 
                        'playback_progress', {
                            'current_playback_segment': current_segment_index + 1,
                            'analysis_completed': completed_segments,
                            'total_segments': total_segments
                        })
                    
                    current_segment_index += 1
            
            continue
        if getattr(_REASONING_CFG, 'use_asr', False) and (not memory['asr']) and len(buf.audios) > 0:
            obs = await _asr_transcribe(ds, buf)
            memory['asr'] = obs.get('text','')
            await _send_trace(ws, 'assistant', obs.get('text','')[:300], 'reasoning', {'audio': obs.get('audio','')}, 'observation')
            continue
        
        print(f" 循环检查: use_asr={getattr(_REASONING_CFG, 'use_asr', False)}, has_asr={bool(memory['asr'])}, audio_count={len(buf.audios)}")
        
        analysis_started = segment_analysis_cache.get('analysis_started', False)
        frames_available = len(buf.frames)
        
        if analysis_started or getattr(buf, 'segment_analysis_started', False):
            print(f" 并行分析已启动，可以结束循环")
            break
        elif step >= 5 and frames_available == 0:
            print(f" 等待5步仍无帧数据，结束循环")
            break
        elif step < step_limit:
            print(f" 继续等待帧数据或触发并行分析 (step {step}/{step_limit})")
            continue
        else:
            print(f" 达到步数限制，结束循环")
            break
        
    if not (segment_analysis_cache.get('analysis_started', False) or getattr(buf, 'segment_analysis_started', False)):
        print(f" 并行分析未启动，回退到逐个片段分析模式")
        while current_segment_index < len(time_segments):
            remaining_segment = time_segments[current_segment_index]
            
            segment_summary = await _generate_time_based_segment_summary(
                memory, remaining_segment, video_duration, buf
            )
            
            if segment_summary:
                segment_summaries.append(segment_summary)
                await _send_trace(ws, 'assistant', 
                    f' 最终片段总结 ({remaining_segment["start_time"]:.1f}s-{remaining_segment["end_time"]:.1f}s): {segment_summary["content"][:100]}...', 
                    'segment_summary', segment_summary)
            
            current_segment_index += 1
    else:
        print(f" 并行分析已启动，跳过逐个片段分析，等待并行结果")
    
    print(f" 完成所有片段分析，总共生成 {len(segment_summaries)} 个时间段总结")

    risk_assessment = await _assess_accumulated_risk(memory['stream_annotations'])
    
    context = {
        'streaming_analysis': {
            'total_annotations': len(memory['stream_annotations']),
            'approval_signals': len(memory['approval_signals']),
            'risk_assessment': risk_assessment,
            'segment_summaries': segment_summaries  # 添加片段总结
        },
        'content_analysis': {
            'vision': memory['vision'][-500:] if memory['vision'] else '',  # 最新视觉分析
            'transcript': memory['asr'][-500:] if memory['asr'] else '',    # 最新音频转录
            'rules_triggered': memory['rules'][:5]                          # 触发的规则
        },
        'evidence_timeline': [
            {
                'progress': ann.get('progress', 0),
                'type': ann.get('type', ''),
                'content': ann.get('content', '')[:100]  # 截取关键内容
            } for ann in memory['stream_annotations'][-5:]  # 最近5个发现
        ]
    }
    
    judge_prompt = (
        '你是流式内容审核智能体，基于边看边标注边审批的完整分析过程，给出最终审核结论。\n'
        '重点考虑：\n'
        '1. 分析完整性和质量\n'
        '2. 时间线上的风险发现\n' 
        '3. 累积证据的一致性\n'
        '4. 审批信号的置信度\n'
        '输出严格JSON格式：\n'
        '{"risk_level": "low|medium|high|ban", "counters": {"low": n, "medium": n, "high": n, "ban": n}, '
        '"summary": "基于流式分析的综合总结，聚焦内容分析结果、风险识别和最终判定依据，不要提及观看覆盖度相关信息"}\n'
        f"流式分析上下文: {json.dumps(context, ensure_ascii=False)}"
    )
    await _send_trace(ws, 'system', '⏳ 等待所有片段分析完成...', 'waiting_analysis')
    
    max_wait_time = 120  # 最大等待2分钟
    wait_time = 0
    check_interval = 2  # 每2秒检查一次
    
    while not segment_analysis_cache['analysis_complete'] and wait_time < max_wait_time:
        await asyncio.sleep(check_interval)
        wait_time += check_interval
        
        progress = segment_analysis_cache.get('analysis_progress', 0)
        completed = len(segment_analysis_cache.get('completed_segments', []))
        total = segment_analysis_cache.get('total_segments_expected', 0)
        
        await _send_trace(ws, 'system', 
            f'⏳ 等待片段分析: {completed}/{total} ({progress:.1f}%) - 已等待{wait_time}s', 
            'waiting_progress', {
                'completed': completed,
                'total': total,
                'progress': progress,
                'wait_time': wait_time
            })
        
        print(f"⏳ 等待片段分析完成: {completed}/{total} ({progress:.1f}%)")
    
    if segment_analysis_cache['analysis_complete']:
        segment_summaries = segment_analysis_cache['completed_segments']
        print(f" 片段分析完成，获得 {len(segment_summaries)} 个片段总结")
        print(f" 审核数据已全部到达！共计{len(segment_summaries)}个片段的分析结果已存储完毕")
    else:
        print(f" 片段分析超时，已等待{wait_time}s，使用现有结果")
        segment_summaries = segment_analysis_cache.get('completed_segments', [])
    
    print(f" 最终片段分析统计: {len(segment_summaries)} 个时间段总结")
    
    try:
        await _send_trace(ws, 'assistant', '合成最终结论…', 'judge')
        judge = await ds.qwen_text(judge_prompt)
        output = judge.get('output') or {}
        parsed: Dict[str, Any] = {}
        if isinstance(output, dict) and {'risk_level','counters','summary'} <= set(output.keys()):
            parsed = output
        else:
            txt = output.get('text') or judge.get('output_text') or json.dumps(output, ensure_ascii=False)
            import re
            m = re.search(r"\{[\s\S]*\}", txt)
            if m:
                parsed = json.loads(m.group(0))
        if parsed:
            parsed['segment_summaries'] = segment_summaries
            return parsed
    except Exception as e:
        await _send_trace(ws, 'system', f'合成失败: {e}', 'judge', None, 'error')
    return {
        'risk_level':'low',
        'counters':{'low':1,'medium':0,'high':0,'ban':0},
        'summary':'回退：样本不足，暂判低风险。',
        'segment_summaries': segment_summaries
    }

@app.post("/agent/audit")
async def agent_audit(req: AuditRequest):
    task_id = str(uuid.uuid4())
    result: Dict[str, Any] = {"risk_level":"medium","counters":{"low":1,"medium":0,"high":0,"ban":0},"summary":"回退：服务暂不可用。"}

    if DSClient is None:
        return JSONResponse({"task_id": task_id, "status": "done", "result": result})

    try:
        ds = DSClient()
        frames = _extract_frames(req.video_url) if req.video_url else []
        transcript = await _asr(ds, req.audio_url)
        vl_tags = {}
        context = {"aweme_id": req.aweme_id, "title": req.title, "desc": req.desc, "transcript": transcript, "vision": vl_tags}
        prompt = (
            "你是短视频内容审核助手。给定结构化上下文(中文)，请输出严格JSON: "
            "{risk_level: one of [low,medium,high,ban], counters:{low,medium,high,ban}, summary:string}.\n"
            f"上下文: {json.dumps(context, ensure_ascii=False)}"
        )
        judge = await ds.qwen_text(prompt)
        output = judge.get("output") or {}
        if isinstance(output, dict) and {"risk_level","counters","summary"} <= set(output.keys()):
            parsed = output
        else:
            text = output.get("text") or judge.get("output_text") or json.dumps(output, ensure_ascii=False)
            try:
                import re
                m = re.search(r"\{[\s\S]*\}", text)
                parsed = json.loads(m.group(0)) if m else {}
            except Exception:
                parsed = {}
        if parsed:
            result = {
                "risk_level": parsed.get("risk_level", result["risk_level"]),
                "counters": parsed.get("counters", result["counters"]),
                "summary": parsed.get("summary", result["summary"]),
            }
    except Exception as e:
        result = {**result, "error": str(e)}

    return JSONResponse({"task_id": task_id, "status": "done", "result": result})


async def _analyze_comment_content(ds: 'DSClient', comment_text: str) -> Dict[str, Any]:
    """使用qwen2.5-flash分析单条评论内容的风险"""
    
    enhanced_prompt = f"""
分析以下用户评论的内容安全性：

评论内容："{comment_text}"

请从以下维度进行分析：
1. **违规类型**：色情低俗、暴力血腥、政治敏感、虚假信息、人身攻击、违法广告、其他
2. **风险等级**：无风险/低风险/中风险/高风险/禁止发布
3. **具体原因**：详细说明判定依据

要求严格输出JSON格式：
{{"risk_level": "low|medium|high|ban", "violation_type": "类型或null", "reason": "详细分析原因", "confidence": 0.95}}
"""
    
    try:
        response = await ds.qwen_text(enhanced_prompt)
        logger.info(f" qwen2.5-flash评论分析响应: {type(response)}")
        
        output = response.get('output') if isinstance(response, dict) else {}
        
        if isinstance(output, dict) and {'risk_level', 'reason'} <= set(output.keys()):
            result = output
        else:
            text = output.get('text') or response.get('output_text') or json.dumps(output, ensure_ascii=False)
            try:
                import re
                m = re.search(r'\{[\s\S]*\}', text)
                result = json.loads(m.group(0)) if m else {}
            except Exception:
                result = {}
        
        risk_level = result.get('risk_level', 'low')
        violation_type = result.get('violation_type')
        reason = result.get('reason', '无明显违规内容')
        confidence = float(result.get('confidence', 0.8))
        
        logger.info(f" 评论分析完成: {risk_level} (置信度: {confidence:.2f})")
        
        return {
            'risk_level': risk_level,
            'violation_type': violation_type,
            'reason': reason,
            'confidence': confidence
        }
        
    except Exception as e:
        logger.error(f" 评论分析失败: {e}")
        return {
            'risk_level': 'low',
            'violation_type': None,
            'reason': f'分析失败: {str(e)}',
            'confidence': 0.0
        }


@app.post("/agent/analyze_comments")
async def analyze_comments(req: CommentAnalysisRequest):
    """分析评论内容的违规风险"""
    task_id = str(uuid.uuid4())
    
    if DSClient is None:
        return JSONResponse({
            "task_id": task_id, 
            "status": "error", 
            "message": "AI服务不可用",
            "results": []
        })
    
    if not req.comments:
        return JSONResponse({
            "task_id": task_id, 
            "status": "done", 
            "results": [],
            "counters": {"low": 0, "medium": 0, "high": 0, "ban": 0}
        })
    
    try:
        ds = DSClient()
        results = []
        counters = {"low": 0, "medium": 0, "high": 0, "ban": 0}
        
        logger.info(f" 开始分析 {len(req.comments)} 条评论")
        
        for i, comment in enumerate(req.comments):
            comment_text = comment.get('detail', comment.get('content', comment.get('text', '')))
            comment_id = comment.get('comment_id', comment.get('id', f'comment_{i}'))
            
            if not comment_text or len(comment_text.strip()) == 0:
                continue
                
            logger.info(f" 分析评论 {i+1}/{len(req.comments)}: {comment_text[:50]}...")
            
            analysis = await _analyze_comment_content(ds, comment_text)
            
            risk_level = analysis['risk_level']
            if risk_level in counters:
                counters[risk_level] += 1
            else:
                counters['low'] += 1  # 默认归为低风险
            
            results.append({
                'comment_id': comment_id,
                'comment_text': comment_text,
                'analysis': analysis,
                'timestamp': time.time()
            })
            
            if i < len(req.comments) - 1:
                await asyncio.sleep(0.1)
        
        logger.info(f" 评论分析完成: {counters}")
        
        return JSONResponse({
            "task_id": task_id,
            "status": "done", 
            "aweme_id": req.aweme_id,
            "results": results,
            "counters": counters,
            "summary": f"已分析 {len(results)} 条评论，发现 {counters['high'] + counters['ban']} 条高风险内容"
        })
        
    except Exception as e:
        logger.error(f" 评论批量分析失败: {e}")
        return JSONResponse({
            "task_id": task_id,
            "status": "error",
            "message": str(e),
            "results": [],
            "counters": {"low": 0, "medium": 0, "high": 0, "ban": 0}
        })

@app.get("/agent/get_cached_result/{session_id}")
async def get_cached_result(session_id: str):
    """获取缓存的分析结果，当WebSocket连接断开时使用"""
    try:
        if session_id in sessions:
            buf = sessions[session_id]
            if buf.final_result:
                print(f" 返回缓存的分析结果，会话: {session_id}")
                return {"success": True, "data": buf.final_result}
            else:
                print(f" 会话 {session_id} 暂无缓存结果")
                return {"success": False, "message": "分析尚未完成或结果不可用"}
        else:
            print(f" 会话 {session_id} 不存在")
            return {"success": False, "message": "会话不存在"}
    except Exception as e:
        print(f" 获取缓存结果失败: {e}")
        return {"success": False, "message": str(e)}


class SessionBuf:
    def __init__(self, sid: str):
        self.sid: str = sid
        self.meta: Dict[str, Any] = {}
        self.frames: List[str] = []  # absolute file paths
        self.audios: List[str] = []  # absolute file paths
        self.last_emit: float = 0.0
        self.dir_path: str = os.path.join(STATIC_ROOT, sid)
        os.makedirs(self.dir_path, exist_ok=True)
        self.total_frames: int = 0
        self.total_audios: int = 0
        self.auto_extract_started: bool = False
        self.auto_extract_done: bool = False
        self.segment_analysis_started: bool = False
        self.segment_analysis_completed: bool = False
        self.segment_analysis_cache: Optional[Dict[str, Any]] = None
        self.segment_analysis_task: Optional[asyncio.Task] = None
        self.cot_react_completed: bool = False
        self.final_result: Optional[Dict[str, Any]] = None

    def _rel_url(self, abs_path: str) -> str:
        rel = os.path.relpath(abs_path, STATIC_ROOT).replace(os.sep, "/")
        return f"/static/{rel}"

    def public_url(self, abs_path: str) -> str:
        rel_url = self._rel_url(abs_path)
        if PUBLIC_BASE_URL:
            return f"{PUBLIC_BASE_URL}{rel_url}"
        return f"http://127.0.0.1:8799{rel_url}"


def _ext_from_mime(mime: str) -> str:
    if "jpeg" in mime:
        return "jpg"
    if "png" in mime:
        return "png"
    if "webm" in mime:
        return "webm"
    if "ogg" in mime:
        return "ogg"
    if "mp4" in mime:
        return "mp4"
    return "bin"


def _save_data_url(data_url: str, dir_path: str, prefix: str) -> Optional[str]:
    try:
        if not data_url or "," not in data_url:
            return None
        header, b64 = data_url.split(",", 1)
        mime = ""
        if header.startswith("data:") and ";base64" in header:
            mime = header[5: header.find(";")]
        ext = _ext_from_mime(mime)
        ts = int(time.time() * 1000)
        filename = f"{prefix}_{ts}.{ext}"
        abs_path = os.path.join(dir_path, filename)
        with open(abs_path, "wb") as f:
            f.write(base64.b64decode(b64))
        return abs_path
    except Exception:
        return None

sessions = {}

from typing import Any
import json, time

async def _auto_extract_frames_for_session(buf: 'SessionBuf', ws: WebSocket):
    """在收到meta后端自动提取30帧并注入到当前会话缓冲区，避免依赖前端逐帧推送。"""
    if buf.auto_extract_started or buf.auto_extract_done:
        return
    video_url = (buf.meta or {}).get('src') or ''
    if not video_url:
        return
    buf.auto_extract_started = True
    try:
        try:
            await ws.send_text(json.dumps({'type': 'trace', 'data': {
                'role': 'system',
                'text': f' 后端自动提取帧启动: 目标30帧',
                'stage': 'auto_extract_start'
            }}, ensure_ascii=False))
        except Exception:
            pass

        frame_paths = await asyncio.to_thread(_extract_frames, video_url, 0, 30)
        if not frame_paths:
            return

        injected = 0
        for path in frame_paths:
            try:
                filename = os.path.basename(path)
                static_path = os.path.join(STATIC_ROOT, 'extracted_frames', filename)
                os.makedirs(os.path.dirname(static_path), exist_ok=True)
                if os.path.exists(path):
                    import shutil
                    shutil.move(path, static_path)
                rel_url = f"/static/extracted_frames/{filename}"
                public_url = f"{PUBLIC_BASE_URL}{rel_url}" if PUBLIC_BASE_URL else f"http://127.0.0.1:8799{rel_url}"
                if public_url not in buf.frames:
                    buf.frames.append(public_url)
                    if len(buf.frames) > 30:
                        buf.frames = buf.frames[-30:]
                    buf.total_frames += 1
                    injected += 1
                    logger.info(f"frame_url sid={buf.sid} url={public_url} count_batch={len(buf.frames)} total={buf.total_frames}")
            except Exception:
                continue

        buf.auto_extract_done = True
        try:
            await ws.send_text(json.dumps({'type': 'trace', 'data': {
                'role': 'system',
                'text': f' 后端自动注入帧完成: {injected}帧',
                'stage': 'auto_extract_done',
                'payload': {'injected': injected}
            }}, ensure_ascii=False))
        except Exception:
            pass
    except Exception:
        buf.auto_extract_done = True
        try:
            await ws.send_text(json.dumps({'type': 'trace', 'data': {
                'role': 'system',
                'text': ' 后端自动提取帧失败',
                'stage': 'auto_extract_error'
            }}, ensure_ascii=False))
        except Exception:
            pass

@app.websocket("/ws/stream")
async def ws_stream(ws: WebSocket):
    await ws.accept()
    
    print(f" WebSocket连接已建立，无心跳超时限制")
    
    sid = str(uuid.uuid4())
    buf = SessionBuf(sid)
    sessions[sid] = buf
    try:
        ds = DSClient() if DSClient else None
        ds_status = "可用" if ds else "不可用"
        logger.info(f"ws connected sid={sid}, PUBLIC_BASE_URL={'set' if PUBLIC_BASE_URL else 'EMPTY'}, DSClient={ds_status}")
        
        try:
            await ws.send_text(json.dumps({'type':'trace','data':{'role':'system','text':f'AI服务状态: DSClient={ds_status}, 公网URL={"已配置" if PUBLIC_BASE_URL else "未配置"}','stage':'init','payload':{'ds_available': bool(ds), 'public_url_set': bool(PUBLIC_BASE_URL)},'ts':int(time.time() * 1000)}}, ensure_ascii=False))
        except Exception:
            pass
        while True:
            msg = await ws.receive_text()
            try:
                data = json.loads(msg)
            except Exception:
                continue
            mtype = data.get('type')
            if mtype == 'meta':
                payload = data.get('data') if isinstance(data.get('data'), dict) else data
                
                print(f"DEBUG: 收到元信息原始数据: {json.dumps(payload, ensure_ascii=False)}")
                
                buf.meta.update({
                    'aweme_id': payload.get('aweme_id') or buf.meta.get('aweme_id'),
                    'title': payload.get('title') or buf.meta.get('title'),
                    'desc': payload.get('desc') or buf.meta.get('desc'),
                    'duration': payload.get('duration', 0),
                    'width': payload.get('width', 0),
                    'height': payload.get('height', 0),
                    'src': payload.get('src', ''),
                    'timestamp': payload.get('timestamp', int(time.time() * 1000))
                })
                
                duration_sec = float(buf.meta.get('duration', 0))
                duration_str = f"{duration_sec:.1f}秒" if duration_sec > 0 else "未知"
                resolution_str = f"{buf.meta.get('width', 0)}x{buf.meta.get('height', 0)}"
                
                try:
                    await ws.send_text(json.dumps({
                        'type': 'trace',
                        'data': {
                            'role': 'system',
                            'text': f"收到完整元信息 aweme_id={buf.meta.get('aweme_id')}, 标题={buf.meta.get('title') or ''}, 时长={duration_str}, 分辨率={resolution_str}",
                            'ts': int(time.time() * 1000)
                        }
                    }, ensure_ascii=False))
                except Exception:
                    pass
                logger.info(f"meta sid={sid} aweme_id={buf.meta.get('aweme_id')} title={buf.meta.get('title')} duration={duration_sec}s resolution={resolution_str}")
                if not buf.auto_extract_started and duration_sec > 0 and buf.meta.get('src'):
                    asyncio.create_task(_auto_extract_frames_for_session(buf, ws))
                
                if not buf.cot_react_completed and duration_sec > 0:
                    print(f" 启动智能体分析会话: {sid}")
                    buf.cot_react_completed = True  # 标记为已启动，避免重复
                    
                    async def run_analysis():
                        try:
                            result = await run_cot_react(ds, buf, ws)
                            buf.final_result = result
                            
                            result_data = json.dumps({'type': 'result', 'data': result}, ensure_ascii=False)
                            print(f" 准备发送最终结果: risk_level={result.get('risk_level', 'unknown')}, segment_count={len(result.get('segment_summaries', []))}")
                            
                            try:
                                max_retries = 3
                                for attempt in range(max_retries):
                                    try:
                                        if hasattr(ws, 'client_state') and ws.client_state.value == 1:  # WebSocketState.CONNECTED = 1
                                            await ws.send_text(result_data)
                                            print(f" 智能体分析完成，最终结果已发送，会话: {sid}")
                                            break
                                        else:
                                            print(f" WebSocket已断开(状态: {getattr(ws, 'client_state', 'unknown')})，尝试重新连接...")
                                            if attempt < max_retries - 1:
                                                await asyncio.sleep(1)  # 等待1秒后重试
                                                continue
                                            else:
                                                print(f" WebSocket连接失败，将结果存储到缓存等待前端重新获取")
                                                buf.final_result = result
                                    except Exception as retry_e:
                                        print(f" 发送尝试 {attempt + 1} 失败: {retry_e}")
                                        if attempt < max_retries - 1:
                                            await asyncio.sleep(1)
                                        else:
                                            print(f" 所有发送尝试失败，结果已保存到缓存")
                                            buf.final_result = result
                            except Exception as send_e:
                                print(f" 发送最终结果失败: {send_e}")
                                buf.final_result = result
                        except Exception as e:
                            print(f" 智能体分析失败: {e}")
                            error_result = {"error": str(e), "risk_level": "unknown"}
                            
                            try:
                                if hasattr(ws, 'client_state') and ws.client_state.value == 1:
                                    await ws.send_text(json.dumps({'type': 'result', 'data': error_result}, ensure_ascii=False))
                                else:
                                    print(f" WebSocket已断开，无法发送错误结果")
                            except Exception as send_error:
                                print(f" 发送错误结果失败: {send_error}")
                    
                    asyncio.create_task(run_analysis())
            elif mtype == 'frame':
                durl = data.get('data', '')
                p = _save_data_url(durl, buf.dir_path, 'frame')
                if p:
                    buf.frames.append(p)
                    if len(buf.frames) > 30:
                        buf.frames = buf.frames[-30:]
                    buf.total_frames += 1
                    if buf.total_frames % 10 == 1:
                        logger.info(f"frames sid={sid} last={p} count_batch={len(buf.frames)} total={buf.total_frames}")
            elif mtype == 'frame_url':
                frame_url = data.get('data', '')
                if frame_url:
                    buf.frames.append(frame_url)
                    if len(buf.frames) > 30:
                        buf.frames = buf.frames[-30:]
                    buf.total_frames += 1
                    logger.info(f"frame_url sid={sid} url={frame_url} count_batch={len(buf.frames)} total={buf.total_frames}")
                    try:
                        await ws.send_text(json.dumps({
                            'type': 'trace',
                            'data': {
                                'role': 'system',
                                'text': f" 接收后端提取帧: {os.path.basename(frame_url)}",
                                'ts': int(time.time() * 1000)
                            }
                        }, ensure_ascii=False))
                    except Exception:
                        pass
            elif mtype == 'audio':
                durl = data.get('data', '')
                p = _save_data_url(durl, buf.dir_path, 'audio')
                if p:
                    buf.audios.append(p)
                    if len(buf.audios) > 6:
                        buf.audios = buf.audios[-6:]
                    buf.total_audios += 1
                    if buf.total_audios % 5 == 1:
                        logger.info(f"audios sid={sid} last={p} count_batch={len(buf.audios)} total={buf.total_audios}")
            elif mtype == 'progress':
                payload = data.get('data') if isinstance(data.get('data'), dict) else data
                try:
                    current_time = float(payload.get('current_time') or payload.get('currentTime') or 0.0)
                except Exception:
                    current_time = 0.0
                try:
                    playback_rate = float(payload.get('playback_rate') or payload.get('playbackRate') or 1.0)
                except Exception:
                    playback_rate = 1.0
                try:
                    duration_sec = float(payload.get('duration') or buf.meta.get('duration') or 0.0)
                except Exception:
                    duration_sec = 0.0

                progress_pct = 0.0
                if duration_sec > 0:
                    progress_pct = max(0.0, min(100.0, (current_time * playback_rate) / duration_sec * 100.0))

                buf.meta['current_time'] = current_time
                buf.meta['playback_rate'] = playback_rate
                buf.meta['watch_progress'] = progress_pct

                logger.info(
                    f"progress sid={sid} current_time={current_time:.2f}s rate={playback_rate:.2f} "
                    f"duration={duration_sec:.2f}s progress={progress_pct:.1f}%"
                )
                try:
                    await ws.send_text(json.dumps({
                        'type': 'trace',
                        'data': {
                            'role': 'system',
                            'text': f" 播放进度上报: {progress_pct:.1f}% (t={current_time:.2f}s, x{playback_rate:.2f})",
                            'stage': 'progress',
                            'payload': {
                                'current_time': current_time,
                                'playback_rate': playback_rate,
                                'duration': duration_sec,
                                'progress': progress_pct,
                            },
                            'ts': int(time.time() * 1000)
                        }
                    }, ensure_ascii=False))
                except Exception:
                    pass

            await asyncio.sleep(10.0)  # 增加到10秒间隔，减少资源消耗
    except WebSocketDisconnect:
        pass
    finally:
        sessions.pop(sid, None)
        print(f" WebSocket连接已关闭，会话清理完成: {sid}")
