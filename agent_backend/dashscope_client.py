import os, httpx, asyncio
from typing import List, Dict, Any
from itertools import cycle
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

_BASE = "https://dashscope.aliyuncs.com/api/v1"

class KeyRotator:
    def __init__(self):
        raw = os.getenv("DASHSCOPE_API_KEYS") or os.getenv("DASHSCOPE_API_KEY", "")
        keys = [k.strip() for k in raw.split(',') if k.strip()]
        if not keys:
            raise RuntimeError("DASHSCOPE_API_KEY(S) not set")
        self._cycle = cycle(keys)
        self._last = None
    def next(self):
        self._last = next(self._cycle)
        return self._last

class DSClient:
    def __init__(self, text_model: str|None=None, vl_model: str|None=None, asr_model: str|None=None):
        self.text_model = text_model or os.getenv('DASHSCOPE_TEXT_MODEL', 'qwq-plus-latest')
        self.vl_model = vl_model or os.getenv('DASHSCOPE_VL_MODEL', 'qwen-vl-plus')
        self.asr_model = asr_model or os.getenv('DASHSCOPE_ASR_MODEL', 'qwen-audio-asr')
        
        print(f" DSClient模型配置:")
        print(f"   文本模型: {self.text_model}")
        print(f"   视觉模型: {self.vl_model}")
        print(f"   语音模型: {self.asr_model}")
        self.temperature = 0.3
        self.top_p = 0.9
        try:
            if os.getenv('DASHSCOPE_TEMPERATURE'):
                self.temperature = float(os.getenv('DASHSCOPE_TEMPERATURE'))
        except Exception:
            pass
        try:
            if os.getenv('DASHSCOPE_TOP_P'):
                self.top_p = float(os.getenv('DASHSCOPE_TOP_P'))
        except Exception:
            pass
        self._rot = KeyRotator()

    def _headers(self, key: str):
        return {"Content-Type":"application/json","Authorization":f"Bearer {key}"}

    async def _post_with_rotate(self, url: str, json: dict, timeout: float=120) -> Dict[str, Any]:
        import logging
        logger = logging.getLogger(__name__)
        tried = 0
        last_err = None
        while tried < 3:
            key = self._rot.next()
            try:
                async with httpx.AsyncClient(timeout=timeout) as client:
                    r = await client.post(url, headers=self._headers(key), json=json)
                    if r.status_code == 400:
                        error_details = r.text
                        logger.error(f"DashScope 400 Bad Request - URL: {url}")
                        logger.error(f"Request payload: {json}")
                        logger.error(f"Response: {error_details}")
                    r.raise_for_status()
                    return r.json()
            except Exception as e:
                tried += 1
                last_err = e
                logger.warning(f"DashScope request failed (attempt {tried}/3): {str(e)}")
                if tried < 3:
                    await asyncio.sleep(0.5)
        raise last_err or RuntimeError("Max retries exceeded")

    async def qwen_vl(self, images: List[str], prompt: str, result_format: str='json'):
        url = f"{_BASE}/services/aigc/multimodal-generation/generation"
        params = {"result_format": result_format}
        if self.temperature is not None:
            params["temperature"] = self.temperature
        if self.top_p is not None:
            params["top_p"] = self.top_p
        payload = {"model": self.vl_model,
                   "input": {"messages":[{"role":"user","content":[{"type":"text","text":prompt}] +
                               [{"type":"image","image":u} for u in images]}]},
                   "parameters": params}
        return await self._post_with_rotate(url, payload)

    async def qwen_text(self, prompt: str, result_format: str='json'):
        url = f"{_BASE}/services/aigc/text-generation/generation"
        params = {"result_format": result_format}
        if self.temperature is not None:
            params["temperature"] = self.temperature
        if self.top_p is not None:
            params["top_p"] = self.top_p
        payload = {"model": self.text_model,
                   "input": {"messages":[{"role":"user","content":prompt}]},
                   "parameters": params}
        return await self._post_with_rotate(url, payload)

    async def asr_paraformer(self, audio_url: str):
        url = f"{_BASE}/services/audio/asr"
        payload = {"model": self.asr_model,
                   "input": {"audio": {"url": audio_url}},
                   "parameters": {"language": "auto"},
                   "task": "transcription"}  # 添加必需的task参数
        return await self._post_with_rotate(url, payload, timeout=300)

class VisionDSClient:
    """专用于视觉分析的DSClient，使用API协调器"""
    def __init__(self, vl_model: str|None=None):
        self.vl_model = vl_model or os.getenv('DASHSCOPE_VL_MODEL', 'qwen-vl-plus')
        self.temperature = 0.3
        self.top_p = 0.9
        
        try:
            if os.getenv('DASHSCOPE_TEMPERATURE'):
                self.temperature = float(os.getenv('DASHSCOPE_TEMPERATURE'))
        except Exception:
            pass
        try:
            if os.getenv('DASHSCOPE_TOP_P'):
                self.top_p = float(os.getenv('DASHSCOPE_TOP_P'))
        except Exception:
            pass
    
    def _headers(self, key: str):
        return {"Content-Type":"application/json","Authorization":f"Bearer {key}"}
    
    async def qwen_vl_with_coordinator(self, images: List[str], prompt: str, coordinator, result_format: str='json'):
        """使用协调器进行视觉分析"""
        import logging
        logger = logging.getLogger(__name__)
        
        key_stats = await coordinator.acquire_api_key(timeout=30.0)
        if not key_stats:
            raise RuntimeError("无法获取可用的视觉API KEY，所有KEY都达到并发限制")
        
        try:
            url = f"{_BASE}/services/aigc/multimodal-generation/generation"
            params = {"result_format": result_format}
            if self.temperature is not None:
                params["temperature"] = self.temperature
            if self.top_p is not None:
                params["top_p"] = self.top_p
            
            payload = {
                "model": self.vl_model,
                "input": {
                    "messages": [{
                        "role": "user",
                        "content": [{"type": "text", "text": prompt}] +
                                 [{"type": "image", "image": u} for u in images]
                    }]
                },
                "parameters": params
            }
            
            async with httpx.AsyncClient(timeout=120) as client:
                r = await client.post(url, headers=self._headers(key_stats.key), json=payload)
                r.raise_for_status()
                return r.json()
                
        finally:
            coordinator.release_api_key(key_stats)

    def _headers(self, key: str):
        return {"Content-Type":"application/json","Authorization":f"Bearer {key}"}
