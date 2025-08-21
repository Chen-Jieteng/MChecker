"""
视觉AI专用API KEY协调器
解决DashScope视觉模型并发限制问题
"""
import asyncio
import time
import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from itertools import cycle
import logging

logger = logging.getLogger(__name__)

@dataclass
class ApiKeyStats:
    """API KEY统计信息"""
    key: str
    requests_this_minute: int = 0
    last_reset_time: float = 0
    current_concurrent: int = 0
    max_concurrent: int = 5  # 每个KEY最大并发数
    max_per_minute: int = 300  # 每分钟最大请求数（保守估计）
    
    def can_accept_request(self) -> bool:
        """检查是否可以接受新请求"""
        now = time.time()
        if now - self.last_reset_time >= 60:
            self.requests_this_minute = 0
            self.last_reset_time = now
        
        return (self.current_concurrent < self.max_concurrent and 
                self.requests_this_minute < self.max_per_minute)
    
    def acquire(self):
        """获取请求许可"""
        self.current_concurrent += 1
        self.requests_this_minute += 1
    
    def try_acquire(self):
        """尝试原子获取请求许可 - 并发安全版本"""
        if self.can_accept_request():
            self.current_concurrent += 1
            self.requests_this_minute += 1
            return True
        return False
    
    def release(self):
        """释放请求许可"""
        if self.current_concurrent > 0:
            self.current_concurrent -= 1

class VisionApiCoordinator:
    """视觉AI API协调器"""
    
    def __init__(self, api_keys: List[str], max_concurrent_per_key: int = 5, max_per_minute_per_key: int = 300):
        """
        初始化协调器
        
        Args:
            api_keys: API KEY列表
            max_concurrent_per_key: 每个KEY的最大并发数
            max_per_minute_per_key: 每个KEY每分钟最大请求数
        """
        if not api_keys:
            raise ValueError("至少需要提供一个API KEY")
        
        self.stats = [
            ApiKeyStats(
                key=key,
                max_concurrent=max_concurrent_per_key,
                max_per_minute=max_per_minute_per_key
            ) 
            for key in api_keys
        ]
        
        
        logger.info(f"🔧 视觉API协调器初始化: {len(api_keys)}个KEY, 每KEY最大并发{max_concurrent_per_key}, 每分钟{max_per_minute_per_key}请求")
    
    async def acquire_api_key(self, timeout: float = 30.0) -> Optional[ApiKeyStats]:
        """
        获取可用的API KEY
        
        Args:
            timeout: 超时时间（秒）
            
        Returns:
            可用的API KEY统计对象，或None（超时）
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            available_keys = [stat for stat in self.stats if stat.can_accept_request()]
            
            if available_keys:
                chosen_key = min(available_keys, key=lambda x: x.current_concurrent)
                
                if chosen_key.try_acquire():
                    logger.debug(f"🔑 并行获取API KEY: {chosen_key.key[:8]}*** "
                               f"(并发: {chosen_key.current_concurrent}/{chosen_key.max_concurrent}, "
                               f"分钟请求: {chosen_key.requests_this_minute}/{chosen_key.max_per_minute})")
                    
                    return chosen_key
            
            await asyncio.sleep(0.1)
        
        logger.warning(f"⚠️ API KEY获取超时({timeout}s)，所有KEY都达到限制")
        return None
    
    def release_api_key(self, key_stats: ApiKeyStats):
        """释放API KEY"""
        key_stats.release()
        logger.debug(f"📤 释放API KEY: {key_stats.key[:8]}*** "
                    f"(剩余并发: {key_stats.current_concurrent}/{key_stats.max_concurrent})")
    
    def get_status(self) -> Dict[str, Any]:
        """获取协调器状态"""
        total_concurrent = sum(stat.current_concurrent for stat in self.stats)
        total_available = sum(
            stat.max_concurrent - stat.current_concurrent 
            for stat in self.stats 
            if stat.can_accept_request()
        )
        
        return {
            'total_keys': len(self.stats),
            'total_concurrent_requests': total_concurrent,
            'total_available_slots': total_available,
            'keys_status': [
                {
                    'key_prefix': stat.key[:8] + '***',
                    'current_concurrent': stat.current_concurrent,
                    'max_concurrent': stat.max_concurrent,
                    'requests_this_minute': stat.requests_this_minute,
                    'max_per_minute': stat.max_per_minute,
                    'available': stat.can_accept_request()
                }
                for stat in self.stats
            ]
        }

_vision_coordinator: Optional[VisionApiCoordinator] = None

def init_vision_coordinator():
    """初始化视觉API协调器"""
    global _vision_coordinator
    
    vision_keys_raw = os.getenv("DASHSCOPE_VISION_API_KEYS", "")
    if not vision_keys_raw:
        vision_keys_raw = os.getenv("DASHSCOPE_API_KEYS") or os.getenv("DASHSCOPE_API_KEY", "")
    
    vision_keys = [k.strip() for k in vision_keys_raw.split(',') if k.strip()]
    
    if not vision_keys:
        logger.warning("⚠️ 未配置视觉API KEY，将使用默认轮换机制")
        return
    
    max_concurrent = int(os.getenv("VISION_API_MAX_CONCURRENT_PER_KEY", "5"))
    max_per_minute = int(os.getenv("VISION_API_MAX_PER_MINUTE_PER_KEY", "300"))
    
    _vision_coordinator = VisionApiCoordinator(
        api_keys=vision_keys,
        max_concurrent_per_key=max_concurrent,
        max_per_minute_per_key=max_per_minute
    )
    
    logger.info(f"✅ 视觉API协调器已启动: {len(vision_keys)}个专用KEY")

def get_vision_coordinator() -> Optional[VisionApiCoordinator]:
    """获取全局视觉API协调器"""
    return _vision_coordinator

init_vision_coordinator()
