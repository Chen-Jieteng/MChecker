#!/usr/bin/env python3
"""
视觉API协调器测试脚本
用于验证多API KEY配置是否正常工作
"""

import asyncio
import os
import sys
from dotenv import load_dotenv

load_dotenv()

async def test_coordinator():
    """测试API协调器功能"""
    
    print(" 测试视觉API协调器...")
    
    try:
        from vision_api_coordinator import get_vision_coordinator, VisionApiCoordinator
        from dashscope_client import VisionDSClient
        
        coordinator = get_vision_coordinator()
        
        if coordinator:
            print(" API协调器已启用")
            
            status = coordinator.get_status()
            print(f" 协调器状态:")
            print(f"   总KEY数: {status['total_keys']}")
            print(f"   当前并发: {status['total_concurrent_requests']}")
            print(f"   可用槽位: {status['total_available_slots']}")
            
            for i, key_status in enumerate(status['keys_status']):
                print(f"   KEY{i+1}: {key_status['key_prefix']} "
                     f"并发{key_status['current_concurrent']}/{key_status['max_concurrent']} "
                     f"可用:{key_status['available']}")
            
            print("\n 测试KEY获取机制...")
            
            key_stats = await coordinator.acquire_api_key(timeout=5.0)
            if key_stats:
                print(f" 成功获取API KEY: {key_stats.key[:8]}***")
                coordinator.release_api_key(key_stats)
                print(" 成功释放API KEY")
            else:
                print(" 获取API KEY超时")
            
            print("\n 协调器测试完成！")
            
        else:
            print(" API协调器未启用")
            print(" 可能的原因:")
            print("   1. 未配置 DASHSCOPE_VISION_API_KEYS")
            print("   2. 环境变量格式错误")
            print("   3. API KEY为空")
            
            vision_keys = os.getenv("DASHSCOPE_VISION_API_KEYS", "")
            general_keys = os.getenv("DASHSCOPE_API_KEYS", "")
            single_key = os.getenv("DASHSCOPE_API_KEY", "")
            
            print(f"\n 环境变量检查:")
            print(f"   DASHSCOPE_VISION_API_KEYS: {'已配置' if vision_keys else '未配置'}")
            print(f"   DASHSCOPE_API_KEYS: {'已配置' if general_keys else '未配置'}")
            print(f"   DASHSCOPE_API_KEY: {'已配置' if single_key else '未配置'}")
            
            if vision_keys:
                keys = [k.strip() for k in vision_keys.split(',') if k.strip()]
                print(f"   解析到 {len(keys)} 个视觉API KEY")
            elif general_keys:
                keys = [k.strip() for k in general_keys.split(',') if k.strip()]
                print(f"   解析到 {len(keys)} 个通用API KEY")
            elif single_key:
                print(f"   配置了单个API KEY（将回退到顺序处理）")
            else:
                print("    没有找到任何API KEY配置")
                
    except ImportError as e:
        print(f" 导入错误: {e}")
        print(" 请确保在agent_backend目录下运行此脚本")
    except Exception as e:
        print(f" 测试失败: {e}")

async def test_vision_client():
    """测试VisionDSClient功能"""
    
    print("\n 测试VisionDSClient...")
    
    try:
        from dashscope_client import VisionDSClient
        from vision_api_coordinator import get_vision_coordinator
        
        vision_client = VisionDSClient()
        coordinator = get_vision_coordinator()
        
        print(" VisionDSClient 初始化成功")
        
        if coordinator:
            print(" 将使用API协调器进行并行分析")
        else:
            print(" 将使用默认轮换机制（顺序处理）")
            
    except Exception as e:
        print(f" VisionDSClient 测试失败: {e}")

def print_config_guide():
    """打印配置指南"""
    
    print("\n" + "="*60)
    print(" 快速配置指南")
    print("="*60)
    print("1. 在 .env 文件中添加多个API KEY：")
    print("   DASHSCOPE_VISION_API_KEYS=sk-key1,sk-key2,sk-key3,sk-key4,sk-key5")
    print("")
    print("2. 可选高级配置：")
    print("   VISION_API_MAX_CONCURRENT_PER_KEY=5")
    print("   VISION_API_MAX_PER_MINUTE_PER_KEY=300")
    print("")
    print("3. 重启后端服务：")
    print("   uvicorn main:app --host 0.0.0.0 --port 8799 --reload")
    print("")
    print("4. 检查协调器状态：")
    print("   GET http://localhost:8799/agent/api/status")
    print("="*60)

async def main():
    """主测试函数"""
    
    print(" 视觉API协调器功能测试")
    print("="*50)
    
    await test_coordinator()
    await test_vision_client()
    
    print_config_guide()
    
    print("\n 测试完成！")

if __name__ == "__main__":
    if not os.path.exists("vision_api_coordinator.py"):
        print(" 请在 agent_backend 目录下运行此脚本")
        sys.exit(1)
    
    asyncio.run(main())
