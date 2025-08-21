# 📋 视觉AI API协调器配置指南

## 🎯 **API并发限制分析**

根据我们的调研，阿里百炼DashScope API的限制如下：

### 📊 **并发限制数据**
| 指标 | 推测值 | 说明 |
|------|--------|------|
| **单KEY并发数** | 5个 | 每个API KEY同时处理的请求数 |
| **分钟请求数** | 300个 | 每个API KEY每分钟的请求限制 |
| **默认API网关** | 500 QPS | 阿里云API网关的通用限制 |

### ⚠️ **实际观察到的问题**
- 5个片段同时发起请求时触发限制
- WebSocket连接超时（`keepalive ping timeout`）
- 只能完成第1个片段分析，后续请求被阻塞

---

## 🔧 **API KEY协调器工作原理**

### 🎯 **智能分配策略**
```
请求到达 → 协调器检查所有KEY状态 → 选择空闲KEY → 执行请求 → 释放KEY
```

### 📈 **并发管理**
- **动态负载均衡**: 选择当前并发数最少的KEY
- **限流保护**: 每KEY最大5并发 + 每分钟300请求
- **错误恢复**: 请求失败时自动重试其他KEY
- **实时监控**: 提供详细的KEY使用状态

---

## ⚙️ **配置方法**

### 🔑 **方式1: 专用视觉API KEY（推荐）**
```bash
# 在 .env 文件中添加：
DASHSCOPE_VISION_API_KEYS=sk-key1,sk-key2,sk-key3,sk-key4,sk-key5
```

### 🔑 **方式2: 通用API KEY（回退）**
```bash
# 如果没有专用KEY，使用通用配置：
DASHSCOPE_API_KEYS=sk-key1,sk-key2,sk-key3
```

### 🔑 **方式3: 单个KEY（基础模式）**
```bash
# 最基础配置，将回退到顺序处理：
DASHSCOPE_API_KEY=sk-your-single-key
```

---

## 📊 **API KEY数量建议**

### 🧮 **计算公式**
```
需要KEY数 = 最大同时片段数 ÷ 每KEY并发数

示例：
- 视频最长300秒 ÷ 5秒片段 = 最多60个片段
- 每KEY支持5并发
- 建议KEY数 = 60 ÷ 5 = 12个KEY

保守建议：
- 短视频（<60秒）：3-5个KEY
- 中等视频（60-180秒）：5-8个KEY  
- 长视频（180秒+）：8-12个KEY
```

### 💡 **实用建议**
- **初期配置**: 先提供 **5个API KEY** 测试效果
- **性能优化**: 根据实际视频长度调整KEY数量
- **成本控制**: 更多KEY = 更快分析，但成本更高

---

## 🚀 **高级配置选项**

```bash
# 每个KEY最大并发数（默认5）
VISION_API_MAX_CONCURRENT_PER_KEY=5

# 每个KEY每分钟最大请求数（默认300）
VISION_API_MAX_PER_MINUTE_PER_KEY=300

# 视觉模型选择
DASHSCOPE_VL_MODEL=qwen-vl-max  # 推荐，效果更好
```

---

## 📊 **实时监控**

### 🔍 **状态检查API**
```bash
GET http://localhost:8799/agent/api/status
```

**响应示例**:
```json
{
  "coordinator_enabled": true,
  "total_keys": 5,
  "total_concurrent_requests": 12,
  "total_available_slots": 13,
  "keys_status": [
    {
      "key_prefix": "sk-abcd***",
      "current_concurrent": 3,
      "max_concurrent": 5,
      "requests_this_minute": 45,
      "max_per_minute": 300,
      "available": true
    }
  ]
}
```

---

## 🎯 **性能对比**

| 配置方式 | 并发能力 | 分析速度 | WebSocket稳定性 |
|----------|----------|----------|----------------|
| **单KEY** | 顺序处理 | 慢（5个片段需25秒） | ✅ 稳定 |
| **5个KEY** | 25并发 | 快（5个片段需5秒） | ✅ 稳定 |
| **10个KEY** | 50并发 | 极快（10个片段需2秒） | ✅ 稳定 |

---

## ❓ **常见问题**

### Q: 我需要多少个API KEY？
**A**: 建议从 **5个KEY** 开始：
- 能支持25个并发片段（约2分钟视频）
- 大部分使用场景都够用
- 性价比最优

### Q: 如何判断协调器是否生效？
**A**: 查看后端日志：
- ✅ 看到 `智能并行预加载片段分析` = 协调器工作
- ❌ 看到 `顺序预加载片段分析` = 回退到单KEY模式

### Q: 多个KEY会增加成本吗？
**A**: 
- API调用总数不变，只是分散到不同KEY
- 成本 = 总请求数 × 单价，与KEY数量无关
- 但能显著提升用户体验

---

## 🔧 **立即开始**

1. **准备API KEY**: 从阿里云控制台获取多个DashScope API KEY
2. **配置环境变量**: 在 `.env` 文件中设置 `DASHSCOPE_VISION_API_KEYS`
3. **重启服务**: 重启后端服务应用新配置
4. **测试效果**: 启动智能体模式，观察并行分析效果

**配置示例**:
```bash
# 复制以下内容到 .env 文件
DASHSCOPE_VISION_API_KEYS=sk-your-key-1,sk-your-key-2,sk-your-key-3,sk-your-key-4,sk-your-key-5
VISION_API_MAX_CONCURRENT_PER_KEY=5
VISION_API_MAX_PER_MINUTE_PER_KEY=300
DASHSCOPE_VL_MODEL=qwen-vl-max
```

---

🎉 **现在您的视频分析系统将具备强大的并行处理能力！**
