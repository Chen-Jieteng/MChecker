# MChecker Agent Backend (CN-first)

- FastAPI 服务
- 国内优先模型：
  - 文本/多模态：通义千问 DashScope（Qwen-VL / Qwen 2.5）
  - 语音转写：SenseVoice（可替 Whisper API）
  - 可选：百度千帆(qianfan)

## 开发
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r agent_backend/requirements.txt
uvicorn agent_backend.main:app --host 0.0.0.0 --port 8799 --reload
```

## API
- POST /agent/audit {aweme_id?, video_url?} → {task_id, result{risk_level, counters, summary}}

> 后续接入：
> 1) /extract: 抽帧+OCR+NSFW+转写 → structured context
> 2) /judge: LLM 审核引擎（DashScope）
 
