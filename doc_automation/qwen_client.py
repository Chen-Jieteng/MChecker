"""
Qwen-plus-latest API客户端
专门用于10万字小说生成，支持1M上下文
"""

import os
import json
import time
import logging
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
import dashscope
from dashscope import Generation

# 配置API
dashscope.api_key = os.getenv('DASHSCOPE_API_KEY', 'sk-placeholder-replace-with-real-key')

@dataclass
class GenerationRequest:
    """生成请求"""
    messages: List[Dict[str, str]]
    max_tokens: int = 3000
    temperature: float = 0.7
    top_p: float = 0.9
    presence_penalty: float = 0.1
    stream: bool = False

@dataclass
class GenerationResponse:
    """生成响应"""
    content: str
    usage: Dict[str, int]
    finish_reason: str
    success: bool
    error_message: Optional[str] = None

class QwenClient:
    """Qwen-plus-latest客户端，专门优化用于长文本生成"""
    
    def __init__(self, api_key: Optional[str] = None):
        if api_key:
            dashscope.api_key = api_key
        
        self.model = "qwen-plus-latest"
        self.max_context_length = 1000000  # 1M tokens
        self.logger = logging.getLogger(__name__)
        
        # 针对小说生成的优化参数
        self.novel_config = {
            "temperature": 0.7,  # 保持创造性
            "top_p": 0.9,       # 高质量采样
            "presence_penalty": 0.1,  # 轻微避免重复
            "frequency_penalty": 0.05,  # 轻微避免重复词汇
        }
    
    def generate_scene(self, request: GenerationRequest) -> GenerationResponse:
        """生成单个场景，专门优化的接口"""
        try:
            # 构建请求参数
            messages = request.messages
            
            # 确保不超过上下文限制
            total_tokens = self._estimate_tokens(messages)
            if total_tokens > self.max_context_length * 0.8:  # 留20%余量
                self.logger.warning(f"上下文过长: {total_tokens} tokens，进行截断")
                messages = self._truncate_messages(messages, int(self.max_context_length * 0.6))
            
            response = Generation.call(
                model=self.model,
                messages=messages,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                stream=request.stream,
                **self.novel_config
            )
            
            if response.status_code == 200:
                output = response.output
                usage = response.usage
                
                return GenerationResponse(
                    content=output.choices[0].message.content,
                    usage={
                        "prompt_tokens": usage.input_tokens,
                        "completion_tokens": usage.output_tokens,
                        "total_tokens": usage.total_tokens
                    },
                    finish_reason=output.choices[0].finish_reason,
                    success=True
                )
            else:
                error_msg = f"API调用失败: {response.status_code} - {response.message}"
                self.logger.error(error_msg)
                return GenerationResponse(
                    content="",
                    usage={},
                    finish_reason="error",
                    success=False,
                    error_message=error_msg
                )
                
        except Exception as e:
            error_msg = f"生成过程异常: {str(e)}"
            self.logger.error(error_msg)
            return GenerationResponse(
                content="",
                usage={},
                finish_reason="error", 
                success=False,
                error_message=error_msg
            )
    
    def generate_with_retry(self, request: GenerationRequest, max_retries: int = 3) -> GenerationResponse:
        """带重试的生成"""
        for attempt in range(max_retries):
            response = self.generate_scene(request)
            
            if response.success:
                return response
            
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # 指数退避
                self.logger.info(f"第{attempt + 1}次重试失败，等待{wait_time}秒后重试")
                time.sleep(wait_time)
        
        return response  # 返回最后一次的失败结果
    
    def generate_outline(self, title: str, premise: str, style: str, target_chapters: int) -> GenerationResponse:
        """生成小说大纲"""
        system_prompt = """你是一位经验丰富的小说大纲设计师。请根据用户提供的题目、设定和风格，生成详细的章节大纲。

要求：
1. 生成具体的章节标题和简要内容
2. 确保情节有起承转合，节奏合理
3. 每章都要有明确的冲突和目标
4. 为主要角色设计成长弧线
5. 埋设合理的伏笔和回收点

输出格式为JSON，包含：
- overall_theme: 整体主题
- main_conflicts: 主要冲突列表
- character_arcs: 主角成长弧线
- chapters: 章节列表，每章包含title、summary、key_events、conflicts"""

        user_prompt = f"""请为以下小说生成{target_chapters}章的详细大纲：

标题：{title}
基本设定：{premise}
文学风格：{style}
目标章节数：{target_chapters}

请确保大纲结构完整，情节连贯，有足够的深度支撑长篇小说的发展。"""

        request = GenerationRequest(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=4000,
            temperature=0.8
        )
        
        return self.generate_with_retry(request)
    
    def generate_character_bible(self, title: str, premise: str, style: str, main_characters: List[str]) -> GenerationResponse:
        """生成角色圣经"""
        system_prompt = """你是一位专业的角色设计师。请根据小说设定，为主要角色创建详细的角色卡片。

每个角色需要包含：
1. 基本信息：姓名、年龄、职业、社会地位
2. 外貌特征：身高、体型、面容、着装风格、标志性特征
3. 性格特点：核心性格、优缺点、行为模式、价值观
4. 背景故事：出身、重要经历、创伤、成就
5. 能力设定：技能、特殊能力、知识领域
6. 人际关系：与其他角色的关系
7. 语言特色：说话方式、口癖、表达习惯
8. 成长目标：想要达成的目标、内心冲突
9. 弱点盲区：致命弱点、恐惧、局限

输出格式为JSON数组。"""

        user_prompt = f"""请为小说《{title}》创建主要角色的详细卡片：

小说设定：{premise}
文学风格：{style}
主要角色：{', '.join(main_characters)}

请确保角色设定符合故事背景，有足够的深度和复杂性，角色间有合理的关系网络。"""

        request = GenerationRequest(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=5000,
            temperature=0.7
        )
        
        return self.generate_with_retry(request)
    
    def generate_world_bible(self, title: str, premise: str, style: str) -> GenerationResponse:
        """生成世界观圣经"""
        system_prompt = """你是一位世界观设计师。请根据小说设定创建详细的世界观圣经。

需要包含的元素：
1. 世界基本设定：时空背景、历史概述、地理环境
2. 社会结构：政治制度、经济体系、社会阶层、文化传统
3. 技术/魔法体系：核心机制、规则限制、发展历程
4. 重要地点：主要场景的详细描述、地理关系
5. 组织机构：各种势力、组织、其影响力和关系
6. 文化要素：语言、宗教、习俗、艺术形式
7. 特殊规则：这个世界独有的规律、禁忌、常识
8. 历史事件：影响当前格局的重要历史

输出格式为JSON结构。"""

        user_prompt = f"""请为小说《{title}》创建详细的世界观圣经：

基本设定：{premise}
文学风格：{style}

请确保世界观设定完整、自洽，有足够的深度支撑长篇小说的展开。"""

        request = GenerationRequest(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=5000,
            temperature=0.7
        )
        
        return self.generate_with_retry(request)
    
    def generate_scene_content(self, 
                              scene_prompt: str,
                              character_context: str,
                              world_context: str,
                              plot_context: str,
                              bridge_content: str,
                              style_guide: str) -> GenerationResponse:
        """生成场景内容"""
        
        system_prompt = f"""你是一位专业的小说作家。请根据提供的上下文信息，写出高质量的小说场景。

写作要求：
1. 保持与已有内容的连贯性
2. 严格遵循角色设定，不能出现人物OOC
3. 遵循世界观规则，保持设定一致性
4. 推进情节，处理指定的冲突点
5. {style_guide}
6. 场景要完整，有明确的开始和结束
7. 控制篇幅在1000-2000字之间
8. 语言生动，节奏合理，有画面感

严禁：
- 重复已有情节
- 违背角色性格
- 破坏世界观设定
- 情节突兀跳跃
- 用词重复、语言贫乏"""

        user_prompt = f"""请写出下一个场景的内容：

【场景任务】
{scene_prompt}

【桥接内容】
{bridge_content}

【相关角色】
{character_context}

【世界观背景】
{world_context}

【情节线索】
{plot_context}

请写出完整的场景内容。"""

        request = GenerationRequest(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=3000,
            temperature=0.7
        )
        
        return self.generate_with_retry(request)
    
    def self_review_content(self, content: str, review_criteria: str) -> GenerationResponse:
        """自检内容质量"""
        system_prompt = """你是一位专业的文学编辑。请对提供的小说内容进行质量评估和改进建议。

评估维度：
1. 语言流畅性：表达是否自然、生动
2. 情节连贯性：与前文是否连接自然
3. 角色一致性：人物行为是否符合设定
4. 节奏把控：紧张缓急是否合理
5. 画面感：描写是否生动具体
6. 对话真实性：对话是否自然、符合角色

输出格式：
{
  "overall_score": 1-10,
  "strengths": ["优点1", "优点2"],
  "weaknesses": ["问题1", "问题2"],
  "specific_suggestions": [
    {"location": "具体位置", "issue": "问题描述", "suggestion": "修改建议"}
  ],
  "revised_version": "如果需要重大修改，提供修改后的版本"
}"""

        user_prompt = f"""请评估以下小说内容的质量：

【评估标准】
{review_criteria}

【待评估内容】
{content}

请给出详细的评估和改进建议。"""

        request = GenerationRequest(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=2000,
            temperature=0.3  # 降低温度以获得更客观的评估
        )
        
        return self.generate_with_retry(request)
    
    def _estimate_tokens(self, messages: List[Dict[str, str]]) -> int:
        """估算token数量（粗略估计）"""
        total_chars = sum(len(msg.get("content", "")) for msg in messages)
        # 中文大约1.5个字符 = 1个token，英文约4个字符 = 1个token
        return int(total_chars * 0.75)  # 保守估计
    
    def _truncate_messages(self, messages: List[Dict[str, str]], max_tokens: int) -> List[Dict[str, str]]:
        """截断消息以适应上下文限制"""
        # 保留system消息和最后的user消息
        if len(messages) <= 2:
            return messages
        
        system_msg = messages[0] if messages[0].get("role") == "system" else None
        user_msg = messages[-1] if messages[-1].get("role") == "user" else None
        middle_msgs = messages[1:-1] if len(messages) > 2 else []
        
        # 估算system和user消息的token数
        reserved_tokens = 0
        if system_msg:
            reserved_tokens += self._estimate_tokens([system_msg])
        if user_msg:
            reserved_tokens += self._estimate_tokens([user_msg])
        
        # 为中间消息分配剩余token
        available_tokens = max_tokens - reserved_tokens
        
        # 从最新的消息开始保留
        truncated_middle = []
        current_tokens = 0
        
        for msg in reversed(middle_msgs):
            msg_tokens = self._estimate_tokens([msg])
            if current_tokens + msg_tokens <= available_tokens:
                truncated_middle.insert(0, msg)
                current_tokens += msg_tokens
            else:
                break
        
        # 重新组装消息
        result = []
        if system_msg:
            result.append(system_msg)
        result.extend(truncated_middle)
        if user_msg:
            result.append(user_msg)
        
        return result
