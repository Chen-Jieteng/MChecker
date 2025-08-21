"""
小说编排器 - 10万字小说生成的核心组件
负责整体规划、分层生成、质量控制、进度管理
"""

import json
import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import uuid
import re

from external_memory import (
    ExternalMemory, Character, WorldElement, PlotThread, 
    SceneSummary, TimelineEvent
)
from qwen_client import QwenClient, GenerationRequest

@dataclass
class NovelConfig:
    """小说生成配置"""
    title: str
    premise: str
    style: str
    target_length: int
    mode: str  # 'create' or 'continue'
    existing_content: str = ""
    chapters_target: int = 12
    words_per_chapter: int = 8000
    words_per_scene: int = 1500

@dataclass
class GenerationProgress:
    """生成进度"""
    current_chapter: int
    current_scene: int
    total_chapters: int
    words_written: int
    target_words: int
    status: str  # planning, writing, reviewing, completed, error
    current_task: str
    estimated_completion: Optional[str]
    created_at: str
    updated_at: str

@dataclass
class SceneTask:
    """场景生成任务"""
    chapter: int
    scene: int
    title: str
    objective: str
    key_events: List[str]
    characters_involved: List[str]
    plot_threads_to_advance: List[str]
    estimated_words: int
    dependencies: List[str]  # 依赖的前置场景
    status: str  # pending, in_progress, completed, failed

class NovelOrchestrator:
    """小说编排器 - 核心生成引擎"""
    
    def __init__(self, memory_dir: str = "./novel_memory"):
        self.memory = ExternalMemory(memory_dir)
        self.qwen_client = QwenClient()
        self.logger = logging.getLogger(__name__)
        
        self.current_config: Optional[NovelConfig] = None
        self.current_progress: Optional[GenerationProgress] = None
        self.scene_tasks: List[SceneTask] = []
        self.generation_state = {}
        
        self.quality_threshold = 7.0  # 1-10分，低于此分数需要重写
        self.max_revision_attempts = 3
        
        self.bridge_config = {
            "context_scenes": 2,  # 回溯多少个场景作为上下文
            "summary_length": 120,  # 摘要长度
            "bridge_paragraph_count": 2  # 桥接段落数量
        }
    
    async def generate_novel(self, config: NovelConfig, progress_callback=None) -> Dict[str, Any]:
        """生成完整小说"""
        try:
            self.current_config = config
            
            self.current_progress = GenerationProgress(
                current_chapter=0,
                current_scene=0,
                total_chapters=config.chapters_target,
                words_written=0,
                target_words=config.target_length,
                status="planning",
                current_task="初始化规划阶段",
                estimated_completion=None,
                created_at=datetime.now().isoformat(),
                updated_at=datetime.now().isoformat()
            )
            
            if progress_callback:
                await progress_callback(self.current_progress)
            
            if config.mode == "create":
                result = await self._generate_from_scratch(progress_callback)
            else:
                result = await self._continue_existing(progress_callback)
            
            return {
                "success": True,
                "result": result,
                "progress": asdict(self.current_progress),
                "stats": self.memory.get_novel_stats()
            }
            
        except Exception as e:
            error_msg = f"小说生成失败: {str(e)}"
            self.logger.error(error_msg)
            
            if self.current_progress:
                self.current_progress.status = "error"
                self.current_progress.current_task = error_msg
                self.current_progress.updated_at = datetime.now().isoformat()
            
            return {
                "success": False,
                "error": error_msg,
                "progress": asdict(self.current_progress) if self.current_progress else None
            }
    
    async def _generate_from_scratch(self, progress_callback) -> str:
        """从零开始生成小说"""
        
        await self._update_progress("planning", "生成整体大纲", progress_callback)
        outline = await self._generate_outline()
        
        await self._update_progress("planning", "创建角色圣经", progress_callback)
        await self._create_character_bible()
        
        await self._update_progress("planning", "构建世界观", progress_callback)
        await self._create_world_bible()
        
        await self._update_progress("planning", "规划场景任务", progress_callback)
        await self._plan_scene_tasks(outline)
        
        await self._update_progress("writing", "开始章节写作", progress_callback)
        novel_content = await self._write_chapters(progress_callback)
        
        await self._update_progress("reviewing", "生成目录和后记", progress_callback)
        final_content = await self._finalize_novel(novel_content)
        
        await self._update_progress("completed", "小说生成完成", progress_callback)
        
        return final_content
    
    async def _continue_existing(self, progress_callback) -> str:
        """续写现有小说"""
        
        await self._update_progress("planning", "分析现有内容", progress_callback)
        analysis = await self._analyze_existing_content()
        
        await self._update_progress("planning", "提取角色和世界观", progress_callback)
        await self._extract_characters_and_world(analysis)
        
        await self._update_progress("planning", "规划续写章节", progress_callback)
        continuation_outline = await self._plan_continuation(analysis)
        
        await self._update_progress("writing", "开始续写", progress_callback)
        continuation_content = await self._write_continuation(continuation_outline, progress_callback)
        
        final_content = self.current_config.existing_content + "\n\n" + continuation_content
        
        await self._update_progress("completed", "续写完成", progress_callback)
        
        return final_content
    
    async def _generate_outline(self) -> Dict[str, Any]:
        """生成详细大纲"""
        response = await asyncio.to_thread(
            self.qwen_client.generate_outline,
            self.current_config.title,
            self.current_config.premise,
            self.current_config.style,
            self.current_config.chapters_target
        )
        
        if not response.success:
            raise Exception(f"大纲生成失败: {response.error_message}")
        
        try:
            outline = json.loads(response.content)
            
            plot_threads = []
            for i, conflict in enumerate(outline.get("main_conflicts", [])):
                thread = PlotThread(
                    id=f"thread_{i+1}",
                    name=conflict,
                    description=f"主要冲突: {conflict}",
                    status="pending",
                    importance=5,
                    planted_chapter=1,
                    must_resolve_by=self.current_config.chapters_target,
                    related_characters=[],
                    related_elements=[],
                    created_at=datetime.now().isoformat(),
                    updated_at=datetime.now().isoformat()
                )
                plot_threads.append(thread)
                self.memory.add_plot_thread(thread)
            
            return outline
            
        except json.JSONDecodeError:
            self.logger.warning("大纲JSON解析失败，使用基础大纲")
            return self._create_basic_outline()
    
    async def _create_character_bible(self):
        """创建角色圣经"""
        main_characters = ["主角", "关键配角1", "关键配角2", "反派"]
        
        response = await asyncio.to_thread(
            self.qwen_client.generate_character_bible,
            self.current_config.title,
            self.current_config.premise,
            self.current_config.style,
            main_characters
        )
        
        if response.success:
            try:
                characters_data = json.loads(response.content)
                for char_data in characters_data:
                    character = Character(
                        name=char_data.get("name", "未知角色"),
                        description=char_data.get("description", ""),
                        appearance=char_data.get("appearance", ""),
                        personality=char_data.get("personality", ""),
                        background=char_data.get("background", ""),
                        motivation=char_data.get("motivation", ""),
                        abilities=char_data.get("abilities", []),
                        relationships=char_data.get("relationships", {}),
                        speech_patterns=char_data.get("speech_patterns", []),
                        growth_arc=char_data.get("growth_arc", ""),
                        current_state=char_data.get("current_state", ""),
                        created_at=datetime.now().isoformat(),
                        updated_at=datetime.now().isoformat()
                    )
                    self.memory.add_character(character)
                    
            except json.JSONDecodeError:
                self.logger.warning("角色数据JSON解析失败")
    
    async def _create_world_bible(self):
        """创建世界观圣经"""
        response = await asyncio.to_thread(
            self.qwen_client.generate_world_bible,
            self.current_config.title,
            self.current_config.premise,
            self.current_config.style
        )
        
        if response.success:
            try:
                world_data = json.loads(response.content)
                
                for element_type, elements in world_data.items():
                    if isinstance(elements, dict):
                        for name, details in elements.items():
                            world_element = WorldElement(
                                name=name,
                                type=element_type,
                                description=str(details),
                                properties={},
                                relationships={},
                                rules=[],
                                created_at=datetime.now().isoformat(),
                                updated_at=datetime.now().isoformat()
                            )
                            self.memory.add_world_element(world_element)
                            
            except (json.JSONDecodeError, TypeError):
                self.logger.warning("世界观数据解析失败")
    
    async def _plan_scene_tasks(self, outline: Dict[str, Any]):
        """规划场景任务"""
        chapters = outline.get("chapters", [])
        
        for chapter_idx, chapter in enumerate(chapters, 1):
            chapter_title = chapter.get("title", f"第{chapter_idx}章")
            chapter_events = chapter.get("key_events", [])
            
            scenes_per_chapter = max(2, len(chapter_events))
            events_per_scene = max(1, len(chapter_events) // scenes_per_chapter)
            
            for scene_idx in range(scenes_per_chapter):
                scene_events = chapter_events[scene_idx * events_per_scene:(scene_idx + 1) * events_per_scene]
                
                task = SceneTask(
                    chapter=chapter_idx,
                    scene=scene_idx + 1,
                    title=f"{chapter_title} - 场景{scene_idx + 1}",
                    objective=f"推进情节: {', '.join(scene_events[:2])}",
                    key_events=scene_events,
                    characters_involved=[],
                    plot_threads_to_advance=[],
                    estimated_words=self.current_config.words_per_scene,
                    dependencies=[],
                    status="pending"
                )
                self.scene_tasks.append(task)
    
    async def _write_chapters(self, progress_callback) -> str:
        """写作所有章节"""
        novel_parts = []
        
        novel_parts.append(f"# {self.current_config.title}\n\n")
        novel_parts.append(f"> **生成时间**: {datetime.now().strftime('%Y年%m月%d日')}\n")
        novel_parts.append(f"> **字数目标**: {self.current_config.target_length:,}字\n")
        novel_parts.append(f"> **文学风格**: {self.current_config.style}\n\n")
        novel_parts.append("---\n\n")
        
        for chapter_num in range(1, self.current_config.chapters_target + 1):
            chapter_tasks = [task for task in self.scene_tasks if task.chapter == chapter_num]
            
            await self._update_progress(
                "writing", 
                f"写作第{chapter_num}章", 
                progress_callback
            )
            
            chapter_content = await self._write_chapter(chapter_num, chapter_tasks)
            novel_parts.append(chapter_content)
            
            self.current_progress.current_chapter = chapter_num
            self.current_progress.words_written = len(''.join(novel_parts))
            
            if progress_callback:
                await progress_callback(self.current_progress)
        
        return ''.join(novel_parts)
    
    async def _write_chapter(self, chapter_num: int, scene_tasks: List[SceneTask]) -> str:
        """写作单个章节"""
        chapter_parts = [f"## 第{chapter_num}章\n\n"]
        
        for task in scene_tasks:
            scene_content = await self._write_scene(task)
            chapter_parts.append(scene_content)
            chapter_parts.append("\n\n")
            
            summary = SceneSummary(
                chapter=task.chapter,
                scene=task.scene,
                title=task.title,
                summary=self._extract_summary(scene_content),
                key_events=task.key_events,
                characters_present=task.characters_involved,
                plot_threads_touched=task.plot_threads_to_advance,
                emotional_tone="",
                word_count=len(scene_content),
                created_at=datetime.now().isoformat()
            )
            self.memory.add_scene_summary(summary)
        
        return ''.join(chapter_parts)
    
    async def _write_scene(self, task: SceneTask) -> str:
        """写作单个场景"""
        bridge_content = await self._build_bridge_context(task.chapter, task.scene)
        character_context = await self._build_character_context(task)
        world_context = await self._build_world_context(task)
        plot_context = await self._build_plot_context(task)
        style_guide = self._get_style_guide()
        
        scene_prompt = f"""
请写作第{task.chapter}章第{task.scene}个场景：{task.title}

场景目标：{task.objective}
关键事件：{', '.join(task.key_events)}
预计字数：{task.estimated_words}字左右

要求：
1. 场景要完整，有清晰的开始和结束
2. 推进指定的情节事件
3. 展现角色性格和成长
4. 保持与前文的连贯性
5. 语言生动，有画面感
"""
        
        response = await asyncio.to_thread(
            self.qwen_client.generate_scene_content,
            scene_prompt,
            character_context,
            world_context,
            plot_context,
            bridge_content,
            style_guide
        )
        
        if not response.success:
            self.logger.error(f"场景生成失败: {response.error_message}")
            return f"[场景生成失败: {task.title}]"
        
        scene_content = response.content
        
        if len(scene_content) > 500:  # 只对足够长的内容进行质量检查
            revised_content = await self._review_and_revise(scene_content, task)
            if revised_content:
                scene_content = revised_content
        
        task.status = "completed"
        self.current_progress.current_scene = task.scene
        
        return scene_content
    
    async def _build_bridge_context(self, current_chapter: int, current_scene: int) -> str:
        """构建桥接上下文"""
        if current_chapter == 1 and current_scene == 1:
            return "这是小说的开篇场景，需要引人入胜地开始故事。"
        
        related_scenes = self.memory.search_scenes(
            f"第{current_chapter}章", 
            top_k=self.bridge_config["context_scenes"]
        )
        
        if not related_scenes:
            return "继续前面的故事发展。"
        
        bridge_parts = ["【前情回顾】"]
        for scene_summary, score in related_scenes:
            bridge_parts.append(f"第{scene_summary.chapter}章第{scene_summary.scene}场景: {scene_summary.summary}")
        
        return "\n".join(bridge_parts)
    
    async def _build_character_context(self, task: SceneTask) -> str:
        """构建角色上下文"""
        if not task.characters_involved:
            characters = self.memory.search_characters(task.objective, top_k=3)
        else:
            characters = [(char, 1.0) for char in self.memory.search_characters(" ".join(task.characters_involved), top_k=3)]
        
        if not characters:
            return "请根据情节需要安排角色出场。"
        
        context_parts = ["【相关角色】"]
        for character, score in characters[:2]:  # 最多2个角色以控制长度
            context_parts.append(f"**{character.name}**: {character.description}")
            context_parts.append(f"性格: {character.personality}")
            if character.current_state:
                context_parts.append(f"当前状态: {character.current_state}")
            context_parts.append("")
        
        return "\n".join(context_parts)
    
    async def _build_world_context(self, task: SceneTask) -> str:
        """构建世界观上下文"""
        world_elements = self.memory.search_world_elements(task.objective, top_k=2)
        
        if not world_elements:
            return "请根据已建立的世界观进行描述。"
        
        context_parts = ["【世界观背景】"]
        for element, score in world_elements:
            context_parts.append(f"**{element.name}** ({element.type}): {element.description}")
        
        return "\n".join(context_parts)
    
    async def _build_plot_context(self, task: SceneTask) -> str:
        """构建情节上下文"""
        active_threads = self.memory.get_active_plot_threads()
        
        if not active_threads:
            return "请推进主要情节发展。"
        
        context_parts = ["【情节线索】"]
        for thread in active_threads[:3]:  # 最多3条线索
            context_parts.append(f"**{thread.name}**: {thread.description} (重要度: {thread.importance})")
        
        return "\n".join(context_parts)
    
    def _get_style_guide(self) -> str:
        """获取文体指南"""
        style_guides = {
            "fantasy": "采用仙侠玄幻风格，语言古雅而不失现代感，注重氛围营造和意境描写",
            "modern": "使用现代都市语言，贴近生活，对话自然，心理描写细腻",
            "scifi": "科幻风格，注重逻辑性和未来感，专业术语使用恰当",
            "historical": "古典文学风格，用词典雅，符合历史背景",
            "mystery": "悬疑风格，节奏紧凑，线索铺排巧妙",
            "romance": "情感细腻，语言温暖，注重情绪渲染"
        }
        
        return style_guides.get(self.current_config.style, "保持文学性和可读性")
    
    async def _review_and_revise(self, content: str, task: SceneTask) -> Optional[str]:
        """质量检查和修订"""
        review_criteria = f"""
评估标准：
1. 是否完成了场景目标：{task.objective}
2. 是否包含了关键事件：{', '.join(task.key_events)}
3. 语言是否流畅自然
4. 是否与前文连贯
5. 字数是否合适（目标{task.estimated_words}字左右）
"""
        
        response = await asyncio.to_thread(
            self.qwen_client.self_review_content,
            content,
            review_criteria
        )
        
        if not response.success:
            return None
        
        try:
            review_result = json.loads(response.content)
            overall_score = review_result.get("overall_score", 0)
            
            if overall_score >= self.quality_threshold:
                return None  # 无需修改
            
            revised_version = review_result.get("revised_version")
            if revised_version and len(revised_version) > len(content) * 0.5:
                self.logger.info(f"场景质量评分{overall_score}，使用修改版本")
                return revised_version
            
            return None
            
        except json.JSONDecodeError:
            return None
    
    async def _finalize_novel(self, novel_content: str) -> str:
        """完成小说最终处理"""
        chapters = re.findall(r'^## 第(\d+)章.*$', novel_content, re.MULTILINE)
        
        toc_parts = ["## 目录\n\n"]
        for chapter_num in chapters:
            toc_parts.append(f"第{chapter_num}章\n")
        
        toc = ''.join(toc_parts)
        
        intro_end = novel_content.find("---\n\n") + 5
        final_content = novel_content[:intro_end] + toc + "\n---\n\n" + novel_content[intro_end:]
        
        stats = self.memory.get_novel_stats()
        epilogue = f"""
---


《{self.current_config.title}》全文完。

**创作统计**:
- 总字数: 约{len(final_content):,}字
- 章节数: {stats['chapters_completed']}
- 场景数: {stats['scenes_written']}
- 角色数: {stats['characters']}
- 世界观元素: {stats['world_elements']}
- 情节线索: {stats['active_plot_threads']}

**技术说明**:
本小说采用RAG增强生成技术，使用Qwen-plus-latest模型，通过外部记忆系统突破上下文限制，
实现了长篇小说的连贯创作。系统运用了分层规划、桥接策略、质量控制等先进技术。

**质量保证**:
- 角色一致性: 通过角色卡片系统维护
- 世界观统一: 通过世界观圣经保证
- 情节连贯: 通过情节线索跟踪
- 语言质量: 通过自检修订机制

感谢您体验这次10万字RAG长文本生成测试！

---

*本作品由AI创作，展示了最新RAG技术在创意写作领域的应用*
"""
        
        final_content += epilogue
        return final_content
    
    def _extract_summary(self, content: str) -> str:
        """提取内容摘要"""
        clean_content = re.sub(r'[#*\-\n]+', ' ', content)
        clean_content = re.sub(r'\s+', ' ', clean_content).strip()
        return clean_content[:120] + "..." if len(clean_content) > 120 else clean_content
    
    def _create_basic_outline(self) -> Dict[str, Any]:
        """创建基础大纲（fallback）"""
        return {
            "overall_theme": "成长与冒险的故事",
            "main_conflicts": ["主角面临的主要挑战", "内心成长的困境", "外在环境的阻碍"],
            "character_arcs": "主角从弱小走向强大的成长历程",
            "chapters": [
                {
                    "title": f"第{i}章",
                    "summary": f"第{i}章的故事发展",
                    "key_events": [f"事件{i}-1", f"事件{i}-2"],
                    "conflicts": f"第{i}章的主要冲突"
                }
                for i in range(1, self.current_config.chapters_target + 1)
            ]
        }
    
    async def _update_progress(self, status: str, task: str, progress_callback):
        """更新进度"""
        if self.current_progress:
            self.current_progress.status = status
            self.current_progress.current_task = task
            self.current_progress.updated_at = datetime.now().isoformat()
            
            if progress_callback:
                await progress_callback(self.current_progress)
    
    async def _analyze_existing_content(self) -> Dict[str, Any]:
        """分析现有内容（简化版本）"""
        content = self.current_config.existing_content
        return {
            "title": "续写小说",
            "chapters_count": content.count("##"),
            "estimated_words": len(content),
            "main_characters": ["主角"],  # 简化版本
            "style": self.current_config.style
        }
    
    async def _extract_characters_and_world(self, analysis: Dict[str, Any]):
        """提取角色和世界观（简化版本）"""
        main_char = Character(
            name="主角",
            description="故事的主人公",
            appearance="",
            personality="",
            background="",
            motivation="",
            abilities=[],
            relationships={},
            speech_patterns=[],
            growth_arc="",
            current_state="",
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat()
        )
        self.memory.add_character(main_char)
    
    async def _plan_continuation(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """规划续写（简化版本）"""
        return {
            "continuation_chapters": 3,
            "new_conflicts": ["新的挑战"],
            "character_development": "角色进一步成长"
        }
    
    async def _write_continuation(self, outline: Dict[str, Any], progress_callback) -> str:
        """写作续写内容（简化版本）"""
        continuation_parts = ["\n\n---\n\n**续写部分开始**\n\n"]
        
        for i in range(1, 4):  # 续写3章
            await self._update_progress("writing", f"续写第{i}章", progress_callback)
            
            chapter_content = f"## 第{i}章 续写内容\n\n"
            chapter_content += "这里是续写的章节内容，基于已有情节继续发展故事...\n\n"
            
            continuation_parts.append(chapter_content)
        
        return ''.join(continuation_parts)
