"""
外部记忆系统 - 解决10万字小说生成的上下文限制问题
支持角色卡、世界观、时间线、伏笔、场景摘要的存储与检索
（这个文件是用来后续开发10万字文档用的，我只是用小说来测试而已）
"""

import json
import sqlite3
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import os

@dataclass
class Character:
    """角色卡片"""
    name: str
    description: str
    appearance: str
    personality: str
    background: str
    motivation: str
    abilities: List[str]
    relationships: Dict[str, str]  # 其他角色关系
    speech_patterns: List[str]  # 口癖、说话方式
    growth_arc: str  # 成长弧线
    current_state: str  # 当前状态
    created_at: str
    updated_at: str
    
@dataclass 
class WorldElement:
    """世界观元素"""
    name: str
    type: str  # location, organization, technology, magic_system, etc.
    description: str
    properties: Dict[str, Any]
    relationships: Dict[str, str]
    rules: List[str]  # 相关规则/限制
    created_at: str
    updated_at: str

@dataclass
class PlotThread:
    """情节线索"""
    id: str
    name: str
    description: str
    status: str  # pending, active, resolved
    importance: int  # 1-5, 5最重要
    planted_chapter: int  # 埋伏笔的章节
    must_resolve_by: Optional[int]  # 必须在第几章前解决
    related_characters: List[str]
    related_elements: List[str]
    created_at: str
    updated_at: str

@dataclass
class SceneSummary:
    """场景摘要"""
    chapter: int
    scene: int
    title: str
    summary: str  # 80-120字摘要
    key_events: List[str]
    characters_present: List[str]
    plot_threads_touched: List[str]
    emotional_tone: str
    word_count: int
    created_at: str

@dataclass
class TimelineEvent:
    """时间线事件"""
    chapter: int
    scene: int
    event: str
    characters_involved: List[str]
    consequences: List[str]
    timestamp: str  # 故事内时间
    created_at: str

class ExternalMemory:
    """外部记忆管理系统"""
    
    def __init__(self, memory_dir: str = "./novel_memory"):
        self.memory_dir = memory_dir
        os.makedirs(memory_dir, exist_ok=True)
        
        self.db_path = os.path.join(memory_dir, "novel_memory.db")
        self._init_database()
        
        self.embedding_model = None
        self.vector_dim = 1024  # bge-m3的向量维度
        try:
            self.embedding_model = SentenceTransformer('BAAI/bge-m3')
        except Exception as e:
            print(f"[ExternalMemory] 向量模型加载失败，将禁用检索: {e}")
        
        self.scene_index = faiss.IndexFlatIP(self.vector_dim)  # 场景摘要向量
        self.character_index = faiss.IndexFlatIP(self.vector_dim)  # 角色描述向量
        self.world_index = faiss.IndexFlatIP(self.vector_dim)  # 世界观向量
        
        self.scene_id_to_summary = {}
        self.character_id_to_card = {}
        self.world_id_to_element = {}
        
        self._load_existing_data()
    
    def _init_database(self):
        """初始化数据库表"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS characters (
                name TEXT PRIMARY KEY,
                data TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS world_elements (
                name TEXT PRIMARY KEY,
                type TEXT NOT NULL,
                data TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS plot_threads (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                status TEXT NOT NULL,
                importance INTEGER NOT NULL,
                data TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS scene_summaries (
                id TEXT PRIMARY KEY,
                chapter INTEGER NOT NULL,
                scene INTEGER NOT NULL,
                data TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS timeline_events (
                id TEXT PRIMARY KEY,
                chapter INTEGER NOT NULL,
                scene INTEGER NOT NULL,
                event TEXT NOT NULL,
                data TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def _load_existing_data(self):
        """加载已存在的数据到向量索引"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT id, data FROM scene_summaries")
        for row in cursor.fetchall():
            summary_id, data_json = row
            summary = SceneSummary(**json.loads(data_json))
            self.scene_id_to_summary[summary_id] = summary
            
            if self.embedding_model is not None:
                text = f"{summary.title} {summary.summary}"
                vector = self.embedding_model.encode([text])[0]
                self.scene_index.add(np.array([vector], dtype=np.float32))
        
        cursor.execute("SELECT name, data FROM characters")
        for row in cursor.fetchall():
            name, data_json = row
            character = Character(**json.loads(data_json))
            self.character_id_to_card[name] = character
            
            if self.embedding_model is not None:
                text = f"{character.name} {character.description} {character.personality}"
                vector = self.embedding_model.encode([text])[0]
                self.character_index.add(np.array([vector], dtype=np.float32))
        
        cursor.execute("SELECT name, data FROM world_elements")
        for row in cursor.fetchall():
            name, data_json = row
            element = WorldElement(**json.loads(data_json))
            self.world_id_to_element[name] = element
            
            if self.embedding_model is not None:
                text = f"{element.name} {element.description}"
                vector = self.embedding_model.encode([text])[0]
                self.world_index.add(np.array([vector], dtype=np.float32))
        
        conn.close()
    
    def add_character(self, character: Character):
        """添加角色"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        data_json = json.dumps(asdict(character), ensure_ascii=False, indent=2)
        cursor.execute('''
            INSERT OR REPLACE INTO characters (name, data, created_at, updated_at)
            VALUES (?, ?, ?, ?)
        ''', (character.name, data_json, character.created_at, character.updated_at))
        
        conn.commit()
        conn.close()
        
        self.character_id_to_card[character.name] = character
        text = f"{character.name} {character.description} {character.personality}"
        vector = self.embedding_model.encode([text])[0]
        self.character_index.add(np.array([vector], dtype=np.float32))
    
    def add_world_element(self, element: WorldElement):
        """添加世界观元素"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        data_json = json.dumps(asdict(element), ensure_ascii=False, indent=2)
        cursor.execute('''
            INSERT OR REPLACE INTO world_elements (name, type, data, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
        ''', (element.name, element.type, data_json, element.created_at, element.updated_at))
        
        conn.commit()
        conn.close()
        
        self.world_id_to_element[element.name] = element
        text = f"{element.name} {element.description}"
        vector = self.embedding_model.encode([text])[0]
        self.world_index.add(np.array([vector], dtype=np.float32))
    
    def add_plot_thread(self, thread: PlotThread):
        """添加情节线索"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        data_json = json.dumps(asdict(thread), ensure_ascii=False, indent=2)
        cursor.execute('''
            INSERT OR REPLACE INTO plot_threads (id, name, status, importance, data, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (thread.id, thread.name, thread.status, thread.importance, data_json, thread.created_at, thread.updated_at))
        
        conn.commit()
        conn.close()
    
    def add_scene_summary(self, summary: SceneSummary):
        """添加场景摘要"""
        summary_id = f"ch{summary.chapter}_sc{summary.scene}"
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        data_json = json.dumps(asdict(summary), ensure_ascii=False, indent=2)
        cursor.execute('''
            INSERT OR REPLACE INTO scene_summaries (id, chapter, scene, data, created_at)
            VALUES (?, ?, ?, ?, ?)
        ''', (summary_id, summary.chapter, summary.scene, data_json, summary.created_at))
        
        conn.commit()
        conn.close()
        
        self.scene_id_to_summary[summary_id] = summary
        text = f"{summary.title} {summary.summary}"
        vector = self.embedding_model.encode([text])[0]
        self.scene_index.add(np.array([vector], dtype=np.float32))
    
    def add_timeline_event(self, event: TimelineEvent):
        """添加时间线事件"""
        event_id = f"ch{event.chapter}_sc{event.scene}_{hashlib.md5(event.event.encode()).hexdigest()[:8]}"
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        data_json = json.dumps(asdict(event), ensure_ascii=False, indent=2)
        cursor.execute('''
            INSERT OR REPLACE INTO timeline_events (id, chapter, scene, event, data, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (event_id, event.chapter, event.scene, event.event, data_json, event.created_at))
        
        conn.commit()
        conn.close()
    
    def search_scenes(self, query: str, top_k: int = 5) -> List[Tuple[SceneSummary, float]]:
        """搜索相关场景"""
        if self.embedding_model is None or self.scene_index.ntotal == 0:
            return []
        
        query_vector = self.embedding_model.encode([query])[0]
        query_vector = np.array([query_vector], dtype=np.float32)
        
        scores, indices = self.scene_index.search(query_vector, min(top_k, self.scene_index.ntotal))
        
        results = []
        scene_list = list(self.scene_id_to_summary.values())
        
        for score, idx in zip(scores[0], indices[0]):
            if 0 <= idx < len(scene_list):
                results.append((scene_list[idx], float(score)))
        
        return results
    
    def search_characters(self, query: str, top_k: int = 3) -> List[Tuple[Character, float]]:
        """搜索相关角色"""
        if self.embedding_model is None or self.character_index.ntotal == 0:
            return []
        
        query_vector = self.embedding_model.encode([query])[0]
        query_vector = np.array([query_vector], dtype=np.float32)
        
        scores, indices = self.character_index.search(query_vector, min(top_k, self.character_index.ntotal))
        
        results = []
        character_list = list(self.character_id_to_card.values())
        
        for score, idx in zip(scores[0], indices[0]):
            if 0 <= idx < len(character_list):
                results.append((character_list[idx], float(score)))
        
        return results
    
    def search_world_elements(self, query: str, top_k: int = 3) -> List[Tuple[WorldElement, float]]:
        """搜索相关世界观元素"""
        if self.embedding_model is None or self.world_index.ntotal == 0:
            return []
        
        query_vector = self.embedding_model.encode([query])[0]
        query_vector = np.array([query_vector], dtype=np.float32)
        
        scores, indices = self.world_index.search(query_vector, min(top_k, self.world_index.ntotal))
        
        results = []
        world_list = list(self.world_id_to_element.values())
        
        for score, idx in zip(scores[0], indices[0]):
            if 0 <= idx < len(world_list):
                results.append((world_list[idx], float(score)))
        
        return results
    
    def get_active_plot_threads(self, importance_threshold: int = 2) -> List[PlotThread]:
        """获取活跃的情节线索"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT data FROM plot_threads 
            WHERE status IN ('pending', 'active') AND importance >= ?
            ORDER BY importance DESC, created_at ASC
        ''', (importance_threshold,))
        
        threads = []
        for row in cursor.fetchall():
            data_json = row[0]
            thread = PlotThread(**json.loads(data_json))
            threads.append(thread)
        
        conn.close()
        return threads
    
    def get_recent_timeline(self, current_chapter: int, lookback_chapters: int = 3) -> List[TimelineEvent]:
        """获取最近的时间线事件"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT data FROM timeline_events 
            WHERE chapter >= ? AND chapter <= ?
            ORDER BY chapter DESC, scene DESC
        ''', (max(1, current_chapter - lookback_chapters), current_chapter))
        
        events = []
        for row in cursor.fetchall():
            data_json = row[0]
            event = TimelineEvent(**json.loads(data_json))
            events.append(event)
        
        conn.close()
        return events
    
    def get_novel_stats(self) -> Dict[str, Any]:
        """获取小说统计信息"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM characters")
        char_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM world_elements")
        world_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM plot_threads WHERE status != 'resolved'")
        active_threads = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM scene_summaries")
        scene_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT MAX(chapter) FROM scene_summaries")
        max_chapter = cursor.fetchone()[0] or 0
        
        cursor.execute("SELECT SUM(json_extract(data, '$.word_count')) FROM scene_summaries")
        total_words = cursor.fetchone()[0] or 0
        
        conn.close()
        
        return {
            "characters": char_count,
            "world_elements": world_count,
            "active_plot_threads": active_threads,
            "scenes_written": scene_count,
            "chapters_completed": max_chapter,
            "total_words": total_words
        }
