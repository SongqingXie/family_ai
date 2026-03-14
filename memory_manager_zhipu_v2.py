"""
记忆管理器 V2 - 支持多用户独立存储
使用 FAISS 向量数据库 + JSON 文件存储
"""

import json
import os
import uuid
from datetime import datetime
from typing import List, Dict, Optional

import faiss
import numpy as np
from zhipuai import ZhipuAI


class ZhipuEmbedding:
    """智谱AI Embedding 封装"""
    
    def __init__(self, api_key: str, model: str = "embedding-3"):
        self.client = ZhipuAI(api_key=api_key)
        self.model = model
    
    def embed_query(self, text: str) -> List[float]:
        """获取单个文本的 Embedding"""
        response = self.client.embeddings.create(
            model=self.model,
            input=text
        )
        return response.data[0].embedding
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """获取多个文本的 Embedding"""
        response = self.client.embeddings.create(
            model=self.model,
            input=texts
        )
        return [item.embedding for item in response.data]


class MemoryManager:
    """
    记忆管理器 V2 - 支持多用户
    每个用户有独立的存储目录
    """
    
    def __init__(
        self,
        zhipu_api_key: str,
        embedding_model: str = "embedding-3",
        storage_dir: str = None,
        vector_dim: int = 2048,
        user_id: str = "default"
    ):
        """
        初始化记忆管理器
        
        Args:
            zhipu_api_key: 智谱AI API Key
            embedding_model: 使用的 Embedding 模型
            storage_dir: 存储目录路径（如果不指定，使用默认用户数据目录）
            vector_dim: 向量维度
            user_id: 用户ID，用于隔离不同用户的记忆
        """
        self.zhipu_api_key = zhipu_api_key
        self.embedding_model = embedding_model
        self.vector_dim = vector_dim
        self.user_id = user_id
        
        # 根据 user_id 设置存储目录
        if storage_dir is None:
            # 默认使用项目目录下的 user_data
            base_dir = "./familyAi/Langchainwithmemory/user_data"
            storage_dir = os.path.join(base_dir, f"user_{user_id}")
        
        self.storage_dir = storage_dir
        
        # 确保存储目录存在
        os.makedirs(storage_dir, exist_ok=True)
        
        # 文件路径
        self.json_path = os.path.join(storage_dir, "memories.json")
        self.faiss_path = os.path.join(storage_dir, "memories.faiss")
        self.mapping_path = os.path.join(storage_dir, "id_mapping.json")
        
        # 初始化 Embedding 模型
        self.embeddings = ZhipuEmbedding(
            api_key=zhipu_api_key,
            model=embedding_model
        )
        
        # 初始化 FAISS 索引
        self.index = self._init_faiss_index()
        
        # 加载或初始化数据
        self.memories = self._load_memories()
        self.id_mapping = self._load_id_mapping()
        
        # 检查并修复数据一致性
        self._sync_data()
        
        print(f"[MemoryManager] 用户 {user_id} 初始化完成")
        print(f"  - 存储目录: {storage_dir}")
        print(f"  - Embedding模型: {embedding_model}")
        print(f"  - 当前记忆数: {len(self.memories)}")
    
    def _init_faiss_index(self) -> faiss.Index:
        """初始化 FAISS 索引"""
        if os.path.exists(self.faiss_path):
            index = faiss.read_index(self.faiss_path)
            print(f"[FAISS] 加载已有索引，包含 {index.ntotal} 条向量")
            return index
        else:
            index = faiss.IndexFlatL2(self.vector_dim)
            print(f"[FAISS] 创建新索引")
            return index
    
    def _load_memories(self) -> Dict[str, Dict]:
        """从 JSON 文件加载记忆数据"""
        if os.path.exists(self.json_path):
            with open(self.json_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def _load_id_mapping(self) -> Dict[str, int]:
        """加载 ID 到 FAISS 索引的映射"""
        if os.path.exists(self.mapping_path):
            with open(self.mapping_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return {k: int(v) for k, v in data.items()}
        return {}
    
    def _save_memories(self):
        """保存记忆到 JSON 文件"""
        with open(self.json_path, 'w', encoding='utf-8') as f:
            json.dump(self.memories, f, ensure_ascii=False, indent=2)
    
    def _save_faiss_index(self):
        """保存 FAISS 索引"""
        faiss.write_index(self.index, self.faiss_path)
    
    def _save_id_mapping(self):
        """保存 ID 映射"""
        with open(self.mapping_path, 'w', encoding='utf-8') as f:
            json.dump(self.id_mapping, f, indent=2)
    
    def _get_embedding(self, text: str) -> List[float]:
        """使用智谱AI获取文本的 Embedding 向量"""
        if not text or not text.strip():
            raise ValueError("输入文本不能为空")
        
        embedding = self.embeddings.embed_query(text)
        return embedding
    
    def format_memory_template(
        self,
        content: str,
        category: str = "general",
        tags: List[str] = None
    ) -> Dict:
        """将原始内容整理成标准化的记忆模板"""
        if tags is None:
            tags = []
        
        memory_id = str(uuid.uuid4())[:8]
        timestamp = datetime.now().isoformat()
        
        memory_data = {
            "id": memory_id,
            "content": content.strip(),
            "category": category,
            "tags": tags,
            "created_at": timestamp,
            "updated_at": timestamp,
            "access_count": 0,
            "importance": 3,
            "embedding_model": self.embedding_model
        }
        
        return memory_data
    
    def add_memory(
        self,
        content: str,
        category: str = "general",
        tags: List[str] = None
    ) -> str:
        """
        添加新记忆
        
        流程：
        1. 将内容整理成模板格式
        2. 使用智谱AI Embedding 编码为向量
        3. 存储到 FAISS 索引
        4. 保存到 JSON 文件
        """
        try:
            # 检查是否已存在
            memory_data = self.format_memory_template(content, category, tags)
            memory_id = memory_data["id"]
            
            if memory_id in self.memories:
                print(f"  [警告] 记忆 {memory_id} 已存在，跳过添加")
                return memory_id
            
            # 获取 Embedding 向量
            print(f"  [Embedding] 正在编码: {content[:50]}...")
            embedding_vector = self._get_embedding(content)
            
            # 转换为 numpy 数组
            vector_np = np.array([embedding_vector], dtype=np.float32)
            
            # 添加到 FAISS 索引
            faiss_idx = self.index.ntotal
            self.index.add(vector_np)
            
            # 保存 ID 映射
            self.id_mapping[memory_id] = faiss_idx
            
            # 保存记忆数据
            self.memories[memory_id] = memory_data
            
            # 持久化到文件（确保顺序正确）
            try:
                self._save_memories()
                self._save_faiss_index()
                self._save_id_mapping()
                print(f"  [成功] 记忆已添加并持久化，ID: {memory_id}")
            except Exception as save_error:
                print(f"  [错误] 持久化失败: {save_error}")
                raise
            
            return memory_id
            
        except Exception as e:
            print(f"  [错误] 添加记忆失败: {str(e)}")
            raise
    
    def query_memory(self, query: str, top_k: int = 3) -> List[Dict]:
        """查询记忆（向量相似度搜索）"""
        if self.index.ntotal == 0:
            return []
        
        # 1. 获取查询的 Embedding
        query_vector = self._get_embedding(query)
        query_np = np.array([query_vector], dtype=np.float32)
        
        # 2. FAISS 搜索
        distances, indices = self.index.search(query_np, min(top_k, self.index.ntotal))
        
        # 3. 根据索引反查记忆 ID
        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            memory_id = None
            for mid, faiss_idx in self.id_mapping.items():
                if faiss_idx == idx:
                    memory_id = mid
                    break
            
            if not memory_id:
                continue
            
            # 检查是否在 memories 中
            if memory_id in self.memories:
                memory = self.memories[memory_id].copy()
                memory["id"] = memory_id
                memory["similarity_score"] = float(1 / (1 + dist))
                memory["data_status"] = "正常"
                results.append(memory)
            else:
                # 异常情况：ID 只在 id_mapping 中，不在 memories 中
                placeholder = {
                    "id": memory_id,
                    "content": "[数据异常：该记忆ID存在但内容丢失]",
                    "category": "unknown",
                    "tags": [],
                    "created_at": "unknown",
                    "date_label": "unknown",
                    "similarity_score": float(1 / (1 + dist)),
                    "data_status": "内容丢失",
                    "embedding_model": self.embedding_model
                }
                results.append(placeholder)
                print(f"  [警告] 查询到ID '{memory_id}' 但内容丢失，创建占位符")
        
        return results
    
    def delete_memory(self, memory_id: str) -> bool:
        """删除记忆（物理删除并重建索引）"""
        # 先去除可能的空白字符
        memory_id = memory_id.strip()
        
        print(f"  [删除] 尝试删除记忆 ID: '{memory_id}'")
        print(f"  [删除] 当前 memories 中有 {len(self.memories)} 条")
        print(f"  [删除] 当前 id_mapping 中有 {len(self.id_mapping)} 条")
        
        found = False
        
        # 1. 检查是否在 memories 中（物理删除，而非仅标记）
        if memory_id in self.memories:
            print(f"  [删除] 在 memories 中找到 ID，执行物理删除")
            del self.memories[memory_id]
            found = True
        
        # 2. 检查是否在 id_mapping 中（即使不在 memories 中）
        if memory_id in self.id_mapping:
            print(f"  [删除] 在 id_mapping 中找到 ID，移除映射")
            del self.id_mapping[memory_id]
            self._save_id_mapping()
            found = True
        
        if found:
            # 统一重建，确保 FAISS / id_mapping 与 memories 严格一致
            self._save_memories()
            self._rebuild_all_vectors()
            print(f"  [成功] 记忆 {memory_id} 已删除")
            return True
        else:
            print(f"  [删除] 错误: ID '{memory_id}' 未找到")
            print(f"  [删除] 可用 memories IDs: {list(self.memories.keys())}")
            print(f"  [删除] 可用 mapping IDs: {list(self.id_mapping.keys())}")
            
            # 尝试模糊匹配
            for key in self.memories.keys():
                if key.lower() == memory_id.lower():
                    print(f"  [删除] 发现大小写不同的匹配: '{key}'")
                    return self.delete_memory(key)
            
            return False
    
    def get_all_memories(self) -> List[Dict]:
        """获取所有未删除的记忆"""
        memories = []
        for mid, mem in self.memories.items():
            if not mem.get("deleted", False):
                mem_copy = mem.copy()
                mem_copy['id'] = mid
                memories.append(mem_copy)
        return memories
    
    def get_stats(self) -> Dict:
        """获取记忆库统计信息"""
        total = len(self.memories)
        deleted = sum(1 for m in self.memories.values() if m.get("deleted", False))
        active = total - deleted
        
        return {
            "total_memories": total,
            "active_memories": active,
            "deleted_memories": deleted,
            "vector_count": self.index.ntotal,
            "embedding_model": self.embedding_model
        }
    
    def _sync_data(self):
        """检查并修复 FAISS 索引、memories 和 id_mapping 之间的一致性"""
        print(f"[Sync] 检查数据一致性...")
        
        # 1. 检查 id_mapping 中有多余的 ID（不在 memories 中）
        ids_only_in_mapping = set(self.id_mapping.keys()) - set(self.memories.keys())
        if ids_only_in_mapping:
            print(f"[Sync] 警告: id_mapping 中有 {len(ids_only_in_mapping)} 个 ID 不在 memories 中")
            for mid in list(ids_only_in_mapping):
                print(f"[Sync] 从 id_mapping 中移除: {mid}")
                del self.id_mapping[mid]
            self._save_id_mapping()
        
        # 2. 检查 memories 中有多余的 ID（不在 id_mapping 中）
        # 仅处理未删除数据，避免把历史软删除数据重建回向量索引
        active_memory_ids = {
            mid for mid, mem in self.memories.items()
            if not mem.get("deleted", False)
        }
        ids_only_in_memories = active_memory_ids - set(self.id_mapping.keys())
        if ids_only_in_memories:
            print(f"[Sync] 警告: memories 中有 {len(ids_only_in_memories)} 个 ID 不在 id_mapping 中")
            print(f"[Sync] 正在重建这些记忆的向量索引...")
            self._rebuild_missing_vectors(ids_only_in_memories)
        
        # 3. 检查 FAISS 索引数量是否匹配
        faiss_count = self.index.ntotal
        mapping_count = len(self.id_mapping)
        memories_count = len([m for m in self.memories.values() if not m.get("deleted", False)])
        
        if faiss_count != mapping_count or faiss_count != memories_count:
            print(f"[Sync] 警告: 数据不一致 - FAISS({faiss_count}), id_mapping({mapping_count}), memories({memories_count})")
            
            # 如果 FAISS 数量远大于实际记忆数，说明有重复，需要完全重建
            if faiss_count > memories_count * 1.5:
                print(f"[Sync] FAISS 索引数量异常，执行完全重建...")
                self._rebuild_all_vectors()
            elif faiss_count != mapping_count:
                print(f"[Sync] 尝试修复索引映射关系...")
                self._fix_faiss_mapping_mismatch()
        
        print(f"[Sync] 同步完成 - Memories: {len(self.memories)}, ID Mapping: {len(self.id_mapping)}, FAISS: {self.index.ntotal}")
    
    def _rebuild_missing_vectors(self, missing_ids: set):
        """为缺失向量的记忆重建向量"""
        print(f"[Sync] 正在为 {len(missing_ids)} 条记忆重建向量...")
        
        # 如果 FAISS 索引数量远大于预期，可能需要清理重复
        if self.index.ntotal > len(self.memories) + len(missing_ids):
            print(f"[Sync] 警告: FAISS 索引({self.index.ntotal})远大于记忆数({len(self.memories)})，建议完全重建")
            self._rebuild_all_vectors()
            return
        
        rebuilt_count = 0
        for mid in missing_ids:
            if mid not in self.memories:
                continue
            
            # 检查是否已经在 id_mapping 中（可能刚添加）
            if mid in self.id_mapping:
                print(f"  [跳过] 记忆 {mid} 已存在映射")
                continue
            
            memory = self.memories[mid]
            if memory.get("deleted", False):
                print(f"[Sync] 跳过已删除记忆: {mid}")
                continue
            content = memory.get("content", "")
            
            if not content:
                print(f"[Sync] 跳过空内容记忆: {mid}")
                continue
            
            try:
                # 获取 Embedding
                print(f"  [Embedding] 重建向量: {content[:40]}...")
                embedding_vector = self._get_embedding(content)
                vector_np = np.array([embedding_vector], dtype=np.float32)
                
                # 添加到 FAISS
                faiss_idx = self.index.ntotal
                self.index.add(vector_np)
                
                # 更新 id_mapping
                self.id_mapping[mid] = faiss_idx
                
                rebuilt_count += 1
                print(f"  [成功] 已重建: {mid} -> index {faiss_idx}")
                
            except Exception as e:
                print(f"  [错误] 重建失败 {mid}: {e}")
        
        # 保存更新后的数据
        if rebuilt_count > 0:
            self._save_faiss_index()
            self._save_id_mapping()
            print(f"[Sync] 成功重建 {rebuilt_count} 条记忆的向量")
        else:
            print(f"[Sync] 没有需要重建的向量")
    
    def _rebuild_all_vectors(self):
        """完全重建所有向量索引（清理重复）"""
        print(f"[Sync] 正在完全重建向量索引...")
        
        # 1. 创建新的空索引
        new_index = faiss.IndexFlatL2(self.vector_dim)
        new_id_mapping = {}
        
        # 2. 遍历所有未删除的记忆，重新添加
        count = 0
        for mid, memory in self.memories.items():
            if memory.get("deleted", False):
                continue
            
            content = memory.get("content", "")
            if not content:
                continue
            
            try:
                embedding_vector = self._get_embedding(content)
                vector_np = np.array([embedding_vector], dtype=np.float32)
                
                faiss_idx = new_index.ntotal
                new_index.add(vector_np)
                new_id_mapping[mid] = faiss_idx
                count += 1
            except Exception as e:
                print(f"  [错误] 重建 {mid} 失败: {e}")
        
        # 3. 替换旧索引
        self.index = new_index
        self.id_mapping = new_id_mapping
        
        # 4. 保存
        self._save_faiss_index()
        self._save_id_mapping()
        
        print(f"[Sync] 完成重建，共 {count} 条记忆")
    
    def _fix_faiss_mapping_mismatch(self):
        """修复 FAISS 和 id_mapping 数量不匹配问题"""
        print(f"[Sync] 修复索引映射不匹配...")
        
        # 如果 id_mapping 少但 FAISS 多，可能是重建导致的重复
        # 最简单的修复是：以 memories 为准，完全重建
        if self.index.ntotal > len(self.id_mapping):
            print(f"[Sync] FAISS({self.index.ntotal}) > id_mapping({len(self.id_mapping)})，执行重建")
            self._rebuild_all_vectors()
        elif self.index.ntotal < len(self.id_mapping):
            # FAISS 向量少，可能是数据丢失，也需要重建
            print(f"[Sync] FAISS({self.index.ntotal}) < id_mapping({len(self.id_mapping)})，执行重建")
            self._rebuild_all_vectors()


# 全局记忆管理器缓存（按 user_id 缓存）
_memory_manager_instances: Dict[str, MemoryManager] = {}


def get_memory_manager(
    zhipu_api_key: str = None,
    embedding_model: str = "embedding-3",
    user_id: str = "default"
) -> MemoryManager:
    """
    获取记忆管理器实例（单例模式，按 user_id 区分）
    
    Args:
        zhipu_api_key: 智谱AI API Key
        embedding_model: Embedding 模型
        user_id: 用户ID，不同用户有独立的记忆库
        
    Returns:
        MemoryManager 实例
    """
    global _memory_manager_instances
    
    cache_key = f"{user_id}_{embedding_model}"
    
    if cache_key not in _memory_manager_instances:
        if zhipu_api_key is None:
            raise ValueError("首次初始化需要提供 zhipu_api_key")
        
        _memory_manager_instances[cache_key] = MemoryManager(
            zhipu_api_key=zhipu_api_key,
            embedding_model=embedding_model,
            user_id=user_id
        )
    
    return _memory_manager_instances[cache_key]
