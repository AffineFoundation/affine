"""
Scorer API Client

专用于 Scorer 的 API 客户端，从 API 层获取采样数据和 Miner 信息
支持 Task ID 范围查询，便于数据集扩展过渡期
"""

import aiohttp
import logging
from typing import Dict, List, Any, Optional

from .models import MinerInfo, SampleStatistics, CompletionStatus, TaskIdRange

logger = logging.getLogger(__name__)


class ScorerAPIClient:
    """
    Scorer 专用 API 客户端
    
    职责:
    - 获取活跃 Miner 列表
    - 获取 Miner 采样结果（支持 Task ID 范围）
    - 保存权重快照
    - 获取系统配置
    """
    
    def __init__(self, base_url: str, timeout_seconds: int = 60):
        """
        初始化 API 客户端
        
        Args:
            base_url: API 基础 URL
            timeout_seconds: 请求超时时间
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = aiohttp.ClientTimeout(total=timeout_seconds)
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """获取或创建 HTTP 会话"""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(timeout=self.timeout)
        return self.session
    
    async def get_active_miners(self) -> List[MinerInfo]:
        """
        获取所有活跃 Miner
        
        Returns:
            MinerInfo 列表
        """
        session = await self._get_session()
        
        try:
            async with session.get(f"{self.base_url}/api/v1/miners/active") as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    logger.error(f"Failed to get active miners: {resp.status} - {error_text}")
                    raise Exception(f"Failed to get active miners: {error_text}")
                
                data = await resp.json()
                miners = data.get("miners", data) if isinstance(data, dict) else data
                
                result = []
                for miner_data in miners:
                    try:
                        result.append(MinerInfo.from_dict(miner_data))
                    except Exception as e:
                        logger.warning(f"Failed to parse miner data: {miner_data}, error: {e}")
                
                logger.info(f"Fetched {len(result)} active miners")
                return result
        
        except aiohttp.ClientError as e:
            logger.error(f"HTTP error fetching active miners: {e}")
            raise
    
    async def get_miner_samples(
        self,
        hotkey: str,
        revision: str,
        env: str,
        dataset_length: int,
        task_id_range: Optional[TaskIdRange] = None,
        dedupe_by_task_id: bool = True
    ) -> Dict[str, Any]:
        """
        获取 Miner 采样结果 (使用新的 Scorer 专用端点)
        
        Args:
            hotkey: Miner hotkey
            revision: 模型版本
            env: 环境名称
            dataset_length: 数据集长度
            task_id_range: Task ID 范围 (用于支持数据集扩展过渡)
            dedupe_by_task_id: 是否按 task_id 去重
        
        Returns:
            {
                "samples": [...],
                "is_complete": bool,
                "completed_count": int,
                "total_count": int,
                "missing_task_ids": [...]
            }
        """
        session = await self._get_session()
        
        params = {
            "model_revision": revision,
            "env": env,
            "dataset_length": dataset_length,
            "deduplicate": str(dedupe_by_task_id).lower()
        }
        
        # 添加 Task ID 范围参数
        if task_id_range is not None and task_id_range.length > 0:
            params["task_id_start"] = task_id_range.start
            params["task_id_end"] = task_id_range.end
        
        try:
            async with session.get(
                f"{self.base_url}/api/v1/samples/scorer/{hotkey}",
                params=params
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    logger.error(
                        f"Failed to get samples for {hotkey}/{env}: {resp.status} - {error_text}"
                    )
                    raise Exception(f"Failed to get miner samples: {error_text}")
                
                data = await resp.json()
                
                # 直接返回响应数据 (已经是正确格式)
                result = {
                    "samples": data.get("samples", []),
                    "is_complete": data.get("is_complete", False),
                    "completed_count": data.get("completed_count", 0),
                    "total_count": data.get("total_count", 0),
                    "missing_task_ids": data.get("missing_task_ids", [])
                }
                
                return result
        
        except aiohttp.ClientError as e:
            logger.error(f"HTTP error fetching samples for {hotkey}/{env}: {e}")
            raise
    
    async def get_dataset_info(self, env: str) -> Dict[str, Any]:
        """
        获取数据集信息（包括 task_id 范围）
        
        Args:
            env: 环境名称
        
        Returns:
            {
                "env": str,
                "dataset_length": int,
                "task_id_start": int,
                "task_id_end": int
            }
        """
        session = await self._get_session()
        
        try:
            async with session.get(
                f"{self.base_url}/api/v1/config/dataset/{env}"
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    logger.error(f"Failed to get dataset info for {env}: {error_text}")
                    raise Exception(f"Failed to get dataset info: {error_text}")
                
                data = await resp.json()
                return data
        
        except aiohttp.ClientError as e:
            logger.error(f"HTTP error fetching dataset info for {env}: {e}")
            raise
    
    async def save_weights(
        self,
        weights: Dict[str, float],
        calculation_details: Dict[str, Any],
        block_number: int
    ) -> str:
        """
        保存权重快照
        
        Args:
            weights: hotkey -> weight 映射
            calculation_details: 计算详情
            block_number: 当前区块号
        
        Returns:
            snapshot_id
        """
        session = await self._get_session()
        
        payload = {
            "scorer_hotkey": "scorer_service",  # 系统服务标识
            "weights": weights,
            "calculation_details": calculation_details,
            "block_number": block_number
        }
        
        try:
            async with session.post(
                f"{self.base_url}/api/v1/scores/weights",
                json=payload
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    logger.error(f"Failed to save weights: {error_text}")
                    raise Exception(f"Failed to save weights: {error_text}")
                
                data = await resp.json()
                snapshot_id = data.get("snapshot_id", "unknown")
                logger.info(f"Saved weight snapshot: {snapshot_id}")
                return snapshot_id
        
        except aiohttp.ClientError as e:
            logger.error(f"HTTP error saving weights: {e}")
            raise
    
    async def get_current_block(self) -> int:
        """
        获取当前区块号
        
        Returns:
            当前区块号，失败时返回 0
        """
        session = await self._get_session()
        
        try:
            async with session.get(f"{self.base_url}/api/v1/chain/block") as resp:
                if resp.status != 200:
                    logger.warning("Failed to get current block, returning 0")
                    return 0
                
                data = await resp.json()
                return data.get("block_number", 0)
        
        except Exception as e:
            logger.warning(f"Error getting current block: {e}, returning 0")
            return 0
    
    async def close(self):
        """关闭 HTTP 会话"""
        if self.session and not self.session.closed:
            await self.session.close()
            self.session = None