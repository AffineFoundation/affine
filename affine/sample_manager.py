#!/usr/bin/env python3
import os
import json
import uuid
import logging
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, TYPE_CHECKING
from dataclasses import dataclass, asdict, field
from datetime import datetime
from . import val_config

if TYPE_CHECKING:
    import affine as af

logger = logging.getLogger("affine.sample_manager")

@dataclass
class Sample:
    """Represents a pre-generated sample for an environment.
    Only stores core metadata (id, env_name, created_at) and the challenge data."""
    id: str
    env_name: str
    created_at: str
    data: Dict[str, Any] = field(default_factory=dict)  # Contains all challenge data (prompt + extra)
    
    @property
    def prompt(self) -> str:
        """Get the prompt from data"""
        return self.data.get("prompt", "")
    
    @property
    def extra(self) -> Dict[str, Any]:
        """Get everything except prompt as extra"""
        data = self.data.copy()
        data.pop("prompt", None)  # Remove prompt if it exists
        return data
    
    @classmethod
    def from_challenge(cls, challenge: Any, env_name: str) -> "Sample":
        """Create a sample from a challenge"""
        # Combine prompt and extra into a single data dict
        data = {"prompt": challenge.prompt}
        data.update(challenge.extra)
        
        return cls(
            id=str(uuid.uuid4()),
            env_name=env_name,
            created_at=datetime.now().isoformat(),
            data=data
        )
    
    def to_challenge(self, env: Any) -> Any:
        """Convert sample back to a challenge"""
        # Import here to avoid circular import
        import affine as af
        
        # Extract prompt and remaining data becomes extra
        data = self.data.copy()
        prompt = data.pop("prompt")
        
        return af.Challenge(
            env=env,
            prompt=prompt,
            extra=data
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "id": self.id,
            "env_name": self.env_name,
            "created_at": self.created_at,
            "data": self.data
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Sample":
        """Create Sample from dictionary, handling both old and new formats"""
        if "data" in data:
            # New format
            return cls(**data)
        else:
            # Old format - combine prompt and extra into data
            sample_data = {}
            if "prompt" in data:
                sample_data["prompt"] = data["prompt"]
            if "extra" in data:
                sample_data.update(data["extra"])
            
            return cls(
                id=data["id"],
                env_name=data["env_name"],
                created_at=data["created_at"],
                data=sample_data
            )

@dataclass 
class SampleStats:
    """Statistics about samples for an environment"""
    total: int = 0
    target_stock: int = val_config.TARGET_STOCK

class SampleManager:
    """Manages pre-generated samples for environments"""
    
    def __init__(self, base_dir: Optional[str] = None):
        self.base_dir = Path(base_dir or os.path.expanduser("~/.affine/samples"))
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
    def _get_sample_file(self, env_name: str) -> Path:
        """Get the JSONL file path for an environment"""
        return self.base_dir / f"{env_name.lower()}.jsonl"
    
    def _load_samples(self, env_name: str) -> List[Sample]:
        """Load all samples for an environment from JSONL file"""
        sample_file = self._get_sample_file(env_name)
        if not sample_file.exists():
            return []
        
        samples = []
        try:
            with open(sample_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data = json.loads(line)
                        samples.append(Sample.from_dict(data))
        except Exception as e:
            logger.error(f"Error loading samples for {env_name}: {e}")
            return []
        
        return samples
    
    def _save_samples(self, env_name: str, samples: List[Sample]) -> None:
        """Save samples to JSONL file"""
        sample_file = self._get_sample_file(env_name)
        try:
            with open(sample_file, 'w') as f:
                for sample in samples:
                    f.write(json.dumps(sample.to_dict()) + '\n')
        except Exception as e:
            logger.error(f"Error saving samples for {env_name}: {e}")
            raise
    
    def add_samples(self, env_name: str, samples: List[Sample]) -> None:
        """Add new samples to the environment's collection"""
        existing_samples = self._load_samples(env_name)
        existing_samples.extend(samples)
        self._save_samples(env_name, existing_samples)
        logger.info(f"Added {len(samples)} samples for {env_name}")
    
    def get_samples(self, env_name: str, count: int) -> List[Sample]:
        """Get up to 'count' samples for an environment"""
        samples = self._load_samples(env_name)
        
        if len(samples) < count:
            logger.warning(f"Requested {count} samples for {env_name}, but only {len(samples)} available")
        
        return samples[:count]
    
    def remove_samples(self, env_name: str, sample_ids: List[str]) -> None:
        """Remove specific samples by ID (delete used samples)"""
        samples = self._load_samples(env_name)
        sample_ids_set = set(sample_ids)
        
        # Keep only samples that are not in the removal list
        remaining_samples = [s for s in samples if s.id not in sample_ids_set]
        removed_count = len(samples) - len(remaining_samples)
        
        self._save_samples(env_name, remaining_samples)
        logger.info(f"Removed {removed_count} samples for {env_name}")
    
    def get_stats(self, env_name: str) -> SampleStats:
        """Get statistics about samples for an environment"""
        samples = self._load_samples(env_name)
        total = len(samples)
        
        return SampleStats(total=total)
    
    def get_all_stats(self) -> Dict[str, SampleStats]:
        """Get statistics for all environments"""
        stats = {}
        for sample_file in self.base_dir.glob("*.jsonl"):
            env_name = sample_file.stem.upper()
            stats[env_name] = self.get_stats(env_name)
        return stats
    
    async def generate_samples_for_env(self, env_instance, n: int, max_concurrent: int = val_config.MAX_CONCURRENT_REQUESTS) -> List[Sample]:
        """Generate n samples for an environment instance with parallel generation.
        Will attempt up to 5x the requested number of samples to get n successful ones."""
        env_name = env_instance.__class__.__name__
        logger.debug(f"Generating {n} samples for {env_name} with max concurrency {max_concurrent}")
        
        max_attempts = n * val_config.MAX_GENERATION_ATTEMPTS_MULTIPLIER  # Try up to 5x the requested number
        successful_samples = []
        total_attempts = 0
        
        while len(successful_samples) < n and total_attempts < max_attempts:
            # Calculate how many more samples we need
            remaining = n - len(successful_samples)
            # Try to generate twice the remaining needed samples each iteration
            attempt_count = min(remaining * 2, max_attempts - total_attempts)
            
            logger.debug(f"Attempting to generate {attempt_count} samples (have {len(successful_samples)}/{n} successful)")
            
            # Check if environment supports batch generation
            if hasattr(env_instance, 'generate_batch'):
                logger.debug(f"Using batch generation for {env_name}")
                challenges = await env_instance.generate_batch(attempt_count)
                batch_samples = [
                    Sample.from_challenge(challenge, env_name)
                    for challenge in challenges
                ]
                successful_samples.extend(batch_samples)
                total_attempts += attempt_count
                
            else:
                # Fall back to individual generation with semaphore
                logger.debug(f"Using individual generation for {env_name}")
                async def generate_single_sample() -> Optional[Sample]:
                    """Generate a single sample, return None if failed"""
                    try:
                        challenge = await env_instance.generate()
                        return Sample.from_challenge(challenge, env_name)
                    except Exception as e:
                        logger.debug(f"Failed to generate sample: {str(e)}")
                        return None
                
                # Generate samples in parallel with concurrency limit
                semaphore = asyncio.Semaphore(max_concurrent)
                
                async def generate_with_semaphore() -> Optional[Sample]:
                    async with semaphore:
                        return await generate_single_sample()
                
                # Create tasks for parallel generation
                tasks = [generate_with_semaphore() for _ in range(attempt_count)]
                
                # Execute all tasks concurrently
                batch_results = await asyncio.gather(*tasks)
                successful_batch = [s for s in batch_results if s is not None]
                successful_samples.extend(successful_batch)
                total_attempts += attempt_count
            
            # Trim to the requested number if we got more than needed
            if len(successful_samples) > n:
                successful_samples = successful_samples[:n]
        
        if len(successful_samples) < n:
            logger.error(f"Failed to generate {n} samples after {total_attempts} attempts. Only got {len(successful_samples)}")
            raise RuntimeError(f"Could not generate {n} samples after maximum attempts")
        
        logger.info(f"Generated {len(successful_samples)} samples for {env_name} after {total_attempts} attempts")
        return successful_samples
    
    async def ensure_samples_available(self, env_instance, required_count: int, max_concurrent: int = val_config.MAX_CONCURRENT_REQUESTS) -> List[Sample]:
        """Ensure at least required_count samples are available, generating more if needed"""
        env_name = env_instance.__class__.__name__
        
        # Get currently available samples
        available_samples = self.get_samples(env_name, required_count)
        
        # If we have enough, return them
        if len(available_samples) >= required_count:
            return available_samples[:required_count]
        
        # Generate the missing samples
        needed = required_count - len(available_samples)
        logger.info(f"Need {needed} more samples for {env_name}, generating in parallel...")
        
        new_samples = await self.generate_samples_for_env(env_instance, needed, max_concurrent)
        
        # Add new samples to storage
        self.add_samples(env_name, new_samples)
        
        # Return all samples we need (existing + new)
        all_samples = available_samples + new_samples
        return all_samples[:required_count]
    
    async def replenish_stock_background(self, env_instance, target_stock: Optional[int] = None, max_concurrent: int = val_config.MAX_CONCURRENT_REQUESTS) -> None:
        """Background task to replenish sample stock to target level"""
        env_name = env_instance.__class__.__name__
        
        try:
            stats = self.get_stats(env_name)
            target = target_stock or stats.target_stock
            current_count = stats.total
            needed = max(0, target - current_count)
            
            if needed > 0:
                logger.info(f"Background: Generating {needed} samples for {env_name} to replenish stock")
                
                # Generate samples in parallel
                new_samples = await self.generate_samples_for_env(env_instance, needed, max_concurrent)
                
                # Add to storage
                self.add_samples(env_name, new_samples)
                logger.info(f"Background: Added {len(new_samples)} samples for {env_name}")
            else:
                logger.debug(f"Background: {env_name} stock is sufficient ({current_count}/{target})")
                
        except Exception as e:
            logger.error(f"Background stock replenishment failed for {env_name}: {e}")
    
    def generate_sample_id(self) -> str:
        """Generate a unique sample ID"""
        return str(uuid.uuid4()) 