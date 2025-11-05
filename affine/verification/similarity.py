"""Similarity checking for model outputs."""

import os
import random
import asyncio
import aiohttp
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

from affine.setup import logger


class SimilarityMetric(Enum):
    """Similarity metric enum."""

    COSINE = "cosine"
    EXACT_MATCH = "exact_match"
    TOKEN_OVERLAP = "token_overlap"
    # Can add more metrics like BLEU, ROUGE later


@dataclass
class ComparisonResult:
    """Result of comparing two outputs."""

    prompt: str
    chutes_output: str
    local_output: str
    similarity_score: float
    metric: SimilarityMetric
    error: Optional[str] = None


class SimilarityChecker:
    """Check similarity between local and Chutes model outputs."""

    def __init__(self, metric: Optional[SimilarityMetric] = None):
        """Initialize similarity checker.

        Args:
            metric: Similarity metric to use (default from env)
        """
        if metric is None:
            metric_str = os.getenv("VERIFICATION_SIMILARITY_METRIC", "token_overlap")
            metric = SimilarityMetric(metric_str)

        self.metric = metric
        logger.info(f"Initialized similarity checker with metric: {metric.value}")

    async def compare_outputs(
        self,
        chutes_endpoint: str,
        local_endpoint: str,
        model: str,
        prompts: List[str],
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> List[ComparisonResult]:
        """Compare outputs from Chutes and local deployments.

        Args:
            chutes_endpoint: Chutes API endpoint (e.g., https://slug.chutes.ai/v1)
            local_endpoint: Local deployment endpoint (e.g., http://host:port/v1)
            model: Model name
            prompts: List of prompts to test
            temperature: Sampling temperature
            max_tokens: Max tokens to generate

        Returns:
            List of comparison results
        """
        results = []

        for prompt in prompts:
            try:
                # Get outputs from both endpoints
                chutes_output, local_output = await asyncio.gather(
                    self._get_model_output(chutes_endpoint, model, prompt, temperature, max_tokens),
                    self._get_model_output(local_endpoint, model, prompt, temperature, max_tokens),
                )

                # Calculate similarity
                similarity_score = self._calculate_similarity(chutes_output, local_output)

                results.append(
                    ComparisonResult(
                        prompt=prompt,
                        chutes_output=chutes_output,
                        local_output=local_output,
                        similarity_score=similarity_score,
                        metric=self.metric,
                    )
                )

                logger.debug(f"Similarity for prompt '{prompt[:50]}...': {similarity_score:.4f}")

            except Exception as e:
                logger.warning(f"Error comparing outputs for prompt '{prompt[:50]}...': {e}")
                results.append(
                    ComparisonResult(
                        prompt=prompt,
                        chutes_output="",
                        local_output="",
                        similarity_score=0.0,
                        metric=self.metric,
                        error=str(e),
                    )
                )

        return results

    async def _get_model_output(
        self,
        endpoint: str,
        model: str,
        prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Get model output from an endpoint.

        Args:
            endpoint: API endpoint
            model: Model name
            prompt: Input prompt
            temperature: Sampling temperature
            max_tokens: Max tokens to generate

        Returns:
            Model output text
        """
        url = f"{endpoint}/chat/completions"

        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        # Add authorization for Chutes endpoint
        headers = {}
        if "chutes.ai" in endpoint:
            chutes_api_key = os.getenv("CHUTES_API_KEY", "")
            if chutes_api_key:
                headers["Authorization"] = f"Bearer {chutes_api_key}"

        timeout = aiohttp.ClientTimeout(total=120)

        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, json=payload, headers=headers) as response:
                response.raise_for_status()
                data = await response.json()

                # Extract text from response
                choices = data.get("choices", [])
                if not choices:
                    return ""

                message = choices[0].get("message", {})
                content = message.get("content", "")
                return content.strip()

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Similarity score between 0 and 1
        """
        if self.metric == SimilarityMetric.EXACT_MATCH:
            return 1.0 if text1 == text2 else 0.0

        elif self.metric == SimilarityMetric.TOKEN_OVERLAP:
            return self._token_overlap_similarity(text1, text2)

        elif self.metric == SimilarityMetric.COSINE:
            return self._cosine_similarity(text1, text2)

        else:
            raise ValueError(f"Unknown metric: {self.metric}")

    def _token_overlap_similarity(self, text1: str, text2: str) -> float:
        """Calculate token overlap similarity (Jaccard similarity).

        Args:
            text1: First text
            text2: Second text

        Returns:
            Similarity score between 0 and 1
        """
        # Simple whitespace tokenization
        tokens1 = set(text1.lower().split())
        tokens2 = set(text2.lower().split())

        if not tokens1 and not tokens2:
            return 1.0

        if not tokens1 or not tokens2:
            return 0.0

        # Jaccard similarity
        intersection = len(tokens1 & tokens2)
        union = len(tokens1 | tokens2)

        return intersection / union if union > 0 else 0.0

    def _cosine_similarity(self, text1: str, text2: str) -> float:
        """Calculate cosine similarity using simple term frequency vectors.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Similarity score between 0 and 1
        """
        import math
        from collections import Counter

        # Tokenize
        tokens1 = text1.lower().split()
        tokens2 = text2.lower().split()

        if not tokens1 or not tokens2:
            return 0.0

        # Build term frequency vectors
        tf1 = Counter(tokens1)
        tf2 = Counter(tokens2)

        # Get all unique terms
        all_terms = set(tf1.keys()) | set(tf2.keys())

        # Build vectors
        vec1 = [tf1.get(term, 0) for term in all_terms]
        vec2 = [tf2.get(term, 0) for term in all_terms]

        # Calculate cosine similarity
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(b * b for b in vec2))

        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0

        return dot_product / (magnitude1 * magnitude2)


async def get_sample_prompts(
    hotkey: str,
    block: int,
    sample_size: int,
) -> List[str]:
    """Get sample prompts for a miner from R2.

    Args:
        hotkey: Miner hotkey
        block: Block number
        sample_size: Number of samples to get

    Returns:
        List of prompts
    """
    from affine.storage import dataset

    # Load results for this miner around this block
    prompts = []
    tail = int(os.getenv("VERIFICATION_SAMPLE_TAIL", "10000"))  # Look back this many blocks

    try:
        logger.info(f"Loading prompts for {hotkey} from last {tail} blocks (around block {block})")
        count_checked = 0

        async for result in dataset(tail=tail, compact=False):
            count_checked += 1

            # Filter by hotkey and block proximity
            if result.miner.hotkey == hotkey and abs(result.miner.block - block) < 100:
                # Try to extract prompt from various possible locations
                prompt = None

                if hasattr(result, 'extra') and isinstance(result.extra, dict):
                    # Try extra.prompt (direct format)
                    prompt = result.extra.get("prompt")

                    # Try extra.details.experiences (ABD/AgentGym format)
                    if not prompt:
                        details = result.extra.get("details", {})
                        if isinstance(details, dict):
                            experiences = details.get("experiences", [])
                            if isinstance(experiences, list) and len(experiences) > 0:
                                first_exp = experiences[0]

                                # Try conversation format (AgentGym)
                                if isinstance(first_exp, dict) and "conversation" in first_exp:
                                    conversation = first_exp["conversation"]
                                    if isinstance(conversation, list):
                                        for msg in conversation:
                                            if isinstance(msg, dict) and msg.get("role") == "user":
                                                prompt = msg.get("content")
                                                if prompt:
                                                    break

                                # Try direct format (ABD)
                                elif isinstance(first_exp, dict) and first_exp.get("role") == "user":
                                    prompt = first_exp.get("content")

                    # Try extra.challenge.prompt (legacy format)
                    if not prompt:
                        challenge = result.extra.get("challenge", {})
                        if isinstance(challenge, dict):
                            prompt = challenge.get("prompt")

                if prompt and isinstance(prompt, str) and len(prompt.strip()) > 0:
                    prompts.append(prompt.strip())
                    logger.debug(f"Found prompt from block {result.miner.block}: {prompt[:50]}...")

                    # Stop when we have enough
                    if len(prompts) >= sample_size * 2:  # Get more than needed
                        logger.info(f"Collected {len(prompts)} prompts after checking {count_checked} results")
                        break

        logger.info(f"Checked {count_checked} results, found {len(prompts)} prompts for {hotkey}")

    except Exception as e:
        logger.warning(f"Error loading prompts from R2: {e}", exc_info=True)

    # If we don't have enough prompts, use default ones
    if len(prompts) < sample_size:
        logger.warning(f"Only found {len(prompts)} prompts for {hotkey}, using defaults")
        default_prompts = _get_default_prompts()
        prompts.extend(default_prompts)

    # Randomly sample
    if len(prompts) > sample_size:
        prompts = random.sample(prompts, sample_size)

    return prompts[:sample_size]


def _get_default_prompts() -> List[str]:
    """Get default prompts for testing.

    Returns:
        List of default prompts
    """
    return [
        "What is the capital of France?",
        "Explain quantum computing in simple terms.",
        "Write a Python function to reverse a string.",
        "What are the main causes of climate change?",
        "Describe the process of photosynthesis.",
        "How does a neural network work?",
        "What is the difference between AI and machine learning?",
        "Explain the theory of relativity.",
        "What are the benefits of renewable energy?",
        "How do vaccines work?",
    ]
