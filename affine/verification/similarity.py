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
    LLM_JUDGE = "llm_judge"  # Use LLM to judge semantic similarity
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
        temperature: float = 0.0,
        max_tokens: int = 512,
        seed: int = 42,
    ) -> List[ComparisonResult]:
        """Compare outputs from Chutes and local deployments.

        Args:
            chutes_endpoint: Chutes API endpoint (e.g., https://slug.chutes.ai/v1)
            local_endpoint: Local deployment endpoint (e.g., http://host:port/v1)
            model: Model name
            prompts: List of prompts to test
            temperature: Sampling temperature (default 0.0 for deterministic output)
            max_tokens: Max tokens to generate
            seed: Random seed for reproducibility

        Returns:
            List of comparison results
        """
        results = []

        for prompt in prompts:
            try:
                # Get outputs from both endpoints
                chutes_output, local_output = await asyncio.gather(
                    self._get_model_output(chutes_endpoint, model, prompt, temperature, max_tokens, seed),
                    self._get_model_output(local_endpoint, model, prompt, temperature, max_tokens, seed),
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
                error_msg = f"{type(e).__name__}: {str(e)}" if str(e) else type(e).__name__
                logger.warning(
                    f"Error comparing outputs for prompt '{prompt[:50]}...': {error_msg}",
                    exc_info=True if not str(e) else False  # Show traceback if error message is empty
                )
                results.append(
                    ComparisonResult(
                        prompt=prompt,
                        chutes_output="",
                        local_output="",
                        similarity_score=0.0,
                        metric=self.metric,
                        error=error_msg,
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
        seed: int,
    ) -> str:
        """Get model output from an endpoint.

        Args:
            endpoint: API endpoint
            model: Model name
            prompt: Input prompt
            temperature: Sampling temperature
            max_tokens: Max tokens to generate
            seed: Random seed for reproducibility

        Returns:
            Model output text
        """
        url = f"{endpoint}/chat/completions"

        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "seed": seed,
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
    tail = int(os.getenv("VERIFICATION_SAMPLE_TAIL", "1000"))  # Look back fewer blocks (was 10000)

    try:
        logger.info(f"Loading prompts from last {tail} blocks around block {block} (from all miners to prevent gaming)")
        count_checked = 0
        max_to_check = int(os.getenv("VERIFICATION_MAX_RESULTS_CHECK", "20000"))  # Configurable limit

        async for result in dataset(tail=tail, compact=False):
            count_checked += 1

            # Stop early if we've checked too many results
            if count_checked >= max_to_check:
                logger.warning(f"Reached max check limit ({max_to_check}), stopping search")
                break

            # Filter by block proximity only (get prompts from any miner to prevent gaming)
            if abs(result.miner.block - block) < 100:
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

        logger.info(f"Checked {count_checked} results, found {len(prompts)} prompts from historical data")

    except Exception as e:
        logger.warning(f"Error loading prompts from R2: {e}", exc_info=True)

    # If we don't have enough prompts, fail verification
    # New miners need sufficient historical data to get weight, so this is expected behavior
    if len(prompts) < sample_size:
        logger.error(
            f"Insufficient historical prompts for verification: found {len(prompts)}, need {sample_size}. "
            f"Miner {hotkey} needs more historical data before verification can proceed."
        )
        return []  # Return empty list to indicate verification cannot proceed

    # Randomly sample
    if len(prompts) > sample_size:
        prompts = random.sample(prompts, sample_size)

    logger.info(f"Got {len(prompts[:sample_size])} prompts for testing from R2 historical data")
    return prompts[:sample_size]


def _get_default_prompts() -> List[str]:
    """Get default prompts for testing.

    Includes diverse prompts across different categories:
    - General knowledge and facts
    - Science and technology
    - Programming and math
    - Creative writing
    - Reasoning and problem solving
    - Explanations and tutorials

    Returns:
        List of default prompts (50+ prompts)
    """
    return [
        # General knowledge (simple)
        "What is the capital of France?",
        "Who wrote Romeo and Juliet?",
        "What is the largest planet in our solar system?",
        "When did World War II end?",
        "What is the chemical symbol for gold?",

        # Science explanations
        "Explain quantum computing in simple terms.",
        "What are the main causes of climate change?",
        "Describe the process of photosynthesis.",
        "How do vaccines work?",
        "Explain the theory of relativity.",
        "What is DNA and how does it work?",
        "Describe the water cycle.",
        "How does electricity flow through a circuit?",
        "What causes earthquakes?",
        "Explain the greenhouse effect.",

        # Programming tasks
        "Write a Python function to reverse a string.",
        "How do I sort a list in Python?",
        "Write a function to check if a number is prime.",
        "Explain what recursion is with an example.",
        "How do you implement a binary search algorithm?",
        "Write code to find the factorial of a number.",
        "Explain the difference between a list and a tuple in Python.",
        "How do you handle exceptions in Python?",
        "Write a function to count vowels in a string.",
        "What is the difference between '==' and 'is' in Python?",

        # AI and Technology
        "How does a neural network work?",
        "What is the difference between AI and machine learning?",
        "Explain what deep learning is.",
        "What is natural language processing?",
        "How do large language models work?",
        "What is computer vision?",
        "Explain reinforcement learning.",
        "What is the difference between supervised and unsupervised learning?",

        # Math problems
        "Solve for x: 2x + 5 = 15",
        "What is the Pythagorean theorem?",
        "Calculate the area of a circle with radius 5.",
        "What is a derivative in calculus?",
        "Explain what a prime number is.",

        # Creative and open-ended
        "Write a short story about a robot learning to cook.",
        "Describe a beautiful sunset in poetic language.",
        "What would happen if gravity suddenly stopped working?",
        "Imagine a world where everyone can read minds. What would it be like?",

        # Reasoning and problem solving
        "If you have 12 apples and give away 3, then buy 5 more, how many do you have?",
        "A farmer has 17 sheep. All but 9 die. How many are left?",
        "You have a 3-liter jug and a 5-liter jug. How can you measure exactly 4 liters?",
        "What comes next in this sequence: 2, 4, 8, 16, ?",

        # Tutorials and how-to
        "How do I make a paper airplane?",
        "Explain how to tie a tie step by step.",
        "What are the steps to brew coffee?",
        "How do you change a tire?",

        # Health and lifestyle
        "What are the benefits of regular exercise?",
        "What are the benefits of renewable energy?",
        "How much water should a person drink daily?",
        "What is a balanced diet?",

        # General explanations
        "What is the difference between weather and climate?",
        "How does the internet work?",
        "What is blockchain technology?",
        "Explain how a refrigerator keeps food cold.",

        # History and geography
        "What caused the fall of the Roman Empire?",
        "Who was the first person to walk on the moon?",
        "What is the Great Wall of China?",
        "Describe the Renaissance period.",
        "What were the main events of the American Revolution?",
        "Who discovered America?",
        "What is the Silk Road?",
        "Explain the Industrial Revolution.",

        # Language and communication
        "What is the difference between 'affect' and 'effect'?",
        "Explain what a metaphor is.",
        "How many languages are spoken in the world?",
        "What is grammar?",
        "Explain the difference between 'their', 'there', and 'they're'.",

        # Business and economics
        "What is supply and demand?",
        "Explain what inflation means.",
        "What is a stock market?",
        "How does compound interest work?",
        "What is GDP?",

        # More programming challenges
        "Write a function to find the largest number in a list.",
        "How do you remove duplicates from a list?",
        "Write code to check if a string is a palindrome.",
        "Explain what an API is.",
        "What is the difference between GET and POST requests?",
        "How do you reverse a linked list?",
        "Write a function to merge two sorted arrays.",

        # Philosophy and ethics
        "What is the meaning of life?",
        "Explain the trolley problem.",
        "What is the difference between right and wrong?",
        "Who was Socrates?",

        # Space and astronomy
        "How far is the sun from Earth?",
        "What is a black hole?",
        "How many planets are in our solar system?",
        "What is the Big Bang theory?",
        "Can humans live on Mars?",

        # Animals and nature
        "How do birds fly?",
        "What is the largest animal on Earth?",
        "How do bees make honey?",
        "Why do cats purr?",
        "What is photosynthesis in simple terms?",

        # Food and cooking
        "How do you make scrambled eggs?",
        "What temperature should chicken be cooked to?",
        "What is the difference between baking and roasting?",
        "How do you boil an egg perfectly?",

        # Sports and games
        "What are the rules of chess?",
        "How many players are on a soccer team?",
        "What is a home run in baseball?",
        "Explain the offside rule in soccer.",
    ]
