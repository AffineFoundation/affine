import random
import re
from datasets import load_dataset
from typing import Any, ClassVar, List, Dict
import affine as af
import logging
from .. import val_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = """You are an expert in science and reasoning. Please answer the following question carefully and provide your best answer.

Question: {question}

Please provide your answer in the following format:
<ANSWER>
[your answer here]
</ANSWER>

You can include any reasoning, analysis, or explanation you want, but make sure to include your final answer within the <ANSWER> </ANSWER> tags."""

class GPQA(af.BaseEnv):
    dataset_name: str = "Idavidrein/gpqa"
    subset: str = "gpqa_extended"
    split: str = "train"
    _dataset: Any = None
    _dataset_list: List[Dict] = None
    
    # Recommended timeout for this environment
    LLM_RESPONSE_TIMEOUT: ClassVar[float] = val_config.LLM_RESPONSE_TIMEOUT
    
    def __init__(self, dataset_name="Idavidrein/gpqa", subset="gpqa_extended", split="train"):
        super().__init__(dataset_name=dataset_name, subset=subset, split=split)
        self._dataset_list = None
        
    def _get_dataset(self):
        """Load dataset"""
        if self._dataset is None:
            try:
                self._dataset = load_dataset(
                    self.dataset_name,
                    self.subset,
                    split=self.split
                )
                # Convert to list for easier random access
                self._dataset_list = list(self._dataset)
                logging.info(f"Successfully loaded dataset: {self.dataset_name}/{self.subset} ({self.split} split) with {len(self._dataset_list)} samples")
            except Exception as e:
                logging.error(f"Error loading dataset: {str(e)}")
                raise RuntimeError(f"Failed to load dataset: {str(e)}")
                
        return self._dataset

    def _get_random_sample(self) -> Dict:
        """Get a random sample from the dataset"""
        if self._dataset_list is None:
            self._get_dataset()
            
        if not self._dataset_list:
            raise RuntimeError("No samples available in dataset")
            
        return random.choice(self._dataset_list)

    async def generate(self) -> af.Challenge:
        """Generate a challenge from a random GPQA sample"""
        sample = self._get_random_sample()
        
        # Extract question and correct answer
        question = sample['Question']
        correct_answer = sample['Correct Answer']
        
        # Create the prompt using the template
        prompt = PROMPT_TEMPLATE.format(question=question)
        
        return af.Challenge(
            env=self,
            prompt=prompt,
            extra={
                "question": question,
                "correct_answer": correct_answer,
                "sample": sample  # Store full sample for reference
            }
        )

    def extract_answer_from_response(self, response: str) -> str:
        """Extract answer from <ANSWER> </ANSWER> tags"""
        # Remove thinking tags first
        response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
        response = re.sub(r'<thinking>.*?</thinking>', '', response, flags=re.DOTALL)
        
        # Look for <ANSWER> </ANSWER> tags - case insensitive
        answer_pattern = r'<ANSWER>(.*?)</ANSWER>'
        matches = re.findall(answer_pattern, response, re.DOTALL | re.IGNORECASE)
        
        if matches:
            # Take the last match if multiple exist
            answer = matches[-1].strip()
            return answer
        
        # If no ANSWER tags found, return the full response as fallback
        return response.strip()

    async def verify_answer_with_llm(self, question: str, correct_answer: str, miner_answer: str) -> bool:
        """Use LLM to verify if miner answer matches correct answer"""
        verification_prompt = f"""You are an expert evaluator. Your task is to determine if two answers to a science question are essentially the same, even if they are worded differently.

Question: {question}

Correct Answer: {correct_answer}

Miner's Answer: {miner_answer}

Please evaluate whether the miner's answer is correct. Consider:
1. The core factual content and scientific accuracy
2. Whether the answers convey the same essential information
3. Minor differences in wording or explanation style should not affect the evaluation
4. The answer should be scientifically accurate and address the question

Respond with either "CORRECT" or "INCORRECT" based on your evaluation.

Your evaluation: """

        try:
            # Use the built-in LLM method from BaseEnv
            verification_response = await self.generate_input_with_llm(
                verification_prompt, 
                model="unsloth/gemma-3-27b-it",
                timeout=self.LLM_RESPONSE_TIMEOUT
            )
            
            # Extract verification result
            if verification_response and "CORRECT" in verification_response.upper():
                return True
            else:
                return False
        except Exception as e:
            logger.error(f"LLM verification failed: {str(e)}")
            # Fallback to simple string comparison
            return self.simple_answer_comparison(correct_answer, miner_answer)

    def simple_answer_comparison(self, correct_answer: str, miner_answer: str) -> bool:
        """Simple fallback comparison method"""
        # Normalize both answers
        correct_normalized = correct_answer.lower().strip()
        miner_normalized = miner_answer.lower().strip()
        
        # Direct comparison
        if correct_normalized == miner_normalized:
            return True
        
        # Check if one contains the other (for cases where one is more detailed)
        if correct_normalized in miner_normalized or miner_normalized in correct_normalized:
            return True
            
        return False

    async def evaluate(self, challenge: af.Challenge, response: af.Response) -> af.Evaluation:
        """Evaluate the response by comparing with correct answer"""
        question = challenge.extra["question"]
        correct_answer = challenge.extra["correct_answer"]
        
        # Extract answer from the miner's response
        miner_answer = self.extract_answer_from_response(response.response or "")
        
        if not miner_answer:
            # No answer found in response
            return af.Evaluation(
                env=self,
                score=0.0,
                extra={
                    "question": question,
                    "correct_answer": correct_answer,
                    "miner_answer": miner_answer,
                    "error": "No answer found in response"
                }
            )
        
        # Use LLM to verify if the answer is correct
        is_correct = await self.verify_answer_with_llm(question, correct_answer, miner_answer)
        score = 1.0 if is_correct else 0.0
        
        return af.Evaluation(
            env=self,
            score=score,
            extra={
                "question": question,
                "correct_answer": correct_answer,
                "miner_answer": miner_answer,
                "is_correct": is_correct,
                "error": None
            }
        )