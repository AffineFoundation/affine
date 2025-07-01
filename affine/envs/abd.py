import random
import asyncio
import re
import json
from datasets import load_dataset, IterableDataset
from typing import Any, ClassVar, List, Dict, Optional
import affine as af
from affine.program_executor import ProgramExecutor
import logging
from .. import val_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = """You are a programming expert. Given a Python program and its expected output, you need to determine the exact input that would produce this output.

Program:
```python
{program}
```

Expected Output:
```
{output}
```

Task: Analyze the program to understand what input format it expects from stdin, then provide the input data that would produce the expected output.

You can provide any explanations, analysis, or reasoning you want. However, you MUST include the input data within <INPUT> </INPUT> tags.

Format the input data like this:
<INPUT>
[input data here - each line on a separate line as the program expects]
</INPUT>

I will extract only the content between these tags.

Requirements for the input data within the tags:
1. Each line of input should be on a separate line
2. Use the exact format the program expects  
3. Provide the raw input values that should be fed to stdin
4. Do not include any prefixes or extra formatting within the INPUT tags

Please analyze the program and provide the required input:"""

INPUT_GENERATION_PROMPT = """Given this Python program, generate a valid input that would be accepted by the program:

Program:
```python
{program}
```

Requirements:
1. Each line of input should be on a separate line
2. Use the exact format the program expects from stdin
3. Provide only raw input values
4. Do not include any prefixes or extra formatting

Format your response with <INPUT> </INPUT> tags like this:
<INPUT>
[your input data here]
</INPUT>

Please generate a valid input:"""

class ABD(af.BaseEnv):
    dataset_name: str
    executor: ProgramExecutor = None
    _dataset: Any = None
    _dataset_iterator: Any = None
    
    # Recommended timeout for this environment
    LLM_RESPONSE_TIMEOUT: ClassVar[float] = val_config.LLM_RESPONSE_TIMEOUT
    
    # Maximum number of samples to keep in memory
    MAX_SAMPLES: ClassVar[int] = 1000
    
    def __init__(self, dataset_name="satpalsr/rl-python"):
        super().__init__(dataset_name=dataset_name, executor=ProgramExecutor(), _dataset=None)
        self._dataset_iterator = None
        
    def _get_dataset(self):
        """Load dataset as an iterator to prevent memory issues"""
        if self._dataset is None:
            try:
                # Load dataset in streaming mode
                self._dataset = load_dataset(
                    self.dataset_name,
                    streaming=True
                )
                
                # Use training split if available, otherwise use the first available split
                if 'train' in self._dataset:
                    self._dataset_iterator = iter(self._dataset['train'])
                else:
                    # Get the first available split
                    split_name = list(self._dataset.keys())[0]
                    self._dataset_iterator = iter(self._dataset[split_name])
                    
                logging.info(f"Successfully initialized streaming dataset: {self.dataset_name}")
            except Exception as e:
                logging.error(f"Error loading dataset: {str(e)}")
                raise RuntimeError(f"Failed to load dataset: {str(e)}")
                
        return self._dataset

    def _get_random_samples(self, n: int) -> List[Dict]:
        """Get n random samples from the dataset using reservoir sampling"""
        if not self._dataset_iterator:
            self._get_dataset()
            
        samples = []
        count = 0
        
        try:
            # Get all samples at once
            while count < self.MAX_SAMPLES:
                try:
                    item = next(self._dataset_iterator)
                    samples.append(item)
                    count += 1
                except StopIteration:
                    # Reset iterator and continue collecting if we need more samples
                    self._dataset_iterator = iter(self._dataset['train'])
                    if count < n:  # Only continue if we don't have enough samples
                        continue
                    break
            
            logger.debug(f"Collected {count} samples from dataset")
            
            # If we have fewer samples than requested, just return what we have
            if count <= n:
                logger.debug(f"Returning all {len(samples)} samples (fewer than requested {n})")
                return samples
            
            # Use reservoir sampling to select n random samples
            result = []
            indices = random.sample(range(count), n)  # Select n random unique indices
            for idx in indices:
                result.append(samples[idx])
            
            logger.debug(f"Selected {len(result)} samples using reservoir sampling")
            return result
            
        except Exception as e:
            logging.error(f"Error during sampling: {str(e)}")
            return samples[:n] if samples else []

    async def generate_new_test_cases(self, n: int, max_concurrent: int = val_config.MAX_CONCURRENT_REQUESTS) -> List[Dict[str, Any]]:
        """Generate test cases in parallel"""
        # Get random samples using reservoir sampling
        logger.debug("Starting to generate new test cases")
        samples = self._get_random_samples(n)
        
        if not samples:
            logger.error("Failed to get samples from dataset")
            raise RuntimeError("Failed to get samples from dataset")

        logger.debug(f"Got {len(samples)} samples from dataset")

        # Generate prompts for input generation
        prompts = [
            INPUT_GENERATION_PROMPT.format(program=sample['program'])
            for sample in samples
        ]

        logger.debug(f"Processing {len(prompts)} prompts in parallel")
        
        # Generate inputs for all prompts in parallel
        try:
            generated_inputs = await self.generate_inputs_parallel(
                prompts, 
                max_concurrent,
                model="deepseek-ai/DeepSeek-R1"
            )
            logger.debug(f"Generated {len(generated_inputs)} inputs")
        except Exception as e:
            logger.error(f"Failed to generate inputs: {str(e)}")
            return []

        # Process each generated input through the program
        new_test_cases = []
        for idx, (sample, generated_input) in enumerate(zip(samples, generated_inputs)):
            logger.debug(f"Processing generated input {idx + 1}")
            
            if not generated_input:
                logger.warning("Empty generated input, skipping")
                continue

            # Extract input from LLM response
            input_data = self.extract_input_from_response(generated_input)
            if not input_data:
                logger.warning("Failed to extract input from LLM response")
                logger.debug(f"Raw LLM response: {generated_input}")
                continue

            # Run the program with generated input
            output, error = self.executor.execute_program(sample['program'], input_data)
            if error:
                logger.warning(f"Program execution failed: {error}")
                logger.debug(f"Program: {sample['program']}")
                logger.debug(f"Input: {input_data}")
                continue

            logger.debug("Successfully generated test case")
            # Create new test case
            new_test_cases.append({
                'program': sample['program'],
                'input': input_data,
                'output': output
            })

            # Break if we have enough test cases
            if len(new_test_cases) >= n:
                break

        if not new_test_cases:
            logger.error("Failed to generate any valid test cases")
            raise RuntimeError("Failed to generate any valid test cases")

        logger.info(f"Successfully generated {len(new_test_cases)} test cases")
        return new_test_cases

    async def generate(self) -> af.Challenge:
        """Generate a challenge by creating new test cases"""
        # Generate one new test case
        test_cases = await self.generate_new_test_cases(1)
        test_case = test_cases[0]  # We know this exists because generate_new_test_cases raises if empty
        
        # Create the prompt using the template
        prompt = PROMPT_TEMPLATE.format(
            program=test_case['program'],
            output=test_case['output']
        )
        
        return af.Challenge(
            env=self,
            prompt=prompt,
            extra={
                "program": test_case['program'],
                "expected_output": test_case['output'],
            }
        )

    async def generate_batch(self, n: int) -> List[af.Challenge]:
        """Generate multiple challenges in parallel"""
        # Generate n test cases at once
        test_cases = await self.generate_new_test_cases(n)
        
        # Convert test cases to challenges
        challenges = []
        for test_case in test_cases:
            prompt = PROMPT_TEMPLATE.format(
                program=test_case['program'],
                output=test_case['output']
            )
            
            challenge = af.Challenge(
                env=self,
                prompt=prompt,
                extra={
                    "program": test_case['program'],
                    "expected_output": test_case['output'],
                }
            )
            challenges.append(challenge)

        # Start background task to generate more samples if needed
        asyncio.create_task(self._replenish_samples_background())
        
        return challenges

    async def _replenish_samples_background(self, target_stock: int = val_config.TARGET_STOCK):
        """Background task to replenish sample stock"""
        try:
            from affine.sample_manager import SampleManager
            sample_manager = SampleManager()
            
            # Get current sample count
            stats = sample_manager.get_stats(self.__class__.__name__)
            current_count = stats.total
            
            if current_count < target_stock:
                # Generate more samples in background
                needed = target_stock - current_count
                logger.info(f"Background: Generating {needed} samples to replenish stock")
                
                # Generate new samples
                new_test_cases = await self.generate_new_test_cases(needed)
                
                # Convert to Sample objects
                new_samples = []
                for test_case in new_test_cases:
                    sample = af.Sample.from_challenge(
                        af.Challenge(
                            env=self,
                            prompt=PROMPT_TEMPLATE.format(
                                program=test_case['program'],
                                output=test_case['output']
                            ),
                            extra={
                                "program": test_case['program'],
                                "expected_output": test_case['output'],
                            }
                        ),
                        self.__class__.__name__
                    )
                    new_samples.append(sample)
                
                # Add to storage
                sample_manager.add_samples(self.__class__.__name__, new_samples)
                logger.info(f"Background: Added {len(new_samples)} samples")
        except Exception as e:
            logger.error(f"Background sample generation failed: {e}")

    def extract_input_from_response(self, response: str) -> str:
        """Extract input data from <INPUT> </INPUT> tags only"""
        # Remove thinking tags first
        response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
        response = re.sub(r'<thinking>.*?</thinking>', '', response, flags=re.DOTALL)
        
        # Look for <INPUT> </INPUT> tags - case insensitive
        input_pattern = r'<INPUT>(.*?)</INPUT>'
        matches = re.findall(input_pattern, response, re.DOTALL | re.IGNORECASE)
        
        if matches:
            # Take the last match if multiple exist
            input_data = matches[-1].strip()
            # Clean up the input data
            lines = input_data.split('\n')
            cleaned_lines = []
            
            for line in lines:
                # Remove leading/trailing whitespace but preserve internal spacing
                cleaned_line = line.rstrip()
                cleaned_lines.append(cleaned_line)
            
            # Remove empty lines at the end
            while cleaned_lines and not cleaned_lines[-1].strip():
                cleaned_lines.pop()
            
            return '\n'.join(cleaned_lines)
        
        # If no INPUT tags found, return empty string
        return ""
    
    def compare_outputs(self, expected: str, actual: str) -> bool:
        """Compare expected and actual outputs with multiple normalization strategies"""
        # Direct comparison
        if expected == actual:
            return True
        
        # Normalize line endings
        expected_normalized = expected.strip().replace('\r\n', '\n')
        actual_normalized = actual.strip().replace('\r\n', '\n')
        
        if expected_normalized == actual_normalized:
            return True
        
        # Compare line by line (strip trailing whitespace)
        expected_lines = [line.rstrip() for line in expected_normalized.split('\n')]
        actual_lines = [line.rstrip() for line in actual_normalized.split('\n')]
        
        if expected_lines == actual_lines:
            return True
        
        # Compact comparison (remove all whitespace)
        expected_compact = ''.join(expected_normalized.split())
        actual_compact = ''.join(actual_normalized.split())
        
        if expected_compact == actual_compact:
            return True
        
        # Numeric comparison (try to parse as numbers)
        try:
            expected_nums = [float(x) for x in expected_normalized.split()]
            actual_nums = [float(x) for x in actual_normalized.split()]
            if len(expected_nums) == len(actual_nums):
                return all(abs(e - a) < 1e-9 for e, a in zip(expected_nums, actual_nums))
        except Exception:
            pass
        
        return False
    
    async def evaluate(self, challenge: af.Challenge, response: af.Response) -> af.Evaluation:
        """Evaluate the response by extracting input and running the program"""
        program = challenge.extra["program"]
        expected_output = challenge.extra["expected_output"]
        
        # Extract input from the LLM response
        generated_input = self.extract_input_from_response(response.response or "")
        
        if not generated_input:
            # No input found in response
            return af.Evaluation(
                env=self,
                score=0.0,
                extra={
                    "expected_output": expected_output,
                    "generated_input": generated_input,
                    "error": "No input found in response"
                }
            )
        
        # Execute the program with the extracted input
        generated_output, error = self.executor.execute_program(program, generated_input)
        
        if error:
            # Program execution failed
            return af.Evaluation(
                env=self,
                score=0.0,
                extra={
                    "expected_output": expected_output,
                    "generated_input": generated_input,
                    "generated_output": generated_output,
                    "error": error
                }
            )
        
        # Compare the actual output with expected output
        outputs_match = self.compare_outputs(expected_output, generated_output)
        score = 1.0 if outputs_match else 0.0
        
        return af.Evaluation(
            env=self,
            score=score,
            extra={
                "expected_output": expected_output,
                "generated_input": generated_input,
                "generated_output": generated_output,
                "outputs_match": outputs_match,
                "error": None
            }
        )
    
    def cleanup(self):
        """Clean up resources"""
        if hasattr(self, 'executor'):
            self.executor.cleanup()
    
    def __del__(self):
        """Destructor to ensure cleanup"""
        try:
            self.cleanup()
        except Exception:
            pass