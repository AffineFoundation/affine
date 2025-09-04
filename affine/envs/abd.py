import re
import time
import random
import asyncio
import affine as af
from typing import Any, Dict, Optional, Tuple

MODELS = ["unsloth/gemma-3-12b-it"]
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

INPUT_GENERATION_PROMPT = """Given this Python program and an example of how it works, generate a NEW valid input that would be accepted by the program:

Program:
```python
{program}
```

Example:
Input: {example_input}
Output: {example_output}

Now generate a NEW input that would work with this program. Make it different from the example but follow the same format and pattern.

Requirements:
1. Each line of input should be on a separate line
2. Use the exact format the program expects from stdin
3. Provide only raw input values
4. Do not include any prefixes or extra formatting
5. Make it different from the example input

Format your response with <INPUT> </INPUT> tags like this:
<INPUT>
[your new input data here]
</INPUT>

Please generate a valid input:"""

dataset = af.singleton('rl-python', lambda: af.utils.R2BufferedDataset(
        dataset_name="satpalsr/rl-python",
        buffer_size=5,
        max_batch=5,
))

class ABD(af.BaseEnv):
    __version__: str = "0.0.0"
    def __init__(self):
        super().__init__()
        self._executor = af.utils.ProgramExecutor()
        
    async def _create_challenge(
        self, program: str, example_in: str, example_out: str
    ) -> Optional[af.Challenge]:
        """Use the LLM to propose a new input, validate & execute."""
        af.logger.trace("Creating challenge with program, example input, and example output.")
        prompt = INPUT_GENERATION_PROMPT.format(
            program=program,
            example_input=example_in,
            example_output=example_out
        )
        af.logger.trace(f"Generated prompt for LLM: {prompt[:10]}...")
        resp = await af.query(prompt, model=random.choice( MODELS ))
        llm_resp = resp.response
        if not llm_resp:
            af.logger.trace(f"No response from LLM error: {resp.error}, continuing to next iteration.")
            return None

        gen_input = self.extract_input_from_response(llm_resp)
        af.logger.trace(f"Extracted input from LLM response: {gen_input}")
        if not self._validate_input_for_program(program, gen_input):
            af.logger.trace("Generated input insufficient, retrying")
            return None

        loop = asyncio.get_running_loop()
        output, error = await loop.run_in_executor(
            None, self._executor.execute, program, gen_input
        )
        af.logger.trace(f"Executed program with generated input. Output: {output}, Error: {error}, program: {program}")
        if error or not output.strip():
            af.logger.trace("Generated input contains error")
            return None            

        af.logger.trace("Challenge created successfully.")
        return af.Challenge(
            env=self,
            prompt=PROMPT_TEMPLATE.format(program=program, output=output),
            extra={"program": program, "expected_output": output, 'timestamp': time.time() },
        )

    async def generate(self) -> af.Challenge:
        af.logger.trace("Generating a new challenge.")
        while True:
            sample = await dataset().get()
            program     = sample.get("program")
            example_in  = sample.get("inputs", "")
            example_out = sample.get("output", "")
            challenge = await self._create_challenge(program, example_in, example_out)
            if challenge is not None:
                af.logger.trace("Generated challenge successfully.")
                return challenge 
            else:
                af.logger.trace(f"Failed generation with null challenge, continuing...")
                await asyncio.sleep(5)
                continue

    def extract_input_from_response(self, response: str) -> str:
        """Pull out the last <INPUT>…</INPUT> block."""
        response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
        response = re.sub(r"<thinking>.*?</thinking>", "", response, flags=re.DOTALL)
        matches = re.findall(r"<INPUT>(.*?)</INPUT>", response, re.IGNORECASE | re.DOTALL)
        if not matches:
            af.logger.trace("No <INPUT> tags found in response.")
            return ""
        lines = [ln.rstrip() for ln in matches[-1].strip().splitlines()]
        while lines and not lines[-1].strip():
            lines.pop()
        extracted_input = "\n".join(lines)
        return extracted_input

    def _validate_input_for_program(self, program: str, inp: str) -> bool:
        """Heuristic: ensure at least as many lines as input() calls."""
        calls = program.count("input()")
        lines = inp.splitlines() if inp else []
        if "for _ in range(int(input()))" in program and lines and lines[0].isdigit():
            valid = len(lines) > int(lines[0])
            af.logger.trace(f"Validation result for loop-based input: {valid}")
            return valid
        valid = len(lines) >= calls
        return valid

    def compare_outputs(self, expected: str, actual: str) -> bool:
        """Normalize line endings & trailing whitespace."""
        af.logger.trace(f"Comparing outputs. Expected: {expected}, Actual: {actual}")
        if expected == actual:
            af.logger.trace("Outputs match exactly.")
            return True
        exp = expected.strip().replace("\r\n", "\n")
        act = actual.strip().replace("\r\n", "\n")
        if exp == act:
            af.logger.trace("Outputs match after normalization.")
            return True
        match = [l.rstrip() for l in exp.splitlines()] == [l.rstrip() for l in act.splitlines()]
        af.logger.trace(f"Outputs match after line comparison: {match}")
        return match

    async def evaluate(self, challenge: af.Challenge, response: af.Response) -> af.Evaluation:
        af.logger.trace(f"Evaluating response. Challenge: {challenge}, Response: {response}")
        prog = challenge.extra["program"]
        expected = challenge.extra["expected_output"]
        gen_in = self.extract_input_from_response(response.response or "")
        resp_text = response.response or ""
        tags_present = bool(re.search(r"<INPUT>.*?</INPUT>", resp_text, re.IGNORECASE | re.DOTALL))
        if not gen_in and not tags_present:
            af.logger.trace("No <INPUT> tags found in response.")
            return af.Evaluation(
                env=self, score=0.0,
                extra={"error": "No input found", "expected_output": expected}
            )
        if not self._validate_input_for_program(prog, gen_in):
            af.logger.trace("Provided input is invalid or insufficient for the program.")
            return af.Evaluation(
                env=self, score=0.0,
                extra={"error": "Invalid input for program", "generated_input": gen_in, "expected_output": expected}
            )
        loop = asyncio.get_running_loop()
        out, err = await loop.run_in_executor(
            None, self._executor.execute, prog, gen_in
        )
        af.logger.trace(f"Executed program with generated input. Output: {out}, Error: {err}")
        if err:
            af.logger.trace("Error occurred during program execution.")
            return af.Evaluation(
                env=self, score=0.0,
                extra={"error": err, "generated_output": out}
            )
        ok = self.compare_outputs(expected, out)
        af.logger.trace(f"Output comparison result: {ok}")
        return af.Evaluation(
            env=self, score=1.0 if ok else 0.0,
            extra={"outputs_match": ok, "generated_input": gen_in, "generated_output": out, 'expected_output': expected}
        )
