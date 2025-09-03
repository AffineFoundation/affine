import re
import affine as af
from typing import Any, Dict, List, Tuple, Optional

dataset = af.singleton('euler', lambda: af.utils.R2BufferedDataset(
        dataset_name="alexandonian/project-euler",
        buffer_size=5,
        max_batch=5,
))

class ELR(af.BaseEnv):
    __version__: str = "0.0.1"
    def __init__(self):
        super().__init__()

    async def generate(self) -> af.Challenge:
        af.logger.trace("Generating a new challenge.")
        sample = await dataset().get()
        prompt = f"""{sample['problem']}

Provide your solution in blocks <Answer>...</Answer>
For instance: 
<Answer>20</Answer>"""
        return af.Challenge(env=self, prompt=prompt, extra=sample)

    def extract_answer_from_response(self, response: str) -> str:
        """Extract the answer from <Answer>...</Answer> tags."""
        if not response:
            return ""
        matches = re.findall(r"<Answer>(.*?)</Answer>", response, re.IGNORECASE | re.DOTALL)
        if not matches:
            return ""
        return matches[-1].strip()

    async def evaluate(self, challenge: af.Challenge, response: af.Response) -> af.Evaluation:
        resp = response.response
        solution = challenge.extra['numerical_answer']
        extracted_answer = self.extract_answer_from_response(resp or "")
        ok = extracted_answer == str(solution)
        return af.Evaluation(env=self, score=1.0 if ok else 0.0, extra={"extracted_answer": extracted_answer, "expected_answer": solution})
