"""
COIN environment - Simple coin flip guessing challenge.
"""
import random
from .. import BaseEnv, Challenge, Response, Evaluation


class COIN(BaseEnv):
    """Simple coin flip guessing challenge environment."""
    
    async def generate(self) -> Challenge:
        """Generate a coin flip challenge."""
        answer = random.choice(["HEADS", "TAILS"])
        return Challenge(
            env=self,
            prompt="I flipped a coin, guess HEADS or TAILS.",
            extra={"answer": answer}
        )

    async def evaluate(self, challenge: Challenge, response: Response) -> Evaluation:
        """Evaluate the coin flip guess."""
        expected_answer = challenge.extra["answer"]
        user_guess = (response.response or "").strip().upper()
        
        is_correct = user_guess == expected_answer
        
        return Evaluation(
            env=self,
            score=float(is_correct),
            extra={
                "expected": expected_answer,
                "guess": user_guess,
                "correct": is_correct
            }
        )