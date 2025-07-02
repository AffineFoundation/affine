import random
import re
import affine as af

class COIN(af.BaseEnv):
    async def generate(self) -> af.Challenge:
        return af.Challenge(
            env=self,
            prompt="""I flipped a coin, guess HEADS or TAILS.

Please provide your guess in the following format:
<GUESS>HEADS</GUESS>
or
<GUESS>TAILS</GUESS>

You can include any reasoning or explanation you want, but make sure to include your final answer within the <GUESS> </GUESS> tags.""",
            extra={"answer": random.choice(["HEADS", "TAILS"])}
        )

    def extract_guess_from_response(self, response: str) -> str:
        """Extract guess from <GUESS> </GUESS> tags"""
        # Remove thinking tags first if any
        response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
        response = re.sub(r'<thinking>.*?</thinking>', '', response, flags=re.DOTALL)
        
        # Look for <GUESS> </GUESS> tags - case insensitive
        guess_pattern = r'<GUESS>(.*?)</GUESS>'
        matches = re.findall(guess_pattern, response, re.DOTALL | re.IGNORECASE)
        
        if matches:
            # Take the last match if multiple exist and clean it
            guess = matches[-1].strip().upper()
            # Only return valid guesses
            if guess in ["HEADS", "TAILS"]:
                return guess
        
        # Fallback: look for HEADS or TAILS in the response
        response_upper = response.upper()
        if "HEADS" in response_upper and "TAILS" not in response_upper:
            return "HEADS"
        elif "TAILS" in response_upper and "HEADS" not in response_upper:
            return "TAILS"
        
        # If we can't determine, return empty string
        return ""

    async def evaluate(self, challenge: af.Challenge, response: af.Response) -> af.Evaluation:
        ans = challenge.extra["answer"]
        guess = self.extract_guess_from_response(response.response or "")
        
        return af.Evaluation(
            env=self,
            score=float(guess == ans),
            extra={"answer": ans, "guess": guess}
        )