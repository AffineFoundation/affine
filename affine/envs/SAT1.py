"""
SAT1 environment - Advanced Boolean satisfiability problem solving.
Based on the functional implementation with planted solutions and robust verification.
"""
import re
import random
import logging
from typing import Any, Dict, List, Optional
from pydantic import model_validator, PrivateAttr

from .. import BaseEnv, Challenge, Response, Evaluation

logger = logging.getLogger("affine")


class SAT1(BaseEnv):
    """
    Advanced SAT environment that generates random difficult boolean SAT problems 
    with planted solutions. Difficulty can be controlled via constructor arguments.
    
    This implementation uses a more sophisticated approach than the basic SAT
    environment, with better validation and challenge tracking.
    """
    name: str = "SAT1"
    n: int = 3  # number of variables
    k: int = 2  # clause size
    m: Optional[int] = None  # number of clauses
    
    # Private attributes for internal state
    _questions: List[str] = PrivateAttr(default_factory=list)
    _formulas: List[List[List[int]]] = PrivateAttr(default_factory=list)
    _assignments: List[Dict[int, bool]] = PrivateAttr(default_factory=list)
    _idx: int = PrivateAttr(default=0)

    @model_validator(mode='after')
    def validate_env(self):
        """Validate SAT1 parameters."""
        if self.k > self.n:
            raise ValueError(f"Clause size k ({self.k}) cannot be larger than the number of variables n ({self.n}).")
        if self.m is None:
            self.m = int(4.26 * self.n)  # Standard SAT phase transition ratio
        return self

    async def generate(self) -> Challenge:
        """Generate a SAT challenge with planted solution."""
        logger.debug(f"Generating SAT1 problem: n={self.n}, k={self.k}, m={self.m}")
        
        # 1) Plant a random solution
        assignment = {i: random.choice([True, False]) for i in range(1, self.n + 1)}
        
        # 2) Generate clauses ensuring each is satisfied by the planted solution
        clauses: List[List[int]] = []
        for _ in range(self.m):
            vars_in_clause = random.sample(range(1, self.n + 1), self.k)
            # Pick one variable to guarantee satisfaction
            sat_var = random.choice(vars_in_clause)
            clause: List[int] = []
            for v in vars_in_clause:
                if v == sat_var:
                    # Make the literal agree with the planted assignment
                    lit = v if assignment[v] else -v
                else:
                    # Other literals random
                    lit = v if random.choice([True, False]) else -v
                clause.append(lit)
            clauses.append(clause)
        
        # Store for later verification
        self._formulas.append(clauses)
        self._assignments.append(assignment)
        
        # Build a human-readable CNF string
        clause_strs = []
        for clause in clauses:
            lits = []
            for lit in clause:
                var = f"x{abs(lit)}"
                if lit < 0:
                    lits.append(f"¬{var}")
                else:
                    lits.append(var)
            clause_strs.append("(" + " ∨ ".join(lits) + ")")
        formula_str = " ∧ ".join(clause_strs)
        
        prompt = (
            f"Find a satisfying assignment for the following {self.k}-SAT formula over variables x1..x{self.n}:\n"
            f"{formula_str}\n"
            "Provide your answer as comma-separated assignments like `x1=True, x2=False, ...`, "
            "or respond `UNSAT` if it has no solution."
        )
        
        self._questions.append(prompt)
        logger.debug(f"Generated SAT1 question #{self._idx}: n={self.n}, m={self.m}, planted={assignment}")
        self._idx += 1
        
        return Challenge(
            env=self,
            prompt=prompt,
            extra={"solution": assignment, "clauses": clauses, "formula": formula_str}
        )

    async def evaluate(self, challenge: Challenge, response: Response) -> Evaluation:
        """Evaluate the SAT1 solution with robust verification."""
        logger.debug("Starting SAT1 verification...")
        
        # Locate the corresponding formula & assignment
        try:
            idx = self._questions.index(challenge.prompt)
            logger.debug(f"Found question at index {idx}")
        except ValueError:
            logger.warning(f"Unknown question in verification")
            return Evaluation(
                env=self,
                score=0.0,
                extra={"correct": False, "expected": None, "extraction": None, "reason": "Unknown question"}
            )

        clauses = self._formulas[idx]
        expected = self._assignments[idx]
        logger.debug(f"Loaded expected assignment: {expected}")

        # Guard against None responses
        if response.response is None:
            logger.debug("Response is None, marking as incorrect.")
            return Evaluation(
                env=self,
                score=0.0,
                extra={"correct": False, "expected": expected, "extraction": None, "reason": "No response from model"}
            )

        resp = response.response.strip()
        logger.debug(f"Verifying response: '{resp}'")
        
        # If user claims UNSAT, that's always incorrect (we planted a solution)
        if re.search(r'\bunsat\b', resp, re.IGNORECASE):
            logger.debug("Response claims UNSAT, which is incorrect as a solution was planted.")
            return Evaluation(
                env=self,
                score=0.0,
                extra={"correct": False, "expected": expected, "extraction": None, "reason": "Claimed UNSAT but solution exists"}
            )

        # Extract assignments of the form x<number>=True/False or x<number>=1/0
        matches = re.findall(r"x(\d+)\s*=\s*(True|False|true|false|1|0)", resp)
        logger.debug(f"Found {len(matches)} variable assignments in response.")
        
        extraction: Dict[int, bool] = {}
        for var_str, val_str in matches:
            var_index = int(var_str)
            val = val_str.lower() in ("true", "1")
            extraction[var_index] = val
        logger.debug(f"Extracted assignment: {extraction}")

        # Must assign every variable
        if set(extraction.keys()) != set(expected.keys()):
            logger.debug(f"Extracted variables {set(extraction.keys())} do not match expected variables {set(expected.keys())}.")
            return Evaluation(
                env=self,
                score=0.0,
                extra={"correct": False, "expected": expected, "extraction": extraction, "reason": "Incomplete variable assignment"}
            )

        # Evaluate each clause under the extracted assignment
        logger.debug("All variables assigned. Evaluating clauses...")
        
        def clause_satisfied(cl: List[int]) -> bool:
            for lit in cl:
                v = abs(lit)
                val = extraction[v]
                satisfied = (lit > 0 and val) or (lit < 0 and not val)
                if satisfied:
                    return True
            return False
        
        clause_results = [clause_satisfied(cl) for cl in clauses]
        correct = all(clause_results)
        
        if not correct:
            failed_clauses = [clauses[i] for i, satisfied in enumerate(clause_results) if not satisfied]
            logger.debug(f"Evaluation failed. {len(failed_clauses)}/{len(clauses)} clauses were not satisfied.")
            logger.debug(f"First failing clause: {failed_clauses[0]}")

        logger.debug(f"Verification result: correct={correct}")
        
        return Evaluation(
            env=self,
            score=float(correct),
            extra={
                "correct": correct, 
                "expected": expected, 
                "extraction": extraction,
                "satisfied_clauses": sum(clause_results),
                "total_clauses": len(clauses)
            }
        ) 