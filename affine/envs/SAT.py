"""
SAT environment - Boolean satisfiability problem solving.
"""
import random
import re
import logging
from typing import Dict, List, Optional
from pydantic import model_validator

from .. import BaseEnv, Challenge, Response, Evaluation

logger = logging.getLogger("affine")


class SAT(BaseEnv):
    """Boolean satisfiability problem solving environment."""
    
    # Pydantic fields (not constructor parameters)
    name: str = "SAT"
    n: int = 3  # number of variables
    k: int = 2  # clause size (literals per clause)
    m: Optional[int] = None  # number of clauses
    
    @model_validator(mode='after')
    def validate_sat_params(self):
        """Validate SAT parameters."""
        if self.k > self.n:
            raise ValueError(f"Clause size k ({self.k}) cannot be larger than number of variables n ({self.n})")
        if self.m is None:
            self.m = int(4.26 * self.n)  # Standard SAT phase transition
        return self
        
    async def generate(self) -> Challenge:
        """Generate a SAT problem with a known solution."""
        logger.debug(f"Generating SAT problem: n={self.n}, k={self.k}, m={self.m}")
        
        # Generate a random solution (planted solution approach)
        solution = {i: random.choice([True, False]) for i in range(1, self.n + 1)}
        
        # Generate clauses that are satisfied by the solution
        clauses = []
        for _ in range(self.m):
            variables = random.sample(list(solution), self.k)
            # Choose one variable to satisfy the clause
            satisfying_var = random.choice(variables)
            
            clause = []
            for var in variables:
                if var == satisfying_var:
                    # This literal will make the clause true
                    literal = var if solution[var] else -var
                else:
                    # Random literal
                    literal = var if random.choice([True, False]) else -var
                clause.append(literal)
            
            clauses.append(clause)
        
        # Format the problem as a string
        problem_text = self._format_sat_problem(clauses)
        
        prompt = (
            f"Find a satisfying assignment for the following {self.k}-SAT formula "
            f"over variables x1..x{self.n}:\n"
            f"{problem_text}\n"
            "Provide your answer as comma-separated assignments like 'x1=True, x2=False, ...' "
            "or respond 'UNSAT' if it has no solution."
        )
        
        logger.debug(f"Generated SAT challenge with planted solution: {solution}")
        
        return Challenge(
            env=self,
            prompt=prompt,
            extra={"solution": solution, "clauses": clauses}
        )
    
    def _format_sat_problem(self, clauses: List[List[int]]) -> str:
        """Format clauses as a readable SAT problem."""
        clause_strings = []
        for clause in clauses:
            literals = []
            for literal in clause:
                if literal > 0:
                    literals.append(f"x{literal}")
                else:
                    literals.append(f"¬x{abs(literal)}")
            clause_strings.append(f"({' ∨ '.join(literals)})")
        
        return " ∧ ".join(clause_strings)
    
    async def evaluate(self, challenge: Challenge, response: Response) -> Evaluation:
        """Evaluate the SAT solution."""
        expected_solution = challenge.extra["solution"]
        clauses = challenge.extra["clauses"]
        
        logger.debug(f"Evaluating SAT response: {response.response}")
        
        # Handle None responses
        if response.response is None:
            logger.debug("Response is None, marking as incorrect")
            return Evaluation(
                env=self,
                score=0.0,
                extra={
                    "expected": expected_solution,
                    "provided": None,
                    "satisfiable": False,
                    "reason": "No response from model"
                }
            )
        
        resp = response.response.strip()
        
        # Check for UNSAT claim (always incorrect since we planted a solution)
        if re.search(r'\bunsat\b', resp, re.IGNORECASE):
            logger.debug("Response claims UNSAT, which is incorrect (solution was planted)")
            return Evaluation(
                env=self,
                score=0.0,
                extra={
                    "expected": expected_solution,
                    "provided": "UNSAT",
                    "satisfiable": False,
                    "reason": "Claimed UNSAT but solution exists"
                }
            )
        
        # Parse the user's solution
        user_solution = self._parse_solution(resp)
        logger.debug(f"Parsed user solution: {user_solution}")
        
        # Check if all variables are assigned
        if set(user_solution.keys()) != set(expected_solution.keys()):
            logger.debug(f"Incomplete assignment: got {set(user_solution.keys())}, expected {set(expected_solution.keys())}")
            return Evaluation(
                env=self,
                score=0.0,
                extra={
                    "expected": expected_solution,
                    "provided": user_solution,
                    "satisfiable": False,
                    "reason": "Incomplete variable assignment"
                }
            )
        
        # Check if the solution satisfies all clauses
        is_satisfiable = self._check_satisfiability(clauses, user_solution)
        
        logger.debug(f"Solution evaluation: satisfiable={is_satisfiable}")
        
        return Evaluation(
            env=self,
            score=float(is_satisfiable),
            extra={
                "expected": expected_solution,
                "provided": user_solution,
                "satisfiable": is_satisfiable
            }
        )
    
    def _parse_solution(self, response_text: str) -> Dict[int, bool]:
        """Parse variable assignments from response text."""
        solution = {}
        
        # Look for patterns like x1=True, x2=False, x3=1, x4=0
        patterns = [
            r"x(\d+)\s*=\s*(True|False|true|false)",
            r"x(\d+)\s*=\s*([01])"
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, response_text)
            for var_str, val_str in matches:
                var_num = int(var_str)
                if val_str.lower() in ("true", "1"):
                    solution[var_num] = True
                elif val_str.lower() in ("false", "0"):
                    solution[var_num] = False
        
        return solution
    
    def _check_satisfiability(self, clauses: List[List[int]], solution: Dict[int, bool]) -> bool:
        """Check if the solution satisfies all clauses."""
        for clause in clauses:
            clause_satisfied = False
            for literal in clause:
                var = abs(literal)
                var_value = solution.get(var)
                
                if var_value is None:
                    continue  # Variable not assigned
                
                # Check if this literal is satisfied
                if (literal > 0 and var_value) or (literal < 0 and not var_value):
                    clause_satisfied = True
                    break
            
            if not clause_satisfied:
                logger.debug(f"Clause {clause} not satisfied by solution {solution}")
                return False
        
        return True
