import time
import random, re
import affine as af

class SAT(af.BaseEnv):
    __version__: str = "0.0.2"
    n: int
    k: int
    m: int
    def __init__(self, n=15, k=10, m=None):
        super().__init__(n=n, k=k, m=m or int(4.26 * n))
        
    async def generate(self):
        sol = {i: random.choice([True, False]) for i in range(1, self.n+1)}
        cls = []
        for _ in range(self.m):
            vs = random.sample(list(sol), self.k)
            sv = random.choice(vs)
            cls.append([(lit := (v if sol[v] else -v)) if v==sv else (v if random.choice([True,False]) else -v) for v in vs])
        formula = " ∧ ".join("(" + " ∨ ".join(f"{'' if l>0 else '¬'}x{abs(l)}" for l in c) + ")" for c in cls)
        prompt = (
            f"Find a satisfying assignment for the following {self.k}-SAT formula over variables x1..x{self.n}:\n"
            f"{formula}\n"
            "Provide your answer as comma-separated assignments like `x1=True, x2=False, ...`, "
            "or respond `UNSAT` if it has no solution."
        )
        return af.Challenge(env=self, prompt=prompt, extra={"sol": sol, "cls": cls})        

    async def evaluate(self, challenge: af.Challenge, response: af.Response):
        sol, cls = challenge.extra["sol"], challenge.extra["cls"]
        got = {int(v): val.lower() in ("true","1")
               for v, val in re.findall(r"x(\d+)=(True|False|1|0)", (response.response or ""))}
        ok = all(any((lit>0)==got.get(abs(lit), None) for lit in c) for c in cls)
        return af.Evaluation(env=self, score=float(ok), extra={"expected": sol, "got": got})
