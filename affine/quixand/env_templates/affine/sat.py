#!/usr/bin/env python3

import re
import time
from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from pydantic import BaseModel
from utils import CreateResponse, EvaluateRequest, EvaluateResponse


app = FastAPI()


class CreateRequest(BaseModel):
    n: Optional[int] = None
    k: Optional[int] = None
    m: Optional[int] = None


def _mk(n: int, k: int, m: Optional[int]) -> tuple[str, Dict[str, Any]]:
    import random
    n = n or 15
    k = k or 10
    m = m or int(4.26 * n)
    sol = {i: random.choice([True, False]) for i in range(1, n + 1)}
    cls = []
    for _ in range(m):
        vs = random.sample(list(sol), k)
        sv = random.choice(vs)
        lits = []
        for v in vs:
            if v == sv:
                lits.append(v if sol[v] else -v)
            else:
                lits.append(v if random.choice([True, False]) else -v)
        cls.append(lits)
    formula = " ∧ ".join("(" + " ∨ ".join(f"{'' if l>0 else '¬'}x{abs(l)}" for l in c) + ")" for c in cls)
    prompt = (
        f"Find a satisfying assignment for the following {k}-SAT formula over variables x1..x{n}:\n"
        f"{formula}\n"
        "Provide your answer as comma-separated assignments like `x1=True, x2=False, ...`, or respond `UNSAT` if it has no solution."
    )
    return prompt, {"sol": sol, "cls": cls, "n": n, "k": k, "m": m}


@app.get("/health")
async def health():
    return "ok"


@app.post("/create", response_model=CreateResponse)
async def create_endpoint(payload: CreateRequest):
    prompt, extra = _mk(payload.n or 15, payload.k or 10, payload.m)
    return CreateResponse(prompt=prompt, extra=extra)


@app.post("/evaluate", response_model=EvaluateResponse)
async def evaluate_endpoint(payload: EvaluateRequest):
    sol = payload.extra.get("sol") or {}
    cls = payload.extra.get("cls") or []
    text = payload.response or ""
    got = {int(v): val.lower() in ("true", "1") for v, val in re.findall(r"x(\d+)=(True|False|1|0)", text)}
    ok = all(any(((lit > 0) == got.get(abs(lit), None)) for lit in c) for c in cls)
    return EvaluateResponse(score=1.0 if ok else 0.0, extra={"expected": sol, "got": got})


 


