"""Stronger math answer grader (affine-math-grader v1).

grade(pred, gold, ...) -> Grade(score, needs_judge, how)

Order: exact normalized match -> multi-part (top-level comma split; order-insensitive when the gold is a set or the problem
has multiple answers) -> intervals / unions -> tuples / sets -> numeric with tolerance -> math-verify -> sympy simplify(a-b)==0.
needs_judge=True marks a miss the rules cannot decide (free text, unparsable expression on either side, equations with
different variable layout); those rows are scored 0 but counted separately so a later LLM-judge pass can revisit them."""
from __future__ import annotations

import re
from dataclasses import dataclass

VERSION = "affine-math-grader-v1"


@dataclass(frozen=True)
class Grade:
    score: float
    needs_judge: bool = False
    how: str = ""


# ------------------------------------------------------------------------------------------------ extraction
def last_boxed(text: str | None) -> str | None:
    """Content of the last \\boxed{...} / \\fbox{...} after the final </think> (balanced braces)."""
    if not text:
        return None
    if "<think>" in text and "</think>" not in text:
        return None
    text = text.rsplit("</think>", 1)[-1]
    best = None
    for m in re.finditer(r"\\(?:boxed|fbox)\s*\{", text):
        depth, j = 0, m.end() - 1
        for k in range(j, len(text)):
            if text[k] == "{":
                depth += 1
            elif text[k] == "}":
                depth -= 1
                if depth == 0:
                    best = text[j + 1:k]
                    break
    return best


# ------------------------------------------------------------------------------------------------ normalization
_UNIT_WORDS = (r"(?:square\s+)?(?:units?|cm|centimeters?|meters?|m|km|kilometers?|inches|inch|in|feet|foot|ft|miles?|mi|degrees?|deg|"
               r"seconds?|sec|s|minutes?|min|hours?|hrs?|h|days?|weeks?|years?|dollars?|cents?|mph|kg|g|grams?|pounds?|lbs?|liters?|L|ml|"
               r"rad|radians?|people|students|ways|points?|cm\^2|m\^2|cm\^3|m\^3)")


def strip_units(s: str, unit: str | None = None) -> str:
    if unit:
        u = re.escape(unit.strip().strip("$"))
        s = re.sub(r"\\(?:text|mathrm|mbox)\{\s*" + u + r"\s*\}", "", s)
        s = re.sub(r"\s*" + u + r"\s*$", "", s)
    s = re.sub(r"\\(?:text|mathrm|mbox|textbf)\{\s*" + _UNIT_WORDS + r"\s*\}(\^\{?\d\}?)?", "", s)
    s = re.sub(r"(?<=[\d}])\s*" + _UNIT_WORDS + r"(\^\{?\d\}?)?\s*$", "", s)
    s = re.sub(r"\^\s*\{?\\circ\}?", "", s).replace("°", "")
    return s


def normalize(s: str, unit: str | None = None) -> str:
    s = str(s).strip()
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac").replace("\\left", "").replace("\\right", "")
    s = s.replace("\\!", "").replace("\\,", "").replace("\\;", "").replace("\\ ", "").replace("~", "")
    s = s.replace("$", "").replace("\\displaystyle", "")
    s = strip_units(s, unit)
    s = re.sub(r"\\%|%", "", s)
    s = re.sub(r"^\s*\\text\{\s*(.*?)\s*\}\s*$", r"\1", s)
    s = re.sub(r"^(?:[a-zA-Z]\s*(?:=|\\in)\s*)(?=[^=]+$)", "", s)       # x = 3  ->  3 ; x \in [1,2] -> [1,2]
    s = re.sub(r"(\d),(?=\d{3}(?!\d))", r"\1", s)                        # 1,000 -> 1000
    s = s.rstrip(".").strip()
    s = re.sub(r"\s+", " ", s)
    return s


# ------------------------------------------------------------------------------------------------ splitting
def split_top(s: str, seps: str = ",;") -> list[str]:
    out, depth, cur = [], 0, ""
    i = 0
    while i < len(s):
        c = s[i]
        if s.startswith("\\{", i) or s.startswith("\\}", i):
            depth += 1 if s[i + 1] == "{" else -1
            cur += s[i:i + 2]; i += 2; continue
        if c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
        if c in seps and depth == 0:
            out.append(cur.strip()); cur = ""
        else:
            cur += c
        i += 1
    if cur.strip():
        out.append(cur.strip())
    return out


def _strip_and(s: str) -> str:
    return re.sub(r"\s*(?:\\text\{\s*and\s*\}|\band\b|\\quad)\s*", ",", s)


_INTERVAL = re.compile(r"^\s*([\(\[])\s*(.+?)\s*,\s*(.+?)\s*([\)\]])\s*$")


def as_union(s: str):
    parts = [p.strip() for p in re.split(r"\\cup|∪|\\text\{\s*or\s*\}|\bU\b", s)]
    out = []
    for p in parts:
        m = _INTERVAL.match(p)
        if not m or len(split_top(m.group(2) + "," + m.group(3))) != 2:
            return None
        out.append((m.group(1), m.group(2), m.group(3), m.group(4)))
    return out


def as_collection(s: str):
    """('set'|'tuple', [items]) or None."""
    t = s.strip()
    if t.startswith("\\{") and t.endswith("\\}"):
        return "set", split_top(t[2:-2])
    if t.startswith("{") and t.endswith("}") and "," in t:
        return "set", split_top(t[1:-1])
    if (t.startswith("(") and t.endswith(")")) or (t.startswith("\\langle") and t.endswith("\\rangle")):
        inner = t[1:-1] if t.startswith("(") else t[len("\\langle"):-len("\\rangle")]
        items = split_top(inner)
        if len(items) >= 2:
            return "tuple", items
    return None


# ------------------------------------------------------------------------------------------------ scalar equivalence
def _sympy(s: str):
    from latex2sympy2_extended import latex2sympy
    return latex2sympy(s)


def numeric(s: str):
    try:
        return float(s.replace(" ", ""))
    except ValueError:
        pass
    try:
        v = _sympy(s)
        if getattr(v, "free_symbols", None):
            return None
        return float(v.evalf())
    except Exception:
        return None


def scalar_equal(a: str, b: str, tol: float | None = None) -> tuple[bool, bool, str]:
    """(equal, undecidable, how) for two normalized scalar answers."""
    if a == b:
        return True, False, "exact"
    if a.replace(" ", "").lower() == b.replace(" ", "").lower():
        return True, False, "exact-ci"
    na, nb = numeric(a), numeric(b)
    if na is not None and nb is not None:
        rel = tol if tol is not None else 1e-6
        ok = abs(na - nb) <= max(rel * max(abs(na), abs(nb)), 1e-9 if tol is None else rel)
        return ok, False, "numeric"
    try:
        from math_verify import parse, verify
        pa, pb = parse("\\boxed{" + a + "}", parsing_timeout=5), parse("\\boxed{" + b + "}", parsing_timeout=5)
        if pa and pb and verify(pb, pa, timeout_seconds=5):
            return True, False, "math-verify"
    except Exception:
        pass
    if _textual(a) or _textual(b):
        return False, True, "undecidable-text"
    try:
        import sympy
        ea, eb = _sympy(a), _sympy(b)
        if isinstance(ea, sympy.Equality) and isinstance(eb, sympy.Equality):
            ea, eb = ea.lhs - ea.rhs, eb.lhs - eb.rhs
            q = sympy.simplify(ea / eb) if eb != 0 else None
            if q is not None and q.is_number and q != 0:
                return True, False, "sympy-eq-scaled"
        if sympy.simplify(ea - eb) == 0:
            return True, False, "sympy"
        return False, False, "sympy"
    except Exception:
        pass
    return False, True, "undecidable-parse"


def _textual(s: str) -> bool:
    """Free text (a word of 3+ letters that is not a LaTeX command or a known function name)."""
    words = re.findall(r"[A-Za-z]{3,}", re.sub(r"\\[A-Za-z]+", " ", s))
    return any(w.lower() not in _FUNCS for w in words)


_FUNCS = {"sin", "cos", "tan", "cot", "sec", "csc", "log", "exp", "max", "min", "gcd", "lcm", "mod", "arcsin", "arccos", "arctan",
          "sinh", "cosh", "tanh", "det", "deg", "inf", "sqrt"}


# ------------------------------------------------------------------------------------------------ main entry
def grade(pred: str | None, gold: str | list, *, unit: str | None = None, tolerance: float | None = None,
          multiple: bool = False, ordered: bool | None = None) -> Grade:
    """pred: the model's extracted final answer (e.g. last_boxed(reply)); gold: string or list of strings (multi-part)."""
    if pred is None or not str(pred).strip():
        return Grade(0.0, False, "no-answer")
    golds = gold if isinstance(gold, list) else None
    a = normalize(_strip_and(pred) if multiple or golds else pred, unit)
    if golds is not None and len(golds) == 1:
        golds, gold = None, golds[0]
    if golds is None:
        b = normalize(_strip_and(gold) if multiple else gold, unit)
        r = _grade_one(a, b, tolerance)
        if r.score or not multiple:
            return r
        golds = split_top(b)
    # multi-part: compare as a multiset (order-insensitive unless ordered=True)
    gs = [normalize(g, unit) for g in golds]
    ps = split_top(a)
    if len(ps) != len(gs):
        return Grade(0.0, False, f"multi-count {len(ps)}!={len(gs)}")
    if ordered:
        res = [_grade_one(p, g, tolerance) for p, g in zip(ps, gs)]
        return Grade(float(all(r.score for r in res)), any(r.needs_judge for r in res) and not all(r.score for r in res), "multi-ordered")
    left, judge = list(ps), False
    for g in gs:
        hit = None
        for i, p in enumerate(left):
            r = _grade_one(p, g, tolerance)
            judge |= r.needs_judge
            if r.score:
                hit = i; break
        if hit is None:
            return Grade(0.0, judge, "multi-miss")
        left.pop(hit)
    return Grade(1.0, False, "multi")


def _grade_one(a: str, b: str, tol: float | None) -> Grade:
    ok, und, how = scalar_equal(a, b, tol) if a == b else (False, False, "")
    if ok:
        return Grade(1.0, False, how)
    ua, ub = as_union(a), as_union(b)
    if ua is not None and ub is not None:
        if len(ua) != len(ub):
            return Grade(0.0, False, "interval-count")
        for (l1, x1, y1, r1), (l2, x2, y2, r2) in zip(sorted(ua, key=str), sorted(ub, key=str)):
            if l1 != l2 or r1 != r2 or not scalar_equal(normalize(x1), normalize(x2), tol)[0] or not scalar_equal(normalize(y1), normalize(y2), tol)[0]:
                return Grade(0.0, False, "interval")
        return Grade(1.0, False, "interval")
    ca, cb = as_collection(a), as_collection(b)
    if ca and cb:
        (ka, ia), (kb, ib) = ca, cb
        if len(ia) != len(ib):
            return Grade(0.0, False, "collection-count")
        if kb == "tuple" and ka == "tuple":
            ok = all(scalar_equal(normalize(x), normalize(y), tol)[0] for x, y in zip(ia, ib))
            return Grade(float(ok), False, "tuple")
        left = list(ia)
        for y in ib:
            j = next((i for i, x in enumerate(left) if scalar_equal(normalize(x), normalize(y), tol)[0]), None)
            if j is None:
                return Grade(0.0, False, "set")
            left.pop(j)
        return Grade(1.0, False, "set")
    ok, und, how = scalar_equal(a, b, tol)
    return Grade(float(ok), und and not ok, how)
