OPEN = "\\boxed{"


# ---------- live: final-channel projection (mimo_answer_projection.project / _boxed_final) ----------
def _box_end(text, start):
    depth = 1
    for at in range(start, len(text)):
        if text[at] == "{":
            depth += 1
        elif text[at] == "}":
            depth -= 1
            if depth == 0:
                return at
    return None


def live_project(text):
    """Return (grading_text, reason). grading_text is None when the live rule marks the reply unresolved."""
    if text.count("<think>") != 1 or text.count("</think>") != 1:
        return None, "think"
    a, b = text.index("<think>"), text.index("</think>")
    if a >= b or text[:a].strip():
        return None, "think"
    final = text[b + len("</think>"):].strip()
    if final.count(OPEN) != 1:
        return None, "n_boxed"
    begin = final.index(OPEN)
    at = _box_end(final, begin + len(OPEN))
    if at is None or not final[begin + len(OPEN):at].strip():
        return None, "unclosed"
    tail, prefix = final[at + 1:].strip(), final[:begin].rstrip()
    if tail:
        opening = {"\\]": "\\[", "\\)": "\\(", "$$": "$$", "$": "$"}.get(tail)
        if opening is None or not prefix.endswith(opening):
            return None, "tail"
    elif prefix.endswith(("\\[", "\\(", "$")):
        return None, "prefix"
    return final[begin:at + 1], "ok"


# ---------- fixes: projection ----------
def _unclosed_dollar(prefix):
    """True if the text before the box leaves a `$` / `$$` open (parity of unescaped `$`)."""
    return re.sub(r"\\\$", "", prefix).count("$") % 2 == 1


def lenient_project(text):
    """Return (grading_text, format_violation, fix). format_violation is None when the reply follows the protocol."""
    box, why = live_project(text)
    if box is not None:
        return box, None, None
    if why == "prefix":  # P1
        final = text.split("</think>")[-1].strip()
        begin = final.index(OPEN)
        if not _unclosed_dollar(final[:begin]) and not final[:begin].rstrip().endswith(("\\[", "\\(")):
            at = _box_end(final, begin + len(OPEN))
            return final[begin:at + 1], None, "P1"
    if why in ("tail", "n_boxed", "think", "prefix"):  # P2
        # no <think>: the whole reply is the answer channel; several </think>: use the last segment
        final = text.split("</think>")[-1]
        at0 = final.rfind(OPEN)
        if at0 >= 0:
            at = _box_end(final, at0 + len(OPEN))
            if at is not None and final[at0 + len(OPEN):at].strip():
                return final[at0:at + 1], why, "P2"
    return None, why, None


# ---------- live: verify.py ----------
_cache = {}


def mv_equal(gold, pred_box):
    """verify.py: True / False, or None where verify.py exits 75 (indeterminate)."""
    key = (gold, pred_box)
    if key not in _cache:
        try:
            g = parse(OPEN + gold + "}", raise_on_error=True)
            p = parse(pred_box, raise_on_error=True)
            _cache[key] = bool(g) and bool(verify(g, p, raise_on_error=True))
        except Exception:
            _cache[key] = None
    return _cache[key]


# ---------- fixes: answer equivalence ----------
def inner(box):
    return box[len(OPEN):-1]


def strip_text(s):
    s = re.sub(r"\\(?:text|textbf|mathrm|mbox)\s*\{([^{}]*)\}", r"\1", s)
    for _ in range(2):  # 1,000,000 needs two passes
        s = re.sub(r"(\d),\\!(\d{3})", r"\1\2", s)
        s = re.sub(r"(\d)\{,\}(\d{3})", r"\1\2", s)
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    return s.strip()


LETTERS = re.compile(r"^\(?([A-Z])\)?(\s*,\s*\(?([A-Z])\)?)*$")


def letter_set(s):
    s = strip_text(s).replace("\\ ", " ").strip().strip("$")
    if LETTERS.match(s.replace(" ", "")):
        return frozenset(re.findall(r"[A-Z]", s))
    return None


def mc_options(prompt):
    """Options written as \\textbf{(A)}\\ xxx or \\text A. xxx -> {letter: text}."""
    opts = {}
    for m in re.finditer(r"\\textbf\{\(([A-E])\)\}\\?\s*(.*?)(?=\\qquad|\\textbf\{\([A-E]\)\}|\$\s*$|$)", prompt, re.S):
        opts[m.group(1)] = m.group(2).strip().rstrip("$").strip()
    if not opts:
        for m in re.finditer(r"\\text\s*([A-E])\.\s*(?:\\ )*\s*(.*?)(?=\\qquad|\\text\s*[A-E]\.|\$\$|$)", prompt, re.S):
            opts[m.group(1)] = m.group(2).strip()
    return opts


def pmatrix_to_tuple(s):
    def col(m):
        rows = [r.strip() for r in m.group(1).split("\\\\")]
        return "(" + ", ".join(rows) + ")" if all("&" not in r for r in rows) else m.group(0)
    return re.sub(r"\\begin\{[pb]matrix\}(.*?)\\end\{[pb]matrix\}", col, s, flags=re.S)


def ineq_to_interval(s):
    s = strip_text(s).replace(" ", "")
    m = re.fullmatch(r"[a-z](?:\\neq|\\ne)(-?[\w\\{}.]+)", s)
    if m:
        c = m.group(1)
        return f"(-\\infty,{c})\\cup({c},\\infty)"
    m = re.fullmatch(r"(-?[\w\\{}.]+)<[a-z]<(-?[\w\\{}.]+)", s)
    if m:
        return f"({m.group(1)},{m.group(2)})"
    m = re.fullmatch(r"[a-z](<|>|\\le|\\ge|\\leq|\\geq)(-?[\w\\{}.]+)", s)
    if m:
        op, c = m.groups()
        return {"<": f"(-\\infty,{c})", ">": f"({c},\\infty)", "\\le": f"(-\\infty,{c}]", "\\leq": f"(-\\infty,{c}]",
                "\\ge": f"[{c},\\infty)", "\\geq": f"[{c},\\infty)"}[op]
    return None


def scalar_identity(pred, gold):
    m = re.fullmatch(r"(-?\d+)\s*\\?(?:mathbf\{I\}|I)", strip_text(pred).replace(" ", ""))
    g = re.fullmatch(r"\\begin\{pmatrix\}(.*)\\end\{pmatrix\}", gold.strip(), re.S)
    if not (m and g):
        return None
    n, k = len(g.group(1).split("\\\\")), m.group(1)
    rows = [" & ".join(k if i == j else "0" for j in range(n)) for i in range(n)]
    return "\\begin{pmatrix} " + " \\\\ ".join(rows) + " \\end{pmatrix}"


def complex_equal(pred, gold):
    try:
        import sympy
        from latex2sympy2_extended import latex2sympy
        a = latex2sympy(strip_text(pred)).subs(sympy.Symbol("i"), sympy.I)
        b = latex2sympy(strip_text(gold)).subs(sympy.Symbol("i"), sympy.I)
        return sympy.simplify(sympy.expand(a - b)) == 0
    except Exception:
        return False


def lenient_equal(gold, pred_box, prompt):
    """Return (correct, fix). The live verify.py is tried first; fixes only run when it says no."""
    if mv_equal(gold, pred_box):
        return True, None
    pred = inner(pred_box)
    if mv_equal(strip_text(gold), OPEN + strip_text(pred) + "}"):
        return True, "E1"
    gs, ps = letter_set(gold), letter_set(pred)
    if gs and ps and gs == ps:
        return True, "E2"
    opts = mc_options(prompt)
    if opts:
        p = strip_text(pred)
        if re.match(r"^\s*\\?(?:text|textbf)?\{?\(?[A-E]\)", p):
            p = re.sub(r"^\s*\\?(?:text|textbf)?\{?\(?([A-E])\)?\}?\s*(\\\s)?\s*", "", p)
        ls = letter_set(pred)
        if ls and len(ls) == 1:
            p = opts.get(next(iter(ls)), p)
        if p and mv_equal(gold, OPEN + p + "}"):
            return True, "E3"
    if mv_equal(pmatrix_to_tuple(gold), OPEN + pmatrix_to_tuple(pred) + "}"):
        return True, "E4"
    iv = ineq_to_interval(pred)
    if iv and mv_equal(gold, OPEN + iv + "}"):
        return True, "E5"
    m = scalar_identity(pred, gold)
    if m and mv_equal(gold, OPEN + m + "}"):
        return True, "E6"
    if "i" in gold and "i" in pred and complex_equal(pred, gold):
        return True, "E7"
    return False, None
