"""Knowledge-base retrieval for the tau2-synth domains (env wave 5, 2026-09-20).

τ³ (tau2-bench `58e5e1a`, domain `banking_knowledge`) moves the agent policy
out of the system prompt into a document store behind a `KB_search` tool
(BM25, offline) and keeps only a generic header in the prompt: "do not make
up policies; all instructions are in the knowledge base". Its `KBSearchMixin`
adds the tool to any τ² toolkit through the `ToolKitType` metaclass (tools
are collected along the MRO). The pods run the tau2-synth fork of τ²
(ancestor `337326e`), which has no `tau2.knowledge`; the parts needed here
are small, so they are re-implemented rather than vendored:

  * `chunk_policy`   the domain's `policy.md` -> one document per `##`
                      section (`###` sub-sections become their own documents,
                      titled "<section> / <sub-section>"), ids
                      `doc_<domain>_<slug>_NNN` like τ³'s;
  * `BM25`            the standard Okapi BM25 (k1 1.5, b 0.75) over a
                      lower-cased word tokenizer; no dependency;
  * `KBSearchMixin`  τ³'s tool, same name, same docstring, same output
                      layout ("1. <title> / ID / Score / Content"); top_k 5;
  * `kb_policy`       τ³'s policy header re-worded for the domain, plus the
                      one line that names the tool;
  * `install`         swap the domain's registered environment constructor
                      for one that builds the plain environment, then
                      replaces its toolkit with `<Tools> + KBSearchMixin` and
                      its policy with the header. τ²'s evaluator only checks
                      WRITE actions and DB state, so the extra READ tool
                      changes no reward.

Decontamination: τ³ has ONE domain (banking, 97 tasks, all on the kingboard
card). Nothing from it is used here except the mechanism and the header
wording; the ten synth domains are not on any card.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from tau2.environment.toolkit import ToolKitType, ToolType, is_tool
from tau2.registry import registry

TOP_K = 5
TOKEN_RE = re.compile(r"[a-z0-9]+")
# "Greenfield Public Library — Agent Policy" -> "Greenfield Public Library"
TITLE_TAIL_RE = re.compile(r"(?:\s*[—–:-]\s*(?:agent |support |assistant |customer service )?policy|\s+policy)\s*$", re.I)


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")[:40] or "section"


@dataclass(frozen=True)
class Document:
    id: str
    title: str
    content: str


def chunk_policy(domain: str, policy: str) -> tuple[str, list[Document]]:
    """(domain title line, documents). One document per `##` section; a `###`
    sub-section is its own document titled "<section> / <sub>"; text before
    the first `##` (minus the `#` title) is the "Overview" document."""
    lines = policy.splitlines()
    title = next((ln.lstrip("# ").strip() for ln in lines if ln.startswith("# ")), domain.replace("_", " ").title())
    title = TITLE_TAIL_RE.sub("", title).strip() or domain.replace("_", " ").title()
    docs: list[Document] = []
    section, sub, buf = "Overview", None, []

    def flush() -> None:
        body = "\n".join(buf).strip()
        if body:
            name = section if sub is None else f"{section} / {sub}"
            docs.append(Document(f"doc_{domain}_{_slug(name)}_{len(docs) + 1:03d}", name, body))
        buf.clear()

    for ln in lines:
        if ln.startswith("# "):
            continue
        if ln.startswith("## "):
            flush(); section, sub = ln[3:].strip(), None
        elif ln.startswith("### "):
            flush(); sub = ln[4:].strip()
        else:
            buf.append(ln)
    flush()
    return title, docs


class BM25:
    def __init__(self, docs: list[Document], k1: float = 1.5, b: float = 0.75):
        self.docs = docs
        self.k1, self.b = k1, b
        self.tf = [Counter(TOKEN_RE.findall(f"{d.title}\n{d.content}".lower())) for d in docs]
        self.dl = [sum(c.values()) for c in self.tf]
        self.avgdl = (sum(self.dl) / len(self.dl)) if self.dl else 1.0
        df: Counter = Counter()
        for c in self.tf:
            df.update(c.keys())
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def search(self, query: str, top_k: int = TOP_K) -> list[tuple[Document, float]]:
        q = TOKEN_RE.findall(query.lower())
        scored = []
        for i, c in enumerate(self.tf):
            s = 0.0
            for t in q:
                if t not in c:
                    continue
                f = c[t]
                s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.dl[i] / self.avgdl))
            if s > 0:
                scored.append((self.docs[i], s))
        scored.sort(key=lambda x: -x[1])
        return scored[:top_k]


class KBSearchMixin(metaclass=ToolKitType):
    """τ³'s `KB_search` tool; the concrete class sets `self._kb` (a BM25)."""

    @is_tool(ToolType.READ)
    def KB_search(self, query: str) -> str:
        """Search the knowledge base for relevant documents.

        Args:
            query: The search query to find relevant documents

        Returns:
            Relevant document excerpts matching the query
        """
        results = self._kb.search(query)
        if not results:
            return "No relevant documents found."
        return "\n".join(
            f"{i}. {d.title}\n   ID: {d.id}\n   Score: {score:.4f}\n   Content: {d.content}\n"
            for i, (d, score) in enumerate(results, 1))


def kb_policy(domain_title: str) -> str:
    """τ³'s policy header (banking_knowledge/prompts/components/policy_header.md +
    the classic_rag line), re-worded for the domain."""
    return f"""# {domain_title}

You are a helpful customer service agent for {domain_title}.
Your goal is to help customers by searching the knowledge base and providing accurate information.

## Guidelines

1. Do not make up policies, information or actions that you can take on behalf of the user. All instructions will be found here or in the knowledge base. If you cannot find relevant information, let the user know.
2. Do not ask for any documentation, receipts... from the customer unless it states very clearly in the knowledge base how to process it, and whether you're allowed to do so.
3. Be polite and professional.
4. Verify the customer's identity by looking up their account before making any change, as the knowledge base describes.
5. If the issue cannot be resolved or is outside your capabilities, tell the user so; do not invent an action. Only transfer to a human agent if the knowledge base says to and the user agrees.
6. Do not give intermediate responses to users while processing that would give away internal information or policies.

**Search the knowledge base** for relevant information before you answer a question about policy or take an action, using the provided `KB_search` tool (BM25 retrieval). The knowledge base holds the complete policy of {domain_title}.
"""


_TOOL_CLASSES: dict[type, type] = {}


def kb_toolkit_class(base: type) -> type:
    cls = _TOOL_CLASSES.get(base)
    if cls is None:
        def __init__(self, db, kb):
            base.__init__(self, db)
            self._kb = kb
        cls = type(f"{base.__name__}WithKBSearch", (KBSearchMixin, base), {"__init__": __init__})
        _TOOL_CLASSES[base] = cls
    return cls


def install(domain: str) -> dict:
    """Replace the registered environment constructor of `domain` with the
    KB-backed one. Returns the KB summary (for logging). Idempotent."""
    plain = registry.get_env_constructor(domain)
    if getattr(plain, "_affine_kb", False):
        return getattr(plain, "_affine_kb_info")
    info: dict = {}

    def constructor(*args, **kwargs):
        env = plain(*args, **kwargs)
        title, docs = chunk_policy(domain, env.policy)
        kb = BM25(docs)
        env.tools = kb_toolkit_class(type(env.tools))(env.tools.db, kb)
        env.policy = kb_policy(title)
        info.update(domain=domain, title=title, documents=len(docs),
                    sections=[d.title for d in docs])
        return env

    constructor._affine_kb = True  # type: ignore[attr-defined]
    constructor._affine_kb_info = info  # type: ignore[attr-defined]
    registry._domains[domain] = constructor
    return info
