"""wvk 18 staging — code side of the prose `text` fallback at tool_call turns.

Knob `[duel].text_fallback_at_tool_turns` (default False = pre-wvk-18
behaviour, so wvk <= 17 verdicts replay unchanged). When on: a rollout at a
`tool_call`-kind turn that CLOSED </think>, contains no parseable tool call
but a non-empty visible reply, splits as (latent reasoning, visible reply)
— a `text` action — instead of ("", "") (a dropped reference / a miner
forfeit). An empty visible reply is still a forfeit; an unclosed think
block never reaches the fallback (require_think_close on the miner side;
on the teacher side the fallback itself requires the closed tag, so a
reasoning-only reference cannot be counted as prose — the wvk-13 hole).

Edits (anchored, idempotent):
  evalsrv/chat.py        split_rollout(..., text_fallback_at_tool_turns)
  evalsrv/vllm_client.py VllmModel / ModelPool: knob + n_text_fallback counter
  evalsrv/dueling.py     pools get the knob; per-side + teacher
                         n_text_fallback telemetry; duel_params stamp
  affine/config.py       DuelCfg.text_fallback_at_tool_turns
"""

from pathlib import Path

REPO = Path("/home/const/subnet120")


def sub(path: Path, old: str, new: str, marker: str) -> None:
    s = path.read_text()
    if marker in s:
        print(f"{path.name}: already has {marker!r}")
        return
    if s.count(old) != 1:
        raise SystemExit(f"{path.name}: anchor not unique/missing: {old[:70]!r}")
    path.write_text(s.replace(old, new))
    print(f"{path.name}: patched ({marker})")


ch = REPO / "affine/evalsrv/chat.py"
sub(ch, '''def split_rollout(text: str, action_kind: str | None = dialects.DEFAULT_KIND,
                  require_think_close: bool = False) -> tuple[str, str]:
''', '''TEXT_FALLBACK_KINDS = ("tool_call",)


def split_rollout(text: str, action_kind: str | None = dialects.DEFAULT_KIND,
                  require_think_close: bool = False,
                  text_fallback_at_tool_turns: bool = False) -> tuple[str, str]:
''', "text_fallback_at_tool_turns: bool = False) -> tuple[str, str]:")
sub(ch, '''    if THINK_CLOSE in text:
        latent, _, rest = text.partition(THINK_CLOSE)
    elif require_think_close:
        return "", ""
    else:
        latent, rest = "", text
    before, y = dialects.split_action(rest, action_kind)
    if not y:
        return "", ""
''', '''    closed = THINK_CLOSE in text
    if closed:
        latent, _, rest = text.partition(THINK_CLOSE)
    elif require_think_close:
        return "", ""
    else:
        latent, rest = "", text
    before, y = dialects.split_action(rest, action_kind)
    if not y and text_fallback_at_tool_turns and closed \\
            and (action_kind or dialects.DEFAULT_KIND) in TEXT_FALLBACK_KINDS:
        # wvk 18 (2026-09-15): at a tool-call turn a reply that closed its
        # reasoning and says something visible but calls no tool is a
        # prose (`text`) action — the whole visible reply — the same rule
        # the fold's teacher probe applies. Only with </think> closed: an
        # unclosed block has no visible reply and stays a forfeit / drop.
        before, y = dialects.split_action(rest, "text")
    if not y:
        return "", ""
''', "TEXT_FALLBACK_KINDS:")

vc = REPO / "affine/evalsrv/vllm_client.py"
sub(vc, '''    def __init__(self, cfg: Served, client: httpx.AsyncClient, sem: asyncio.Semaphore,
                 require_think_close: bool = False):
''', '''    def __init__(self, cfg: Served, client: httpx.AsyncClient, sem: asyncio.Semaphore,
                 require_think_close: bool = False,
                 text_fallback_at_tool_turns: bool = False):
''', "text_fallback_at_tool_turns: bool = False):")
sub(vc, '''        self.n_samples = 0
        self.n_think_closed = 0

    async def _post(self, payload: dict) -> dict:
''', '''        self.n_samples = 0
        self.n_think_closed = 0
        # [duel].text_fallback_at_tool_turns (wvk 18): a closed-think prose
        # reply at a tool_call turn is a `text` action. Counted per sample.
        self.text_fallback_at_tool_turns = text_fallback_at_tool_turns
        self.n_text_fallback = 0

    async def _post(self, payload: dict) -> dict:
''', "self.n_text_fallback = 0")
sub(vc, '''        text = d["choices"][0]["text"]
        self.n_samples += 1
        self.n_think_closed += int(think_closed(text))
        return split_rollout(text, action_kind,
                             require_think_close=self.require_think_close)
''', '''        text = d["choices"][0]["text"]
        self.n_samples += 1
        self.n_think_closed += int(think_closed(text))
        z, y = split_rollout(text, action_kind,
                             require_think_close=self.require_think_close,
                             text_fallback_at_tool_turns=self.text_fallback_at_tool_turns)
        if (y and self.text_fallback_at_tool_turns
                and (action_kind or dialects.DEFAULT_KIND) in TEXT_FALLBACK_KINDS
                and dialects.count_actions(y, action_kind) == 0):
            self.n_text_fallback += 1
        return z, y
''', "self.n_text_fallback += 1")
sub(vc, '''    @property
    def n_think_closed(self) -> int:
        return sum(r.n_think_closed for r in self.replicas)
''', '''    @property
    def n_think_closed(self) -> int:
        return sum(r.n_think_closed for r in self.replicas)

    @property
    def n_text_fallback(self) -> int:
        """Samples split as a prose `text` action at a tool_call turn (wvk 18)."""
        return sum(r.n_text_fallback for r in self.replicas)
''', "def n_text_fallback(self)")
sub(vc, """from .chat import (
    chat_prompt,
    extract_action,
    force_text,
    gen_prompt,
    get_tokenizer,
    inject_prompt,
    split_rollout,
    think_closed,
    thought_text,
)
""", """from affine import dialects

from .chat import (
    TEXT_FALLBACK_KINDS,
    chat_prompt,
    extract_action,
    force_text,
    gen_prompt,
    get_tokenizer,
    inject_prompt,
    split_rollout,
    think_closed,
    thought_text,
)
""", "    TEXT_FALLBACK_KINDS,\n")

du = REPO / "affine/evalsrv/dueling.py"
sub(du, '''        def _pool(served: Served | list[Served],
                  require_close: bool = False) -> ModelPool:
            items = served if isinstance(served, list) else [served]
            return ModelPool([
                VllmModel(s, http, asyncio.Semaphore(conc),
                          require_think_close=require_close) for s in items
            ])
''', '''        text_fallback = bool(duel_cfg.get("text_fallback_at_tool_turns", False))

        def _pool(served: Served | list[Served],
                  require_close: bool = False) -> ModelPool:
            items = served if isinstance(served, list) else [served]
            return ModelPool([
                VllmModel(s, http, asyncio.Semaphore(conc),
                          require_think_close=require_close,
                          text_fallback_at_tool_turns=text_fallback) for s in items
            ])
''', "text_fallback_at_tool_turns=text_fallback")
sub(du, '''        summary["n_samples"] = pool.n_samples
        summary["think_close_rate"] = pool.think_close_rate
''', '''        summary["n_samples"] = pool.n_samples
        summary["think_close_rate"] = pool.think_close_rate
        # wvk 18: samples that became a prose `text` action at a tool_call
        # turn (0 when the knob is off — telemetry either way).
        summary["n_text_fallback"] = pool.n_text_fallback
''', 'summary["n_text_fallback"]')
sub(du, '''            "max_thought_tokens": int(duel_cfg["max_thought_tokens"]),
            "max_action_tokens": int(duel_cfg["max_action_tokens"]),
            # Teacher-only reference budget (wvk 17); None = shared cap.
''', '''            "max_thought_tokens": int(duel_cfg["max_thought_tokens"]),
            "max_action_tokens": int(duel_cfg["max_action_tokens"]),
            # wvk 18: prose reply at a tool_call turn = `text` action.
            "text_fallback_at_tool_turns": text_fallback,
            # Teacher-only reference budget (wvk 17); None = shared cap.
''', '"text_fallback_at_tool_turns": text_fallback,')

cfg = REPO / "affine/affine/config.py"
sub(cfg, '''    ref_max_tokens: int | None = None
''', '''    ref_max_tokens: int | None = None
    # wvk 18 (2026-09-15): at a tool_call turn a closed-think reply with no
    # tool call but a non-empty visible reply is a `text` action (teacher
    # reference and miner alike). False = pre-wvk-18 (dropped / forfeit).
    text_fallback_at_tool_turns: bool = False
''', "text_fallback_at_tool_turns: bool = False")
sub(cfg, '''        ref_max_tokens=_ref_max_tokens(d),
''', '''        ref_max_tokens=_ref_max_tokens(d),
        text_fallback_at_tool_turns=bool(d.get("text_fallback_at_tool_turns", False)),
''', 'text_fallback_at_tool_turns=bool(d.get(')
print("done")
