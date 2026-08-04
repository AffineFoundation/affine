"""Published prior bank for the Λ2_bank positivity gate.

A miner's thoughts must beat every prior in this bank (including empty) on
the teacher's action for a pair to count as bank-positive. Generic
paraphrase / SWE-prior attacks score ≤0 by construction (RT-1b, RT-2c).

Public by design: the bank is part of the chain contract, so miners can test
against it. Changing it is a contract change and must be announced.
"""

PRIOR_BANK: dict[str, str] = {
    "empty": "",
    "ls": (
        "I should explore the repository structure first. Standard SWE workflow: "
        "list the top-level files, then inspect the package directory."
    ),
    "cat": (
        "I need to read the relevant source file mentioned in the issue to "
        "understand the current implementation before making changes."
    ),
    "grep": (
        "I should search the codebase for the symbol or class mentioned in the "
        "PR description to find where the bug lives."
    ),
    "find": (
        "Let me locate the relevant files in the repository tree before reading "
        "any one of them in detail."
    ),
    "test": (
        "I should run a small reproduction script or the existing tests to "
        "confirm the failure mode before editing."
    ),
    "para_ls": (
        "Next I should list the files in the working directory. That is the "
        "natural next step given the current repository state."
    ),
}
