# Native Prolog dispatch qualification

The common environment interface accepts the explicit
`prime-native-prolog-dispatch-v2` contract for the original three N-Queens
fixtures. Admission requires exact actor/session source hashes, the qualified
container runtime, original task population, and bounded turn/token settings.
An invalid native revision or source pin is rejected before provider fallback.

The portable dispatch fixture lives in
`tests/fixtures/prolog_dispatch_public.json`. Run the refusal and interface tests:

```sh
PYTHONPATH=.:tests .venv/bin/python -m unittest test_native_common_dispatch
```

Root additionally checked positive, negative, and fresh positive replays against
the original native observations and terminal grades for all three fixtures
(nine executions, two distinct board geometries). These checks establish native
dispatch and replay compatibility. They do not establish a common GPU mining,
verification, or training epoch. That requires a newly signed source package and
model qualification; existing live source packages remain immutable.

IFEval interface tests use committed public constraint tasks and bind a fresh
environment specification to the source under test. Historical specifications
and their source hashes remain unchanged.
