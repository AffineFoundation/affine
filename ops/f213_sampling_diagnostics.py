"""Default-off scalar diagnostics for the exact frozen f213 sampler.

No sampler predicates, draws, logits, contracts, return values or exceptions
are replaced. Activation is a separate prospective ROOT execution decision.
"""
import ast
import contextvars
import functools
import hashlib
import json
import logging
import os
from pathlib import Path
import stat
import time
import types

SOURCE_SHA256 = '2aeed63ee253c5b69b6036c6056a1bd582716d32edb233f0aec27deeb00fd52e'
SCIENCE_SHA256 = 'f21373d7ccb167bcd868f5ed03ce9ac7ef567d894e8ecc9e61f5d7f8645b67b8'
_STATE = contextvars.ContextVar('f213_sampling_diagnostic', default=None)
_HOOK = '_f213_sampling_diagnostic_hook'


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def _code_value(code):
    return (code.co_code.hex(), tuple(_code_value(c) if isinstance(c, types.CodeType)
            else repr(c) for c in code.co_consts), code.co_names, code.co_varnames,
            code.co_freevars, code.co_cellvars, code.co_argcount,
            code.co_posonlyargcount, code.co_kwonlyargcount, code.co_flags,
            code.co_exceptiontable.hex(), code.co_linetable.hex(), code.co_firstlineno)


def _hook(event, *args):
    record = _STATE.get()
    if record is None:
        return
    try:
        if event == 'intervals':
            n, vocab, mass, low, high, u, error = args
            supported = mass > 0
            inside = (u >= low) & (u < high) & supported
            expanded_outside = (u < low-error) | (u >= high+error)
            record.update(prefill_positions_checked=int(n), vocabulary_size=int(vocab),
                          unsupported_token_positions=int((~supported).sum().item()),
                          exact_interval_pass_positions=int(inside.sum().item()),
                          expanded_only_ambiguity_positions=int((~inside & ~expanded_outside & supported).sum().item()),
                          outside_expanded_positions=int((expanded_outside & supported).sum().item()))
            # Scalars only; the probability/CDF tensors stay in the original frame.
        elif event == 'fallback_reason':
            record['fallback_reason'] = args[0]
        elif event == 'replay_forward_started':
            record['cached_replay_forwards_started'] += 1
        elif event == 'replay_forward_completed':
            record['cached_replay_forwards_completed'] += 1
        elif event == 'replay_token':
            position, claimed, expected = args
            record['cached_replay_tokens_compared'] += 1
            if claimed == expected:
                record['cached_replay_tokens_matched'] += 1
            elif record['cached_replay_first_mismatch_position'] is None:
                record['cached_replay_first_mismatch_position'] = int(position)
    except Exception as exc:
        # Diagnostic failure cannot change the scientific verdict.
        record['counter_capture_error_type'] = type(exc).__name__


def _call(event, *args):
    return ast.Expr(ast.Call(ast.Name(_HOOK, ast.Load()),
                             [ast.Constant(event)] + list(args), []))


class _Instrument(ast.NodeTransformer):
    def visit_FunctionDef(self, node):
        if node.name not in ('verify_intervals', 'verify_sampling', 'verify_cached_reference'):
            return node
        self.function = node.name
        return self.generic_visit(node)

    def visit_Assign(self, node):
        if self.function == 'verify_cached_reference':
            text = ast.unparse(node)
            if text.startswith('result = runtime.model('):
                p = ast.Name('position', ast.Load())
                return [_call('replay_forward_started', p), node,
                        _call('replay_forward_completed', p)]
            if text.startswith('expected = pick('):
                return [node, _call('replay_token', *(ast.Name(x, ast.Load())
                                                      for x in ('position', 'claimed', 'expected')))]
        return node

    def visit_If(self, node):
        node = self.generic_visit(node)
        if self.function == 'verify_intervals' and ast.unparse(node.test) == 'bool((mass <= 0).any())':
            args = [ast.Name(x, ast.Load()) for x in ('n', 'v', 'mass', 'low', 'high', 'u', 'error')]
            return [_call('intervals', *args), node]
        return node

    def visit_ExceptHandler(self, node):
        node = self.generic_visit(node)
        if self.function == 'verify_sampling' and node.name == 'uncertainty':
            reason = ast.Attribute(ast.Call(ast.Name('type', ast.Load()),
                                            [ast.Name('uncertainty', ast.Load())], []), '__name__', ast.Load())
            node.body.insert(0, _call('fallback_reason', reason))
        return node


def install(module, sink, *, enabled=False, original_job_sha256=None):
    """Explicit prospective activation; returns an undo handle.

    Disabled means no source reads, imports or monkeypatches. Enabled verifies
    exact frozen source AND loaded function bytecode before inserting counters.
    The operator still must qualify the full177 imports/runtime separately.
    """
    if enabled is False:
        return lambda: None
    if enabled is not True or not callable(sink):
        raise ValueError('explicit diagnostic activation and receipt sink')
    if not isinstance(original_job_sha256, str) or len(original_job_sha256) != 64 or any(
            c not in '0123456789abcdef' for c in original_job_sha256):
        raise ValueError('original signed job digest')
    path = Path(module.__file__)
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SOURCE_SHA256 or _HOOK in module.__dict__:
        raise ValueError('exact frozen f213 source; no duplicate activation')
    fresh = compile(raw, str(path), 'exec', dont_inherit=True, optimize=0)
    names = ('distribution', 'verify_intervals', 'verify_sampling', 'verify_cached_reference')
    expected = {c.co_name: c for c in fresh.co_consts if isinstance(c, types.CodeType)}
    originals = {name: getattr(module, name) for name in names}
    for name, function in originals.items():
        if (not isinstance(function, types.FunctionType) or function.__globals__ is not module.__dict__
                or _code_value(function.__code__) != _code_value(expected[name])):
            raise ValueError('loaded frozen sampling code differs: ' + name)
    tree = ast.parse(raw, filename=str(path))
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names[1:]]
    instrument = _Instrument()
    transformed = ast.fix_missing_locations(ast.Module(
        body=[instrument.visit(n) for n in functions], type_ignores=[]))
    # Compile only three original function definitions; no module imports or GPU.
    namespace = dict(module.__dict__)
    namespace[_HOOK] = _hook
    exec(compile(transformed, str(path), 'exec'), namespace)
    generated = {name: namespace[name] for name in names[1:]}
    # Ensure instrumentation is exactly additive by removing hook calls and
    # comparing ASTs to the original source, including every verdict predicate.
    class Strip(ast.NodeTransformer):
        def visit_Expr(self, node):
            if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id == _HOOK:
                return None
            return self.generic_visit(node)
    stripped = Strip().visit(transformed)
    original_ast = ast.Module(body=[n for n in ast.parse(raw).body
                              if isinstance(n, ast.FunctionDef) and n.name in names[1:]], type_ignores=[])
    if ast.dump(stripped, include_attributes=False) != ast.dump(original_ast, include_attributes=False):
        raise ValueError('diagnostic transform changed scientific AST')
    # Rebuild with module globals, so original verify_sampling sees the observed
    # original interval/replay functions. No other scientific function changes.
    module.__dict__[_HOOK] = _hook
    for name, fn in generated.items():
        setattr(module, name, types.FunctionType(fn.__code__, module.__dict__, name,
                                                fn.__defaults__, fn.__closure__))
    patched_sampling = module.verify_sampling
    patched_replay = module.verify_cached_reference

    @functools.wraps(originals['verify_cached_reference'])
    def replay(*args, **kwargs):
        record = _STATE.get()
        start = time.monotonic()
        if record is not None:
            record['cached_replay_invocations'] += 1
        try:
            result = patched_replay(*args, **kwargs)
            if record is not None:
                record['cached_replay_outcome'] = 'pass'
            return result
        except BaseException as exc:
            if record is not None:
                record['cached_replay_outcome'] = type(exc).__name__
            raise
        finally:
            if record is not None:
                record['cached_replay_seconds'] += time.monotonic() - start

    @functools.wraps(originals['verify_sampling'])
    def sampling(runtime, rollout, turn_index, prompt, output, logprobs):
        record = dict(version='prospective-f213-sampling-diagnostics-v1',
                      original_job_sha256=original_job_sha256, scientific_source_sha256=SCIENCE_SHA256,
                      sampler_source_sha256=SOURCE_SHA256,
                      sampler_loaded_code_sha256=digest(_code_value(originals['verify_sampling'].__code__)),
                      turn_index=turn_index,
                      prefill_positions_checked=0, unsupported_token_positions=0,
                      exact_interval_pass_positions=0, expanded_only_ambiguity_positions=0,
                      outside_expanded_positions=0, fallback_reason=None,
                      cached_replay_invocations=0, cached_replay_forwards_started=0,
                      cached_replay_forwards_completed=0, cached_replay_tokens_compared=0,
                      cached_replay_tokens_matched=0, cached_replay_first_mismatch_position=None,
                      cached_replay_outcome='not_invoked', cached_replay_seconds=0.,
                      sampling_outcome='not_completed')
        try:
            record.update(task_hash=rollout['task_hash'], task_index=rollout['index'],
                          attempt=rollout['seed'], rollout_sha256=digest(rollout),
                          prompt_sha256=digest(prompt),
                          output_sha256=digest(output), output_tokens=len(output),
                          sampling_context_sha256=digest(runtime.sampling_context),
                          calibration_sha256=digest(runtime.fast_sampling_calibration),
                          cdf_abs_error=runtime.fast_sampling_calibration['cdf_abs_error'],
                          temperature=runtime.harness['temperature'], top_p=runtime.harness['top_p'],
                          output_cap=runtime.harness['max_output_tokens'],
                          runtime_revision=getattr(runtime, 'runtime_revision', None))
        except Exception as exc:
            record['binding_capture_error_type'] = type(exc).__name__
        token = _STATE.set(record)
        start = time.monotonic()
        try:
            result = patched_sampling(runtime, rollout, turn_index, prompt, output, logprobs)
            record['sampling_outcome'] = 'pass'
            return result
        except BaseException as exc:
            record['sampling_outcome'] = type(exc).__name__
            raise
        finally:
            record['elapsed_seconds'] = time.monotonic() - start
            _STATE.reset(token)
            try:
                sink(record)
            except Exception as exc:
                logging.getLogger(__name__).warning('Sampling diagnostic receipt unavailable (%s)', type(exc).__name__)

    module.verify_cached_reference = replay
    module.verify_sampling = sampling

    def undo():
        for name, function in originals.items():
            setattr(module, name, function)
        module.__dict__.pop(_HOOK, None)
    return undo


class PrivateReceiptSink:
    """Independent immutable scalar files in an already owned0700 namespace."""
    def __init__(self, directory):
        self.directory = Path(directory)
        info = self.directory.lstat()
        if (not stat.S_ISDIR(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o700
                or info.st_uid != os.getuid() or self.directory.resolve() != self.directory):
            raise ValueError('private owned diagnostic namespace')
        self.identity = (info.st_dev, info.st_ino)

    def __call__(self, record):
        raw = canonical(record)
        if len(raw) > 32768:
            raise ValueError('bounded scalar diagnostic receipt')
        path = self.directory / ('sampling-' + str(time.time_ns()) + '-' + digest(record) + '.json')
        parent = os.open(self.directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            info = os.fstat(parent)
            if ((info.st_dev, info.st_ino) != self.identity or info.st_uid != os.getuid()
                    or stat.S_IMODE(info.st_mode) != 0o700):
                raise ValueError('diagnostic namespace changed')
            fd = os.open(path.name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                         0o600, dir_fd=parent)
            with os.fdopen(fd, 'wb') as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
            os.fsync(parent)
        finally:
            os.close(parent)
