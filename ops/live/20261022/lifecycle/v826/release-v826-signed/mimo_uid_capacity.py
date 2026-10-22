"""Exact host-only roster expansion; frozen scientific source remains unchanged."""
import ast
import hashlib
import inspect
import textwrap

CAPACITY = 1000
SITES = {
    'Lifecycle._plan': ('len(rows) <= 256', "0 <= row['uid'] < 256"),
    'Coordinator.open': ('len(miners) <= 256',),
}


def method_sha(method):
    return hashlib.sha256(inspect.getsource(method).encode()).hexdigest()


def expanded_method(method, name, expected_sha256):
    if name not in SITES or method.__qualname__ != name or method_sha(method) != expected_sha256:
        raise ValueError('exact original host method required')
    source = textwrap.dedent(inspect.getsource(method))
    tree = ast.parse(source)
    expected = {ast.dump(ast.parse(site, mode='eval').body): site for site in SITES[name]}
    found = []
    class Expand(ast.NodeTransformer):
        def visit_Compare(self, node):
            key = ast.dump(node)
            if key in expected:
                constants = [value for value in node.comparators if isinstance(value, ast.Constant)
                             and type(value.value) is int and value.value == 256]
                if len(constants) != 1:
                    raise ValueError('exact original capacity comparison')
                constants[0].value = CAPACITY
                found.append(key)
                return node
            return self.generic_visit(node)
    changed = Expand().visit(tree)
    if sorted(found) != sorted(expected):
        raise ValueError('exact original capacity sites required')
    ast.fix_missing_locations(changed)
    namespace = dict(method.__globals__)
    exec(compile(changed, inspect.getsourcefile(method), 'exec'), namespace)
    result = namespace[method.__name__]
    result.__defaults__ = method.__defaults__
    result.__kwdefaults__ = method.__kwdefaults__
    result.__annotations__ = dict(method.__annotations__)
    result.__qualname__ = method.__qualname__
    result.__module__ = method.__module__
    return result


def replacements(Lifecycle, Coordinator, pins):
    if set(pins) != set(SITES):
        raise ValueError('exact two original host method pins')
    # Prepare both before installation, so a failed pin cannot partially install.
    return (expanded_method(Lifecycle._plan, 'Lifecycle._plan', pins['Lifecycle._plan']),
            expanded_method(Coordinator.open, 'Coordinator.open', pins['Coordinator.open']))


# ---------------------------------------------------------------------------
# v8.2.4 host overrides (ROOT-admitted through this file's pinned digest in the
# uid-operator config; the frozen scientific source / worker closure is unchanged).
#   1. Arbos accepts v2 model-publication receipts (streamed sha256 + per-part
#      Content-MD5 + HEAD length; the trainer no longer full-GETs). v1 unchanged.
#      Arbos' own full-object SHA-256 readback stays mandatory before authorization.
#   2. Finalize clock fix: a tick that fetched TRAINING-RESULT after the trainer
#      reported must not reject it as completed_at > now+5 (re-read the clock).
#   3. max_tasks 128 <-> 256 only at an epoch BOUNDARY (CURRENT re-stamped to the configured value
#      only when the current epoch has no PLAN yet; an in-flight epoch keeps its planned size).
# ---------------------------------------------------------------------------
V824_PINS = {
    'mimo_training_worker.validate_result': '81823040aa55f198cfd456763d40ea806f73ae8cce92958105d0f2fc8394c1e4',
    'Lifecycle._training': '93a538d13ee04cabc0b9137ccea53dd07087e38f8442b4ab12bb7dd8308634c7',
    'Lifecycle._current': '311d719cfde2ae8ed5cb07dcd6a8fa7cb4e4d2d1896ade1cd8e1336e556d319a',
}
V824_MAX_TASKS = (128, 256)
V824_UPLOAD_VERIFICATION = 'streamed-sha256+part-content-md5+head-length'
V824_PUBLICATION_VERIFIED = '''
VERSION_V2 = 'affine-mimo-model-publication-v2'
UPLOAD_VERIFICATION_V2 = %r


def publication_verified(value):
    """v1: trainer full GET; v2: streamed sha256 + part MD5 + HEAD (Arbos does the full GET)."""
    if value.get('version') == VERSION:
        return value.get('model_full_get_verified') is True
    return (value.get('version') == VERSION_V2 and type(value.get('model_full_get_verified')) is bool and
            value.get('model_upload_verification') == UPLOAD_VERIFICATION_V2)
''' % V824_UPLOAD_VERIFICATION


def _replace_exact(source, old, new):
    if source.count(old) != 1:
        raise ValueError('exact original v8.2.4 override site: ' + old[:60])
    return source.replace(old, new)


def v824_validate_result_source(original):
    source = textwrap.dedent(original)
    source = _replace_exact(source, "    from .mimo_grpo.publication import VERSION as PUBLICATION_VERSION\n",
                            "    from .mimo_grpo.publication import publication_verified\n")
    source = _replace_exact(source, "need(published.get('version') == PUBLICATION_VERSION and",
                            "need(publication_verified(published) and")
    source = _replace_exact(source, "         published.get('model_full_get_verified') is True and\n", "")
    return source


def v824_training_source(original):
    source = textwrap.dedent(original)
    return _replace_exact(source,
        "        if result_envelope is None:return None\n",
        "        if result_envelope is None:return None\n"
        "        now=max(now,int(__import__('time').time()))  # v8.2.4 finalize clock fix\n")


def _method_from(method, source):
    namespace = dict(method.__globals__)
    exec(compile(source, inspect.getsourcefile(method), 'exec'), namespace)
    result = namespace[method.__name__]
    result.__defaults__, result.__kwdefaults__ = method.__defaults__, method.__kwdefaults__
    result.__qualname__, result.__module__ = method.__qualname__, method.__module__
    return result


def v824_install(Lifecycle):
    """Prepare everything, check every pin, then install (no partial install)."""
    import json as _json
    import subnet.mimo_training_worker as worker
    import subnet.mimo_grpo.publication as publication
    from subnet.controller import save_manifest
    if (method_sha(worker.validate_result) != V824_PINS['mimo_training_worker.validate_result'] or
            method_sha(Lifecycle._training) != V824_PINS['Lifecycle._training'] or
            method_sha(Lifecycle._current) != V824_PINS['Lifecycle._current']):
        raise ValueError('exact original v8.2.4 host override methods required')
    validate_source = v824_validate_result_source(inspect.getsource(worker.validate_result))
    training = _method_from(Lifecycle._training, v824_training_source(inspect.getsource(Lifecycle._training)))
    original_current = Lifecycle._current

    def _current(self):
        path = self.root / 'CURRENT.json'
        target = self.__dict__.setdefault('_v824_target_max_tasks', self.max_tasks)
        if target in V824_MAX_TASKS and path.is_file() and not path.is_symlink():
            current = _json.loads(path.read_bytes())
            stamped = current.get('max_tasks')
            if stamped in V824_MAX_TASKS and stamped != target:
                planned = (self.root / 'epochs' / str(current['epoch_index']) / 'PLAN.json').exists()
                if planned:
                    self.max_tasks = stamped                       # in-flight epoch keeps its planned size
                else:
                    save_manifest(path, dict(current, max_tasks=target))  # epoch-boundary re-stamp (either direction)
                    self.max_tasks = target
            else:
                self.max_tasks = target
        return original_current(self)
    _current.__qualname__, _current.__module__ = original_current.__qualname__, original_current.__module__
    # install
    exec(compile(V824_PUBLICATION_VERIFIED, inspect.getsourcefile(publication), 'exec'), publication.__dict__)
    exec(compile(validate_source, inspect.getsourcefile(worker), 'exec'), worker.__dict__)
    Lifecycle._training = training
    Lifecycle._current = _current
    return dict(v824=True, max_tasks=list(V824_MAX_TASKS))
