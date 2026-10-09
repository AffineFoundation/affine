"""Explicit ROOT-scoped missing queue evidence, never an audit verdict.

Known missing originals do not stop unrelated audits. Present queue records use
the unmodified admission path; new missing records remain fail-closed.
"""
import contextvars
import hashlib
import inspect
import json
from pathlib import Path
import textwrap

VERSION = 'acknowledged-missing-audit-originals-v1'


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b''):
            value.update(chunk)
    return value.hexdigest()


class Registry:
    def __init__(self, document, assessment, authority, authenticate, directory):
        value = authenticate(document, authority)
        previous = authenticate(assessment, authority)
        fields = {'version', 'created_at', 'previous_assessment_sha256', 'jobs',
                  'original_signatures_verified', 'score_evidence', 'penalty_evidence'}
        if (set(value) != fields or value['version'] != VERSION
                or value['previous_assessment_sha256'] != digest(assessment)
                or value['original_signatures_verified'] is not True
                or value['score_evidence'] is not False or value['penalty_evidence'] is not False):
            raise ValueError('exact ROOT missing-original registry')
        known = {r['identifier'] for r in previous['evidence_refusals']
                 if r.get('kind') == 'job' and r.get('reason') == 'original queue job absent'}
        if type(value['jobs']) is not dict or not known or set(value['jobs']) != known:
            raise ValueError('exact previously acknowledged missing originals')
        self.jobs = value['jobs']
        self.sha256 = digest(document)
        self.directory = Path(directory)
        if self.directory.is_symlink() or self.directory.resolve() != self.directory:
            raise ValueError('canonical original audit directory')
        for identifier, row in self.jobs.items():
            if (type(identifier) is not str or Path(identifier).name != identifier
                    or set(row) != {'job_sha256', 'original_file_sha256', 'row_sha256s'}
                    or not row['row_sha256s'] or len(row['row_sha256s']) != len(set(row['row_sha256s']))
                    or any(type(v) is not str or len(v) != 64 or any(c not in '0123456789abcdef' for c in v)
                           for v in [row['job_sha256'], row['original_file_sha256'], *row['row_sha256s']])):
                raise ValueError('exact original job and selected row digests')

    def original(self, identifier):
        row = self.jobs.get(identifier)
        if row is None:
            raise ValueError('unacknowledged missing original audit job: ' + identifier)
        # ROOT authenticated these bytes when sealing the exclusion inventory.
        # No current file/report authenticity is inferred here: the queue row is
        # absent and contributes no evidence. Restoration must authenticate the
        # real original again, through the unchanged present-row admission path.
        return row

    def defer(self, service, identifier, context):
        row = self.original(identifier)
        actual = service.state['jobs'][identifier]
        ids = actual.get('row_sha256s', [actual['row_sha256']])
        if actual['job_sha256'] != row['job_sha256'] or ids != row['row_sha256s']:
            raise ValueError('missing-original journal binding changed')
        context['missing'].append(dict(job_id=identifier, job_sha256=row['job_sha256'],
                                      status='unavailable', score_evidence=False, penalty_evidence=False))


class ScopedDeferrals(list):
    """The reviewed numerical resolver may defer only an exact absent original."""
    def __init__(self, missing):
        super().__init__()
        self.allowed = {r['job_sha256'] for r in missing}

    def append(self, issue):
        if (issue.get('kind') != 'numerical_resolution' or issue.get('status') != 'unresolved'
                or issue.get('original_job_sha256') not in self.allowed):
            raise ValueError('numerical resolution lacks acknowledged absent original')
        super().append(issue)


def install(service, registry, numerical_module, reviewed_apply):
    """Patch two reviewed CPU orchestration boundaries; no report checks change."""
    original = service.ContinuousAuditor.hourly_snapshot
    if getattr(original, '_acknowledged_missing_originals_v1', False):
        raise ValueError('missing-original overlay already installed')
    source = textwrap.dedent(inspect.getsource(original))
    lines = source.splitlines()
    missing = [i for i, line in enumerate(lines)
               if line.strip() == "if actual is None:raise ValueError('unknown original audit job')"]
    publication = [i for i, line in enumerate(lines)
                   if line.strip().startswith('document=self.controller.signed(result);atomic(target,document);')]
    if len(missing) != 1 or len(publication) != 1:
        raise ValueError('exact original hourly snapshot boundaries')
    i = publication[0]
    indent = lines[i][:-len(lines[i].lstrip())]
    lines.insert(i, indent + "result['unavailable_original_evidence']=_missing_metadata()")
    i = missing[0]
    indent = lines[i][:-len(lines[i].lstrip())]
    lines[i:i+1] = [indent + 'if actual is None:',
                   indent + ' _missing_defer(self,jobid)', indent + ' continue']
    current = contextvars.ContextVar('acknowledged_missing_audit_originals', default=None)

    def defer(instance, identifier):
        context = current.get()
        if context is None:
            raise ValueError('missing-original snapshot scope')
        registry.defer(instance, identifier, context)

    def metadata():
        context = current.get()
        return dict(version=VERSION, registry_sha256=registry.sha256,
                    unavailable_job_count=len(context['missing']), jobs=context['missing'],
                    numerical_resolutions=context['numerical'], score_evidence=False, penalty_evidence=False,
                    current_original_files_authenticated=False)

    namespace = dict(original.__globals__, _missing_defer=defer, _missing_metadata=metadata)
    exec(compile('\n'.join(lines) + '\n', '<ROOT-scoped-missing-audit-originals>', 'exec'), namespace)
    patched = namespace[original.__name__]
    original_apply = numerical_module.apply

    def apply(*args, **kwargs):
        context = current.get()
        if context is None:
            return original_apply(*args, **kwargs)
        if kwargs.pop('unavailable_execution_deferrals', None) is not None:
            raise ValueError('original numerical caller changed')
        scoped = ScopedDeferrals(context['missing'])
        result = reviewed_apply(*args, **kwargs, unavailable_execution_deferrals=scoped)
        context['numerical'].extend(scoped)
        return result

    def hourly_snapshot(self, *args, **kwargs):
        token = current.set(dict(missing=[], numerical=[]))
        try:
            return patched(self, *args, **kwargs)
        finally:
            current.reset(token)

    hourly_snapshot._acknowledged_missing_originals_v1 = True
    service.ContinuousAuditor.hourly_snapshot = hourly_snapshot
    numerical_module.apply = apply
    return dict(version=VERSION, acknowledged_jobs=len(registry.jobs),
                original_hourly_source_sha256=hashlib.sha256(source.encode()).hexdigest())
