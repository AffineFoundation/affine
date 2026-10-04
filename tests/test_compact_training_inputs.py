"""Prospective CPU controls: synthetic evidence is never live admission."""
import copy
import hashlib
import json
import sqlite3
from unittest.mock import patch

import unittest
import tempfile
from pathlib import Path
from nacl.signing import SigningKey

from subnet import compact_training_inputs as compact
from subnet import training_receipts as v1
from subnet.storage import canonical
from training_receipt_fixtures import transport_fixture, signed_receipt, sign


def parameterize(names, values):
    def decorate(function):
        function.cases = [(tuple(names.split(',')), value if isinstance(value, tuple) else (value,))
                          for value in values]
        return function
    return decorate


def raises(kind, match=None):
    case = unittest.TestCase()
    return case.assertRaises(kind) if match is None else case.assertRaisesRegex(kind, match)


def make_setup(tmp_path):
    key = SigningKey.generate(); fx = transport_fixture(key)
    authority = key.verify_key.encode().hex()
    fx['manifest']['training_input_policy']=compact.VERSION
    fx['receipt'],fx['audit'],fx['verify_job'],fx['worker_request']=signed_receipt(
        key,fx['manifest'],fx['miner'],fx['frozen'],fx['batch'])
    fx['submission']['verifier_receipt']=fx['receipt']
    request = fx['worker_request']; job = fx['verify_job']
    row = dict(id=job['payload']['job_id'],status='complete',role='verify',
        envelope=json.dumps(job),digest=v1.sha(job['payload']),report=json.dumps(request['payload']['report']),
        report_digest=v1.sha(request['payload']['report']),report_request=json.dumps(request),worker=request['signer'])
    manifest = dict(fx['manifest'],training_input_policy=compact.VERSION)
    def build(row_override=None, receipt_override=None, audit_override=None):
        return compact.prepare_from_completed_row(row if row_override is None else row_override,
            authority,{request['signer']:['verify']},manifest,fx['miner'],fx['frozen'],
            fx['audit'] if audit_override is None else audit_override,
            fx['receipt'] if receipt_override is None else receipt_override)
    data, payload = build()
    obj = dict(sha256=payload['compact_sha256'],size=payload['compact_size'],
        verifier_receipt=sign(key,payload),accepted_batch_sha256=[v1.sha(fx['batch'])])
    path = tmp_path/'compact.json';path.write_bytes(data)
    return dict(fx=fx,key=key,authority=authority,row=row,manifest=manifest,build=build,
                data=data,payload=payload,obj=obj,path=path)


def admit(s):
    return compact.admitted_submission(s['path'],s['obj'],s['manifest'],s['authority'])


def resign_artifact(s, artifact=None, data=None):
    data = canonical(artifact) if data is None else data
    payload = dict(s['payload'],compact_sha256=hashlib.sha256(data).hexdigest(),compact_size=len(data))
    s['obj'].update(sha256=payload['compact_sha256'],size=len(data),verifier_receipt=sign(s['key'],payload))
    s['path'].write_bytes(data)


def test_same_pairs_without_opening_original_zip_or_running_verification(setup):
    s=setup
    with patch('subnet.batches.submission_records',side_effect=AssertionError('ZIP opened')), \
         patch('subnet.forced_sampling.require_report',side_effect=AssertionError('verification')):
        summary,pairs=admit(s)
    assert summary['submission_sha256']==s['fx']['frozen']['sha256']
    assert summary['submission_size']==s['fx']['frozen']['size']
    assert summary['original_report_sha256']==s['fx']['receipt']['payload']['original_report_sha256']
    assert summary['trainer_verification_performed'] is False
    assert pairs[0][1:]==tuple(s['fx']['batch']['rollouts'])
    assert summary['accepted']==[s['fx']['batch']]
    assert json.loads(s['data'])['documents'][0]['batch']==s['fx']['batch']
    assert 'arrays' not in json.loads(s['data'])


@parameterize('field,value', [('status','leased'),('role','train'),('digest','0'*64),
    ('report_digest','0'*64),('worker','0'*64),('id','different-original')])
def test_complete_queue_lineage_is_required(setup,field,value):
    row=dict(setup['row']);row[field]=value
    with raises(ValueError):setup['build'](row_override=row)


def test_signed_worker_report_alone_cannot_replace_complete_row(setup):
    row=dict(setup['row']);request=json.loads(row['report_request'])
    request['payload']['report']['audits'][0]['accepted'][0]['rollouts'][0]['turns'][0]['output']=[999]
    row['report_request']=json.dumps(request)
    with raises(ValueError):setup['build'](row_override=row)


def test_local_audit_cannot_supply_different_tokens(setup):
    audit=copy.deepcopy(setup['fx']['audit']);audit['accepted'][0]['rollouts'][0]['turns'][0]['output']=[999]
    with raises(ValueError):setup['build'](audit_override=audit)


def test_receipt_cannot_claim_a_different_original_report(setup):
    receipt=copy.deepcopy(setup['fx']['receipt']);receipt['payload']['original_report_sha256']='0'*64
    receipt=sign(setup['key'],receipt['payload'])
    with raises(ValueError,match='differs'):setup['build'](receipt_override=receipt)


@parameterize('mutation', ['tokens','reward','sampling','epoch','checkpoint','task','slot','extra','drop'])
def test_even_signed_compact_transport_cannot_change_original_documents(setup,mutation):
    s=setup;artifact=json.loads(s['data']);document=artifact['documents'][0];batch=document['batch']
    if mutation=='tokens':batch['rollouts'][0]['turns'][0]['output']=[99]
    elif mutation=='reward':batch['rollouts'][0]['reward']=0
    elif mutation=='sampling':batch['rollouts'][0]['sampling']={}
    elif mutation=='epoch':artifact['epoch']='other'
    elif mutation=='checkpoint':artifact['checkpoint']='0'*64
    elif mutation=='task':batch['index']=1
    elif mutation=='slot':document['batch_number']=1
    elif mutation=='extra':artifact['documents'].append(copy.deepcopy(document))
    elif mutation=='drop':artifact['documents']=[]
    resign_artifact(s,artifact)
    with raises(ValueError):admit(s)


def test_full_rollout_hash_is_checked_independently(setup):
    s=setup;inner=copy.deepcopy(s['payload']['original_verifier_receipt']['payload'])
    inner['fully_audited_batches'][0]['positive_rollout_sha256']=['0'*64]
    receipt=sign(s['key'],inner)
    artifact=json.loads(s['data']);artifact['original_verifier_receipt_sha256']=v1.sha(receipt)
    s['payload']['original_verifier_receipt']=receipt
    resign_artifact(s,artifact)
    with raises(ValueError,match='rollout hashes'):admit(s)


@parameterize('field', ['checkpoint','source_bundle','epoch','training_policy'])
def test_manifest_context_substitution(setup,field):
    s=setup
    if field=='checkpoint':s['manifest']['checkpoint']=dict(s['manifest']['checkpoint'],id='0'*64)
    elif field=='source_bundle':s['manifest']['source_bundle']={'sha256':'0'*64}
    else:s['manifest'][field]='other'
    with raises(ValueError):admit(s)


def test_nested_receipt_signature_cannot_be_forged(setup):
    s=setup;s['payload']['original_verifier_receipt']['signature']='AAAA'
    s['obj']['verifier_receipt']=sign(s['key'],s['payload'])
    with raises(ValueError):admit(s)


def test_compact_receipt_signature_cannot_be_forged(setup):
    setup['obj']['verifier_receipt']['signature']='AAAA'
    with raises(ValueError):admit(setup)


@parameterize('data_kind', ['append','truncated','duplicate-key','noncanonical','nonfinite','deep'])
def test_framing_and_canonical_bytes(setup,data_kind):
    s=setup
    if data_kind=='append':s['path'].write_bytes(s['data']+b' ')
    elif data_kind=='truncated':s['path'].write_bytes(s['data'][:-1])
    elif data_kind=='duplicate-key':resign_artifact(s,data=b'{"version":"a","version":"b"}')
    elif data_kind=='noncanonical':resign_artifact(s,data=json.dumps(json.loads(s['data']),indent=2).encode())
    elif data_kind=='nonfinite':resign_artifact(s,data=b'{"bad":NaN}')
    elif data_kind=='deep':resign_artifact(s,data=b'['*2000+b']'*2000)
    with raises(ValueError):admit(s)


def test_oversized_signed_artifact_refused_before_read(setup):
    s=setup;payload=dict(s['payload'],compact_size=compact.MAX_BYTES+1)
    s['obj'].update(size=payload['compact_size'],verifier_receipt=sign(s['key'],payload))
    with patch.object(type(s['path']),'open',side_effect=AssertionError('read')):
        with raises(ValueError,match='budget'):admit(s)


def test_symlink_refused(setup,tmp_path):
    s=setup;link=tmp_path/'alias';link.symlink_to(s['path']);s['path']=link
    with raises(ValueError,match='symlink'):admit(s)


def test_unsigned_or_old_policy_cannot_select_new_transport(setup):
    setup['manifest']['training_input_policy']=v1.VERSION
    with raises(ValueError,match='prospective'):admit(setup)


def test_old_v1_zip_still_works_and_does_not_accept_v2(setup,tmp_path):
    s=setup;fx=s['fx'];path=tmp_path/'original.zip';path.write_bytes(fx['data'])
    summary,pairs=v1.admitted_submission(path,fx['submission'],fx['manifest'],s['authority'])
    assert summary['version']==v1.VERSION
    assert pairs[0][1:]==tuple(fx['batch']['rollouts'])
    with raises(ValueError):v1.admitted_submission(s['path'],s['obj'],fx['manifest'],s['authority'])


def test_read_only_authoritative_queue_adapter(setup,tmp_path):
    s=setup;path=tmp_path/'jobs.sqlite'
    with sqlite3.connect(path) as database:
        database.execute('CREATE TABLE jobs ('+','.join(k+' TEXT' for k in s['row'])+')')
        database.execute('INSERT INTO jobs VALUES ('+','.join('?' for _ in s['row'])+')',tuple(s['row'].values()))
    audit=dict(s['fx']['audit'],remote_job_id=s['row']['id'])
    data,payload=compact.prepare_from_queue(path,s['authority'],{s['row']['worker']:['verify']},
        s['manifest'],s['fx']['miner'],s['fx']['frozen'],audit,s['fx']['receipt'])
    assert data==s['data'] and payload==s['payload']
    audit['remote_job_id']='another-job'
    with raises(ValueError):compact.prepare_from_queue(path,s['authority'],{},
        s['manifest'],s['fx']['miner'],s['fx']['frozen'],audit,s['fx']['receipt'])


def prospective_job(s):
    return dict(role='train',training_policy=s['manifest']['training_policy'],
        training_input_policy=compact.VERSION,submissions=[s['obj']],
        source_files={'subnet/compact_training_inputs.py':'a'*64,'subnet/training_receipts.py':'b'*64})


def test_prospective_job_and_truthful_report(setup):
    s=setup;job=prospective_job(s);compact.validate_job(job,s['manifest'],s['authority'])
    summary,_=admit(s)
    report=dict(training_admissions=[summary],audits=[],training=dict(training_input_policy=compact.VERSION,
        trainer_verification_performed=False,all_pairs_authenticated_verifier_receipts=True))
    compact.validate_report(report,job,s['manifest'],s['authority'])
    for field in ('submission_sha256','original_report_sha256','compact_sha256','original_verifier_receipt_sha256'):
        bad=copy.deepcopy(report);bad['training_admissions'][0][field]='0'*64
        with raises(ValueError):compact.validate_report(bad,job,s['manifest'],s['authority'])
    bad=copy.deepcopy(report);bad['training_admissions'][0]['accepted'][0]['rollouts'][0]['turns'][0]['output']=[99]
    with raises(ValueError):compact.validate_report(bad,job,s['manifest'],s['authority'])
    for field,value in [('training_input_policy',v1.VERSION),('trainer_verification_performed',True),
        ('all_pairs_authenticated_verifier_receipts',False),('all_pairs_independently_reaudited',True)]:
        bad=copy.deepcopy(report);bad['training'][field]=value
        with raises(ValueError):compact.validate_report(bad,job,s['manifest'],s['authority'])
    report['audits']=[{'new':'audit'}]
    with raises(ValueError):compact.validate_report(report,job,s['manifest'],s['authority'])


def test_duplicate_job_and_old_amendment_refused(setup):
    s=setup;job=prospective_job(s);job['submissions']*=2
    with raises(ValueError):compact.validate_job(job,s['manifest'],s['authority'])
    job=prospective_job(s);s['manifest']['training_execution_amendment']={'old':'v1'}
    with raises(ValueError):compact.validate_job(job,s['manifest'],s['authority'])


def test_missing_source_pin_cannot_admit_compact_job(setup):
    s=setup;job=prospective_job(s);del job['source_files']['subnet/compact_training_inputs.py']
    with raises(ValueError):compact.validate_job(job,s['manifest'],s['authority'])


class CompactTrainingInputTests(unittest.TestCase):
    """Each parameter gets a fresh authenticated synthetic evidence chain."""


def _install_test(function, names=(), values=()):
    def run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            arguments = dict(setup=make_setup(root), **dict(zip(names, values)))
            if 'tmp_path' in function.__code__.co_varnames[:function.__code__.co_argcount]:
                arguments['tmp_path'] = root
            function(**arguments)
    return run


for _name, _function in list(globals().items()):
    if _name.startswith('test_') and callable(_function):
        for _number, (_names, _values) in enumerate(getattr(_function, 'cases', [((), ())])):
            setattr(CompactTrainingInputTests, _name + '_' + str(_number), _install_test(_function, _names, _values))
