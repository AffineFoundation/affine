"""Signed fixture composition with real compact queue evidence and EMA math.

The original writer host is represented by an explicit spy, never a live wallet.
New source imports use the actual isolated package path, not a fake adapter.
"""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType,SimpleNamespace
import unittest
from unittest.mock import patch

import test_mimo_audit_bridge as audit_fixture
from test_mimo_worker import sign
from subnet.storage import canonical,Identity
from subnet.continuous_audit_policy import snapshot,observations
from subnet.current_assessment import calculate,fallback,recipients
from subnet.mimo_audit_scheduler import AuditScheduler
from subnet.mimo_assessment_bridge import VERSION,TRANSITION
from subnet.distributed_roles import Coordinator

PATH=Path(__file__).resolve().parents[1]/'ops/mimo_hourly_writer.py'
spec=importlib.util.spec_from_file_location('tested_mimo_hourly_writer',PATH)
entry=importlib.util.module_from_spec(spec);spec.loader.exec_module(entry)


class HourlyCompositionTests(unittest.TestCase):
    def setUp(self):
        f=audit_fixture.AuditBridgeTests();f.setUp();self.addCleanup(f.doCleanups);self.f=f;self.calls=[]
        actual=f.f;self.root=actual.root;self.authority=actual.authority
        row=f.queue_evidence();queue=Coordinator(self.root/'queue-evidence.db',self.authority,f.verifiers,clock=lambda:f.at+1)
        directory=self.root/'scheduler'
        scheduler=AuditScheduler(directory,actual.bucket,queue,Identity(actual.operator.encode()),
            manifest_guard=f.guard,metadata=lambda m:f.execution[m['source_bundle']['sha256']])
        scheduler.admit(f.signed_population());state=scheduler._load();record=f.population()['records'][0]
        from subnet.mimo_audit_bridge import digest
        state['draws'][digest(record)]=dict(row=record,job_id=row['id'],expires_at=f.at+100,generation=0)
        scheduler._save(state);self.scheduler=scheduler
        reader=sign(actual.operator,dict(version=VERSION,directory=str(directory),queue_path=queue.path,
            release_policy=f.release.envelope,approved_execution=f.execution,audit_policy=f.policy,
            verifiers=list(f.verifiers),transition_policy=TRANSITION))
        self.cutoff=int((f.at+3600)//3600)*3600
        self.legacy=dict(snapshots=[],committed_at_by_epoch={},evidence_hashes={'legacy_admission':'a'*64})
        self.pdoc=sign(actual.operator,dict(verifiers=list(f.verifiers),half_life_hours=6,
            cursor='same-original-cursor',global_lock='same-original-lock'))
        self.cutover=sign(actual.operator,{'original_cutover':True});self.anchor=sign(actual.operator,{'original_anchor':True})
        def validate(document,authority,cutover,anchor):
            self.assertEqual((document,authority,cutover,anchor),(self.pdoc,self.authority,self.cutover,self.anchor))
            return entry.authenticated(document,authority)
        def once(document,cutover,anchor,authority,*,execute,evidence_loader):
            validate(document,authority,cutover,anchor);self.calls.append((execute,evidence_loader))
            evidence=evidence_loader('unchanged-audit-config',authority=authority,cutoff=self.cutoff,
                                      verifiers=list(f.verifiers))
            return calculate(evidence['snapshots'],evidence['committed_at_by_epoch'],self.cutoff)
        self.writer=SimpleNamespace(validate_policy=validate,run_once=once)
        self.evidence=ModuleType('ops.current_assessment_evidence');ep=self.root/'legacy_evidence.py';ep.write_text('# fixture original evidence callbacks\n')
        self.evidence.__file__=str(ep)
        self.evidence.load_evidence=lambda *a,**k:deepcopy(self.legacy)
        self.evidence.snapshot=snapshot;self.evidence.observations=observations
        # This explicit fixture callback checks actual deduplicated observations;
        # production supplies the original pinned global-estimator function.
        def estimates(audits,miners,round,policy):
            self.assertEqual(len(audits),1);self.assertEqual(audits[0]['outcome'],'verified_valid')
            return {m:dict(validity_probability=2/3,reward_multiplier=1.,resolution_coverage_factor=1.)for m in miners}
        self.evidence.current_estimates=estimates
        self.modules=patch.dict(sys.modules,{'ops.current_assessment_evidence':self.evidence});self.modules.start();self.addCleanup(self.modules.stop)
        self.api=ModuleType('_mimo_fixture_writer_host');sys.modules[self.api.__name__]=self.api
        self.addCleanup(sys.modules.pop,self.api.__name__,None)
        old=self.root/'legacy_entry.py';old.write_text('from _mimo_fixture_writer_host import config,writer\ndef prepare(document): return config,writer\n')
        self.api.config={'parent_policy':self.save('writer.json',self.pdoc),'cutover':self.save('cutover.json',self.cutover),
                         'anchor':self.save('anchor.json',self.anchor)}
        self.api.writer=self.writer
        legacy=sign(actual.operator,dict(entry=self.pin(old)))
        source=Path(__import__('subnet').__file__).resolve().parents[1]
        self.payload=dict(version=entry.VERSION,execute_enabled=False,entry=self.pin(PATH),legacy_entry=self.pin(old),
            legacy_composition=self.save('legacy.json',legacy),evidence_module=self.pin(ep),
            new_source_root=str(source),new_source_files={p.relative_to(source).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
                for p in(source/'subnet').rglob('*.py')},reader_policy=self.save('reader.json',reader),allow_public=False,maximum_seconds=720)
        # Every test has the same fixture entry bytes but a different directory.
        self.alias='_affine_original_writer_'+self.pin(old)['sha256'][:24]
        sys.modules.pop(self.alias,None);self.addCleanup(sys.modules.pop,self.alias,None)
    def pin(self,path):return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    def save(self,name,value):
        p=self.root/name;p.write_bytes(canonical(value));p.chmod(0o600);return self.pin(p)
    def prepare(self,payload=None):return entry.prepare(sign(self.f.f.operator,payload or self.payload),self.authority)

    def test_actual_signed_queue_to_isolated_reader_to_same_writer_ema_and_uid_filter(self):
        original=sys.modules['subnet'];context=self.prepare()
        self.assertIs(sys.modules['subnet'],original)
        self.assertEqual(self.calls,[])
        value=entry.run(context,execute=False)
        self.assertEqual(value['half_life_hours'],6);self.assertGreater(value['points'][self.f.f.identity.id],0)
        self.assertFalse(value['chain_executed']);self.assertEqual(len(self.calls),1)
        roster={'registered-now':dict(uid=85,public_key=self.f.f.identity.id)}
        points,selected,excluded=recipients(value,roster)
        self.assertEqual(set(points),{'registered-now'});self.assertEqual(selected['registered-now']['uid'],85)
        self.assertEqual(recipients(value,{})[0],{})
        with self.assertRaisesRegex(ValueError,'activation'):entry.run(context,execute=True)
        self.assertEqual(len(self.calls),1)

    def test_source_pin_and_reader_roster_substitution_fail_before_writer(self):
        p=deepcopy(self.payload);p['new_source_files']['subnet/mimo_assessment_bridge.py']='0'*64
        with self.assertRaisesRegex(ValueError,'source bytes'):self.prepare(p)
        reader=json.loads(Path(self.payload['reader_policy']['path']).read_bytes())['payload']
        reader['verifiers']=['e'*64];p=deepcopy(self.payload)
        p['reader_policy']=self.save('other-reader.json',sign(self.f.f.operator,reader))
        with self.assertRaisesRegex(ValueError,'verifier identities'):self.prepare(p)
        self.assertEqual(self.calls,[])

    def test_unavailable_new_registry_preserves_original_fallback_semantics(self):
        context=self.prepare();previous=entry.run(context,execute=False)
        next(self.scheduler.root.glob('*.population.json')).write_bytes(b'{}')
        with self.assertRaises(ValueError):entry.run(context,execute=False)
        result=fallback(previous,self.cutoff+3600,'fixture evidence failure')
        self.assertEqual(result['points'],previous['points']);self.assertTrue(result['assessment_stale'])
        self.assertFalse(result['chain_executed'])

    def test_mutated_outer_signature_or_cached_alias_path_cannot_redirect_reader(self):
        document=sign(self.f.f.operator,self.payload);document['payload']['execute_enabled']=True
        with self.assertRaises(Exception):entry.prepare(document,self.authority)
        context=self.prepare()
        name=context['evidence_source'].__class__.__module__.split('.')[0]
        package=sys.modules[name];previous=package.__path__
        try:
            package.__path__=[str(self.root)]
            with self.assertRaisesRegex(ValueError,'package search path'):self.prepare()
        finally:package.__path__=previous
        self.assertEqual(self.calls,[])


if __name__=='__main__':unittest.main()
