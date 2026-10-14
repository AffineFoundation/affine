import json
import multiprocessing
import os
import signal
import subprocess
import sys
import tempfile
from pathlib import Path
from contextlib import contextmanager
import time
import unittest
from unittest.mock import patch
from ops.parallel_evidence_readers import compose_loader, _result


def merge(old, new, *, transition_policy):
    return dict(old=old, new=new, policy=transition_policy)


def old(*args, **kwargs):
    return dict(args=list(args), kwargs=kwargs)


def new(**kwargs):
    return dict(kwargs=kwargs)


class ParallelReadersTests(unittest.TestCase):
    def setUp(self):
        self.kw=dict(authority='a', cutoff=123, verifiers=['x'], extra='legacy-only')
        self.before={p.pid for p in multiprocessing.active_children()}

    def tearDown(self):
        self.assertEqual({p.pid for p in multiprocessing.active_children()},self.before)

    def test_exact_arguments_and_output(self):
        got=compose_loader(old,new,merge=merge,transition_policy='policy')('config',**self.kw)
        expected=merge(old('config',**self.kw),new(**{k:self.kw[k]for k in ('authority','cutoff','verifiers')}),transition_policy='policy')
        self.assertEqual(json.dumps(got,sort_keys=True),json.dumps(expected,sort_keys=True))

    def test_dictionary_order_and_floating_reduction_preserved(self):
        def reader(*a,**kw):return {'z':1e16,'a':-1e16,'b':1.0}
        def reduce(old,new,*,transition_policy):
            return {'old_keys':list(old),'new_keys':list(new),'sum':sum(old.values())}
        got=compose_loader(reader,reader,merge=reduce,transition_policy='p')(**self.kw)
        self.assertEqual(got,{'old_keys':['z','a','b'],'new_keys':['z','a','b'],'sum':1.0})

    def test_readers_actually_overlap(self):
        def reader(*a,**kw):
            start=time.monotonic();time.sleep(.15);return dict(start=start,end=time.monotonic(),pid=os.getpid())
        got=compose_loader(reader,reader,merge=merge,transition_policy='p')(**self.kw)
        self.assertNotEqual(got['old']['pid'],got['new']['pid'])
        self.assertLess(max(got['old']['start'],got['new']['start']),min(got['old']['end'],got['new']['end']))

    def test_errors_never_merge(self):
        def bad(*a,**kw):raise ValueError('bad signature')
        def forbidden(*a,**kw):self.fail('must not merge partial evidence')
        for a,b in ((bad,new),(old,bad)):
            with self.assertRaisesRegex(ValueError,'bad signature'):
                compose_loader(a,b,merge=forbidden,transition_policy='p')(**self.kw)

    def test_original_error_order(self):
        def first(*a,**kw):time.sleep(.05);raise TypeError('legacy failure')
        def second(*a,**kw):raise ValueError('mimo failure')
        with self.assertRaisesRegex(TypeError,'legacy failure'):
            compose_loader(first,second,merge=merge,transition_policy='p')(**self.kw)

    def test_timeout_kills_and_reaps_children(self):
        def stall(*a,**kw):time.sleep(60)
        with self.assertRaises(TimeoutError):
            compose_loader(stall,stall,merge=merge,transition_policy='p',timeout_seconds=.1)(**self.kw)

    def test_parent_alarm_cleanup(self):
        def stall(*a,**kw):time.sleep(60)
        def timeout(*a):raise TimeoutError('outer deadline')
        previous=signal.signal(signal.SIGALRM,timeout)
        try:
            signal.setitimer(signal.ITIMER_REAL,.1)
            with self.assertRaisesRegex(TimeoutError,'outer deadline'):
                compose_loader(stall,stall,merge=merge,transition_policy='p')(**self.kw)
        finally:
            signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,previous)

    def test_child_crash_is_failure(self):
        def crash(*a,**kw):os._exit(42)
        with self.assertRaises(RuntimeError):
            compose_loader(crash,new,merge=merge,transition_policy='p')(**self.kw)

    def test_result_schema_fail_closed(self):
        for doc in ({},[],{'ok':True},{'ok':True,'value':None},{'ok':False,'error':3,'message':'bad'}):
            with self.assertRaises(ValueError):_result(json.dumps(doc).encode())

    def test_no_fork_when_multithreaded(self):
        with patch('ops.parallel_evidence_readers.threading.active_count',return_value=2),patch('ops.parallel_evidence_readers.multiprocessing.get_context',side_effect=AssertionError('unsafe fork')):
            self.assertEqual(compose_loader(old,new,merge=merge,transition_policy='p')(**self.kw)['policy'],'p')

    def test_no_fork_on_non_linux(self):
        with patch('ops.parallel_evidence_readers.sys.platform','darwin'),patch('ops.parallel_evidence_readers.multiprocessing.get_context',side_effect=AssertionError('unsafe fork')):
            self.assertEqual(compose_loader(old,new,merge=merge,transition_policy='p')(**self.kw)['policy'],'p')

    def test_native_threads_prevent_fork(self):
        for tasks in (['1','2'],None):
            change=patch('ops.parallel_evidence_readers.os.listdir',return_value=tasks) if tasks else patch('ops.parallel_evidence_readers.os.listdir',side_effect=OSError('proc unavailable'))
            with change,patch('ops.parallel_evidence_readers.multiprocessing.get_context',side_effect=AssertionError('unsafe fork')):
                self.assertEqual(compose_loader(old,new,merge=merge,transition_policy='p')(**self.kw)['policy'],'p')

    def test_ten_minute_host_budget(self):
        got=compose_loader(old,new,merge=merge,transition_policy='p',timeout_seconds=600)(**self.kw)
        self.assertEqual(got['policy'],'p')

    def test_deadline_is_bounded(self):
        for timeout in (True,0,-1,601,float('nan')):
            with self.assertRaises(ValueError):compose_loader(old,new,merge=merge,transition_policy='p',timeout_seconds=timeout)

    def test_large_pipe_payload_completes(self):
        def large(*a,**kw):return {'body':'x'*(2*1024**2)}
        got=compose_loader(large,large,merge=merge,transition_policy='p')(**self.kw)
        self.assertEqual(len(got['old']['body']),2*1024**2)
        self.assertEqual(got['old'],got['new'])

    def test_oversized_result_never_merges(self):
        def large(*a,**kw):return {'body':'x'*1000}
        with patch('ops.parallel_evidence_readers.MAX_RESULT_BYTES',300):
            with self.assertRaisesRegex(ValueError,'bounded evidence child response'):
                compose_loader(large,new,merge=merge,transition_policy='p')(**self.kw)

    def test_cache_factory_enters_only_legacy_child(self):
        parent=os.getpid()
        @contextmanager
        def factory():
            if os.getpid()==parent:raise AssertionError('cache opened in parent')
            yield os.getpid()
        def legacy(*a,**kw):
            with kw['factory']() as pid:return {'cache_pid':pid}
        got=compose_loader(legacy,new,merge=merge,transition_policy='p')(**self.kw,factory=factory)
        self.assertNotEqual(got['old']['cache_pid'],parent)
        self.assertNotIn('factory',got['new']['kwargs'])

    def test_parent_death_kills_children(self):
        with tempfile.TemporaryDirectory() as directory:
            script="""import os,time,sys
from pathlib import Path
from ops.parallel_evidence_readers import compose_loader
def reader(*a,**kw):
 Path(sys.argv[1],str(os.getpid())).write_text('started')
 time.sleep(60)
def merge(*a,**kw):return {}
compose_loader(reader,reader,merge=merge,transition_policy='p')(authority='a',cutoff=1,verifiers=['v'])
"""
            parent=subprocess.Popen([sys.executable,'-c',script,directory])
            try:
                deadline=time.monotonic()+5
                while len(list(Path(directory).iterdir()))<2 and time.monotonic()<deadline:time.sleep(.02)
                pids=[int(p.name)for p in Path(directory).iterdir()]
                self.assertEqual(len(pids),2)
                parent.kill();parent.wait(timeout=5)
                deadline=time.monotonic()+5
                def alive(pid):
                    try:return Path('/proc',str(pid),'stat').read_text().split(') ')[1][0]!='Z'
                    except FileNotFoundError:return False
                while any(alive(pid)for pid in pids)and time.monotonic()<deadline:time.sleep(.02)
                self.assertFalse(any(alive(pid)for pid in pids))
            finally:
                if parent.poll()is None:parent.kill();parent.wait(timeout=5)

    def test_non_json_success_fails_closed(self):
        def invalid(*a,**kw):return {'bad':float('nan')}
        with self.assertRaises(ValueError):compose_loader(invalid,new,merge=merge,transition_policy='p')(**self.kw)

if __name__=='__main__':unittest.main()
