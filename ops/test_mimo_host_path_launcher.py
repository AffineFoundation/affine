import hashlib,json,os,subprocess,sys,tempfile,unittest
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from ops.mimo_host_path_launcher import UniquePath
LAUNCHER=Path(__file__).with_name('mimo_host_path_launcher.py')

class PathTests(unittest.TestCase):
    def test_insert_precedence(self):
        for initial in (['a','b','a'],['','a','','b'],['x','y']):
            for index in (-100,-1,0,1,100):
                for value in('a','b','z'):
                    # Initial installation already deduplicates once.
                    old=list(dict.fromkeys(initial));old.insert(index,value)
                    actual=UniquePath(initial);actual.insert(index,value)
                    self.assertEqual(actual,list(dict.fromkeys(old)))
    def test_append_extend_slice_and_arithmetic(self):
        p=UniquePath(['a','b']);p.append('a');p.extend(['c','b']);self.assertEqual(p,['a','b','c'])
        p[1:2]=['a','d'];self.assertEqual(p,['a','d','c'])
        p+=['d','e'];self.assertEqual(p,['a','d','c','e'])
        p*=3;self.assertEqual(p,['a','d','c','e']);p*=0;self.assertEqual(p,[])
    def test_nonstring_preserved_and_copy_plain(self):
        marker=[];p=UniquePath(['a',marker,'a',marker]);self.assertEqual(p,['a',marker,marker]);self.assertIs(type(p.copy()),list)
    def test_concurrent_insertions_bounded(self):
        p=UniquePath(['stdlib'])
        def loop(i):
            for _ in range(1000):p.insert(0,str(i))
        with ThreadPoolExecutor(max_workers=4)as pool:list(pool.map(loop,range(4)))
        self.assertEqual(set(p),{'stdlib','0','1','2','3'});self.assertEqual(len(p),5)
    def test_exact_launcher_reimport_installs_before_worker_module(self):
        with tempfile.TemporaryDirectory()as temp:
            root=Path(temp);(root/'ops').mkdir();(root/'subnet').mkdir();(root/'ops/__init__.py').write_text('');(root/'subnet/__init__.py').write_text('')
            (root/'subnet/mimo_worker.py').write_text('def source_inventory():return {}\n')
            entry=root/'ops/mimo_lifecycle.py';entry.write_text('''import sys,json,multiprocessing
from concurrent.futures import ProcessPoolExecutor
IMPORTED_WITH=type(sys.path).__name__
def initializer():assert type(sys.path).__name__=='UniquePath' and IMPORTED_WITH=='UniquePath'
def work(i):
 for _ in range(500):
  for p in ['/synthetic/a','/synthetic/b']:sys.path.insert(0,p)
 return dict(unique=len(sys.path)==len(set(sys.path)),type=type(sys.path).__name__,main=sys.modules['__main__'].__file__)
def main():
 with ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn'),initializer=initializer)as pool:rows=list(pool.map(work,range(4)))
 print(json.dumps(rows))
''')
            lc=root/'lifecycle.json';lc.write_text(json.dumps({'adapters':{'source_files':{}}}));lc.chmod(0o600)
            ref=lambda p:{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
            cfg=root/'operator.json';cfg.write_text(json.dumps({'version':'affine-host-import-path-housekeeping-v1','source_root':str(root),'entry':ref(entry),'lifecycle_config':ref(lc),'argv':['--config',str(lc),'--allow-public-mimo']}));cfg.chmod(0o600)
            run=subprocess.run([sys.executable,'-I','-B',str(LAUNCHER),'--operator-config',str(cfg)],capture_output=True,text=True,timeout=30)
            self.assertEqual(run.returncode,0,run.stderr)
            for row in json.loads(run.stdout):self.assertTrue(row['unique']);self.assertEqual(row['type'],'UniquePath');self.assertEqual(row['main'],str(LAUNCHER))

if __name__=='__main__':unittest.main()
