"""Activate only the qualified reader on the existing sole hourly writer."""
from pathlib import Path
import hashlib,json,os,subprocess,time
O=Path(__file__).resolve().parent
P=O/'parallel-evidence-v6'
A='3301134b38401196d006a621ac4a772bb4b0e6afa15a7d34a0d1ae5f2c630bcd'
q=json.loads((P/'FINAL-V5-QUALIFICATION.json').read_text())
assert q['completed'] and q['private_assessment_validated'] and q['evidence_seconds']<600
assert q['helper_sha256']==hashlib.sha256((P/'parallel_evidence_readers.py').read_bytes()).hexdigest()
assert q['entry_sha256']==hashlib.sha256((P/'hourly_population_index_entry_v6.py').read_bytes()).hexdigest()
env={**os.environ,**json.loads((P/'REQUIRED-HOST-ENV.json').read_text())}
cmd=['/home/const/subnet120-rewrite/.venv/bin/python','-I','-B',str(P/'hourly_population_index_entry_v6.py'),'--policy',str(P/'HOST-COMPOSITION.ROOT-SIGNED.private.json'),'--authority',A]
subprocess.run(cmd+['--check'],env=env,check=True)
unit=Path('/home/const/.config/systemd/user/affine-live-reward-writer.service.d/zzzz~~mimo-population-index.conf')
old=unit.read_bytes();backup=O/'ORIGINAL-writer-dropin.conf'
with backup.open('xb') as f:f.write(old)
lines=['[Service]','ExecStart=','ExecStart='+' '.join(cmd+['--execute'])]
for k,v in json.loads((P/'REQUIRED-HOST-ENV.json').read_text()).items():lines.append('Environment="'+k+'='+v+'"')
raw=('\n'.join(lines)+'\n').encode();tmp=unit.with_suffix('.tmp');tmp.write_bytes(raw);tmp.replace(unit)
subprocess.run(['systemctl','--user','daemon-reload'],check=True)
assert subprocess.check_output(['systemctl','--user','is-active','affine-live-reward-writer.timer'],text=True).strip()=='active'
(O/'ACTIVATION.json').write_text(json.dumps(dict(at=time.time(),previous_unit_sha256=hashlib.sha256(old).hexdigest(),new_unit_sha256=hashlib.sha256(raw).hexdigest(),manual_writer_invocations=0,chain_calls=0,cursor_changes=0,timer_retained=True),indent=2))
print('Activated qualified reader; existing timer and hourly cursor retained.')
