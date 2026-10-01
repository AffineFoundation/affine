"""Upload approved remote checkpoint bytes with per-object delegated PUTs only."""
import argparse,hashlib,json
from pathlib import Path
import requests
from subnet.client import direct_r2_url


def main():
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--receipt',required=True);a=p.parse_args()
 c=json.loads(Path(a.config).read_text());rows=[]
 for name,item in c['files'].items():
  if Path(name).name!=name:raise ValueError('checkpoint filename')
  path=Path(c['checkpoint'])/name;h=hashlib.sha256()
  with path.open('rb') as f:
   while data:=f.read(1048576):h.update(data)
  if h.hexdigest()!=item['sha256']:raise ValueError('checkpoint byte mismatch')
  direct_r2_url(item['put_url'])
  with path.open('rb') as f:
   r=requests.put(item['put_url'],data=f,headers={'Content-Type':'application/octet-stream'},allow_redirects=False,timeout=900)
  if r.status_code!=200:raise RuntimeError('R2 checkpoint upload status '+str(r.status_code))
  rows.append(dict(name=name,sha256=h.hexdigest(),size=path.stat().st_size,status=r.status_code))
 Path(a.receipt).write_text(json.dumps(dict(checkpoint=c['checkpoint_id'],files=rows),indent=2))
if __name__=='__main__':main()
