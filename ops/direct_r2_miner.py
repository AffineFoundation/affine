"""Owned remote pilot: fail closed on any HTTP request outside direct R2.

Bootstrap contains only a public-discovery read capability and owned identity path;
never account credentials. Request evidence excludes URLs, queries and key material.
"""
import argparse,json,os,sys,time
from pathlib import Path
from urllib.parse import urlsplit
import requests


def main():
    p=argparse.ArgumentParser();p.add_argument('--bootstrap',required=True);p.add_argument('--evidence',required=True);a=p.parse_args()
    config=json.loads(Path(a.bootstrap).read_text());target=Path(a.evidence);target.parent.mkdir(parents=True,exist_ok=True)
    evidence=json.loads(target.read_text()) if target.exists() else dict(events=[],blocked_requests=0,started_at=time.time())
    original=requests.sessions.Session.request
    def record():
        tmp=target.with_suffix('.tmp');tmp.write_text(json.dumps(evidence));tmp.chmod(0o600);tmp.replace(target)
    def guarded(session,method,url,*args,**kwargs):
        parsed=urlsplit(url);host=parsed.hostname or ''
        if parsed.scheme!='https' or not host.endswith('.r2.cloudflarestorage.com'):
            evidence['blocked_requests']+=1;evidence['events'].append(dict(timestamp=time.time(),method=method,host=host,status='blocked'));record()
            raise ValueError('pilot attempted non-R2 request')
        kwargs['allow_redirects']=False
        result=original(session,method,url,*args,**kwargs)
        evidence['events'].append(dict(timestamp=time.time(),method=method,host=host,status=result.status_code));record()
        return result
    requests.sessions.Session.request=guarded
    sys.argv=['miner','--gateway','https://unused-gateway.invalid','--authority',config['authority'],'--current-url',config['current_url'],'--key',config['key'],'--state',config['state'],'--max-batches',str(config.get('max_batches',2))]
    from subnet.cli import main as miner
    miner()

if __name__=='__main__':main()
