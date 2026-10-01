"""Public signed manifest fetch and private key handling for a real miner."""
import base64
import json
import hashlib
import os
from urllib.parse import urlparse,parse_qs
from pathlib import Path
import requests
from nacl.signing import VerifyKey
from .storage import canonical,Identity,sha

def direct_r2_url(url):
    if not isinstance(url,str):raise ValueError('missing direct R2 URL')
    parsed=urlparse(url);query=parse_qs(parsed.query)
    if parsed.scheme!='https' or not (parsed.hostname or '').endswith('.r2.cloudflarestorage.com') or parsed.username or parsed.password or not query.get('X-Amz-Signature') or query.get('X-Amz-Algorithm')!=['AWS4-HMAC-SHA256']:
        raise ValueError('invalid direct R2 URL')
    return url

def fetch_signed(url,authority):
    response=requests.get(url,timeout=60);response.raise_for_status();value=response.json()
    if value['signer']!=authority:raise ValueError('wrong authority')
    VerifyKey(bytes.fromhex(authority)).verify(canonical(value['payload']),base64.b64decode(value['signature']))
    return value['payload']

def identity(path):
    path=Path(path)
    if path.exists():return Identity(bytes.fromhex(path.read_text().strip()))
    path.parent.mkdir(parents=True,exist_ok=True)
    key=Identity();path.write_text(key.key.encode().hex());path.chmod(0o600);return key

def checkpoint_download(manifest,destination):
    direct=manifest.get('transport_policy')=='direct-r2-v1'
    urls=manifest['checkpoint'].get('read_urls',{})
    if direct:
        if not isinstance(urls,dict) or set(urls)!=set(manifest['checkpoint']['files']):raise ValueError('direct R2 checkpoint URL file binding')
        for url in urls.values():direct_r2_url(url)
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    def digest(path):
        h=hashlib.sha256()
        with path.open('rb') as f:
            while data:=f.read(1024*1024):h.update(data)
        return h.hexdigest()
    for name,expected in manifest['checkpoint']['files'].items():
        if Path(name).name!=name:raise ValueError('checkpoint path')
        path=destination/name
        if path.exists() and digest(path)==expected:continue
        # Identical files can belong to multiple immutable checkpoint manifests.
        # Only reuse cached bytes after checking the new authority-pinned hash.
        for folder in destination.parent.iterdir():
            candidate=folder/name
            if folder!=destination and folder.is_dir() and candidate.is_file() and digest(candidate)==expected:
                temporary=path.with_suffix(path.suffix+'.tmp')
                temporary.unlink(missing_ok=True);os.link(candidate,temporary);temporary.replace(path);break
        if path.exists() and digest(path)==expected:continue
        temporary=path.with_suffix(path.suffix+'.tmp')
        for attempt in range(3):
            try:
                h=hashlib.sha256();size=0
                url=urls[name] if direct else urls.get(name) or manifest['checkpoint']['base_url']+'/'+name
                with requests.get(url,timeout=180,stream=True) as response:
                    response.raise_for_status()
                    with temporary.open('wb') as out:
                        for part in response.iter_content(1024*1024):
                            size+=len(part)
                            if size>20_000_000_000:raise ValueError('checkpoint file budget')
                            h.update(part);out.write(part)
                if h.hexdigest()!=expected:raise ValueError('checkpoint integrity')
                temporary.replace(path);break
            except requests.RequestException:
                temporary.unlink(missing_ok=True)
                if attempt==2:raise
            except Exception:
                temporary.unlink(missing_ok=True);raise
    return destination
