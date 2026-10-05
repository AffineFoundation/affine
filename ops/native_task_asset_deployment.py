"""CPU deployment of an exact approved task snapshot outside immutable code."""
import hashlib,io,json,os,secrets,tarfile,urllib.request
from pathlib import Path
SNAPSHOT='assets/original-math7496.tasks.json'
SHA='77a4524abc279d0e6e95ec87d0e5604f501c8e409ac5ab656ecabf060ab50fe3'
SIZE=8146242
class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,*args,**kwargs):raise ValueError('task asset redirect forbidden')
def approved_snapshot(archive,archive_sha256,archive_size):
    if len(archive)!=archive_size or hashlib.sha256(archive).hexdigest()!=archive_sha256:raise ValueError('approved archive integrity')
    with tarfile.open(fileobj=io.BytesIO(archive),mode='r:gz')as tar:
        rows=[m for m in tar.getmembers()if m.name==SNAPSHOT]
        if len(rows)!=1 or not rows[0].isfile()or rows[0].size!=SIZE:raise ValueError('exact regular snapshot member')
        data=tar.extractfile(rows[0]).read(SIZE+1)
    if len(data)!=SIZE or hashlib.sha256(data).hexdigest()!=SHA:raise ValueError('approved snapshot integrity')
    return data
def download_snapshot(url,archive_sha256,archive_size,opener=None):
    if not isinstance(url,str)or not url.startswith('https://')or not 0<archive_size<=32*1024**2:raise ValueError('bounded HTTPS approved archive')
    opener=opener or urllib.request.build_opener(NoRedirect)
    with opener.open(urllib.request.Request(url,method='GET'),timeout=30)as response:
        if response.status!=200:raise ValueError('approved archive GET')
        data=response.read(archive_size+1)
    return approved_snapshot(data,archive_sha256,archive_size)
def install_snapshot(data,root):
    if len(data)!=SIZE or hashlib.sha256(data).hexdigest()!=SHA:raise ValueError('snapshot integrity before install')
    root=Path(root)
    if not root.is_absolute()or root.absolute()!=root.resolve():raise ValueError('unaliased absolute asset root')
    root.mkdir(mode=0o700,parents=True,exist_ok=True)
    if root.stat().st_uid!=os.geteuid()or root.stat().st_mode&0o022:raise ValueError('private owned asset root')
    folder=root/SHA
    if folder.is_symlink():raise ValueError('asset directory symlink')
    folder.mkdir(mode=0o700,exist_ok=True)
    if folder.stat().st_uid!=os.geteuid()or folder.stat().st_mode&0o022:raise ValueError('private owned asset directory')
    target=folder/'original-math7496.tasks.json'
    def checked():
        if target.is_symlink()or not target.is_file()or target.stat().st_size!=SIZE or hashlib.sha256(target.read_bytes()).hexdigest()!=SHA:raise ValueError('existing task snapshot changed')
        return target
    if target.exists()or target.is_symlink():return checked()
    tmp=folder/('.incomplete-'+secrets.token_hex(12));fd=os.open(tmp,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(fd,'wb')as f:f.write(data);f.flush();os.fsync(f.fileno())
    try:
        try:os.link(tmp,target)
        except FileExistsError:checked()
    finally:tmp.unlink(missing_ok=True)
    dirfd=os.open(folder,os.O_RDONLY)
    try:os.fsync(dirfd)
    finally:os.close(dirfd)
    return checked()
def path_only_config(config,path):
    c=json.loads(json.dumps(config));changed=0
    for row in c['environments']:
        spec=row['spec']
        if spec['id']=='affine_math':
            if spec['config'].get('task_snapshot')!=SNAPSHOT:raise ValueError('only original relative snapshot can move')
            spec['config']['task_snapshot']=str(path);changed+=1
    if changed!=1:raise ValueError('exact one original MATH environment')
    return c
def hydrate_snapshot(archive,root,opener=None):
    """Every deployment checks local SHA; only cold nodes download the archive."""
    p=Path(root)/SHA/'original-math7496.tasks.json'
    if p.exists()or p.is_symlink():
        if p.is_symlink()or not p.is_file()or p.stat().st_size!=SIZE or hashlib.sha256(p.read_bytes()).hexdigest()!=SHA:raise ValueError('existing snapshot integrity')
        return install_snapshot(p.read_bytes(),root)
    return install_snapshot(download_snapshot(archive['read_url'],archive['sha256'],archive['size'],opener),root)
def verify_archive_descriptor(document,authority):
    import base64
    from nacl.signing import VerifyKey
    if set(document)!={'payload','signature','signer'}or document['signer']!=authority:raise ValueError('approved authority archive descriptor')
    VerifyKey(bytes.fromhex(authority)).verify(json.dumps(document['payload'],sort_keys=True,separators=(',',':'),allow_nan=False).encode(),base64.b64decode(document['signature'],validate=True))
    a=document['payload']['source_bundle']
    if a['sha256']!='7459c28cbe11b0999b41644aea2aa672d40eb44faa94ccb261e176d8f71b0d46'or a['format']!='tar.gz':raise ValueError('exact original approved dataset archive')
    return a
def main():
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--descriptor',required=True);p.add_argument('--authority',default='3301134b38401196d006a621ac4a772bb4b0e6afa15a7d34a0d1ae5f2c630bcd');p.add_argument('--asset-root',default='/var/tmp/affine-approved-native-task-assets');a=p.parse_args()
    archive=verify_archive_descriptor(json.loads(Path(a.descriptor).read_bytes()),a.authority);target=hydrate_snapshot(archive,a.asset_root)
    print(json.dumps(dict(snapshot_path=str(target),sha256=SHA,size=SIZE,model_or_GPU_created=False,calibration_dispatched=False)))
if __name__=='__main__':main()
