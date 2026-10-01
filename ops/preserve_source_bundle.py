"""Publish credential-free historical code/task bundle with signed hash descriptor."""
import argparse,base64,hashlib,json,shutil,tarfile,time
from pathlib import Path
from subnet.storage import Bucket,Identity,canonical

def main():
 p=argparse.ArgumentParser();p.add_argument('epoch');a=p.parse_args();state=Path('state/multi-environment');source=state/'miner-source.tar.gz';digest=hashlib.sha256(source.read_bytes()).hexdigest();destination=Path('state/source-bundles')/(digest+'.tar.gz');destination.parent.mkdir(exist_ok=True)
 with tarfile.open(source) as tar:
  members=tar.getmembers()
  for member in members:
   name=member.name
   if Path(name).is_absolute() or '..' in Path(name).parts or member.issym() or member.islnk() or Path(name).name=='.env' or 'wallet' in name or (name.startswith('state/') and not name.startswith('state/original-task-snapshots/')):
    raise ValueError('unsafe historical source member')
 shutil.copy2(source,destination)
 metadata={'sha256':digest,'bytes':destination.stat().st_size,'members':len(members),'credentials_included':False,'epochs':[a.epoch],'public_path':'/public/source-bundles/'+destination.name,'created_at':time.time()}
 authority=Identity(bytes.fromhex((state/'authority.seed').read_text()));signed={'payload':metadata,'signer':authority.id,'signature':base64.b64encode(authority.key.sign(canonical(metadata)).signature).decode()}
 bucket=Bucket(json.loads(Path('state/mock-r2.json').read_text()));bucket.upload('public/source-bundles/'+destination.name,destination);bucket.json('public/source-bundles/'+digest+'-signed.json',signed);destination.with_name(digest+'-signed.json').write_bytes(canonical(signed));print(json.dumps(metadata))

if __name__=='__main__':main()
