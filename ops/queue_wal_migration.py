"""Explicit quiescent queue WAL migration; no job/authentication/schema rewrite.

The caller must prove its authorized API/auditor writer handoff. This helper
never stops a service or signals a worker. FULL applies to every new connection
in the reviewed operator overlay; enabling WAL itself is an explicit action.
"""
import hashlib,json,math,sqlite3,time
from pathlib import Path

def _inode(path,expected):
    path=Path(path)
    if not path.is_absolute()or path!=path.resolve(strict=True):raise ValueError('exact authoritative queue path')
    s=path.stat()
    if [s.st_dev,s.st_ino]!=list(expected):raise ValueError('authoritative queue inode changed')

def logical_snapshot(db):
    schema=db.execute("SELECT type,name,tbl_name,sql FROM sqlite_master WHERE name NOT LIKE 'sqlite_autoindex_%' ORDER BY type,name").fetchall()
    h=hashlib.sha256();counts={}
    h.update(json.dumps(schema,separators=(',',':'),ensure_ascii=False).encode())
    for name, in db.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name").fetchall():
        quoted='"'+name.replace('"','""')+'"';columns=db.execute('PRAGMA table_info('+quoted+')').fetchall();keys=sorted((c[5],c[1])for c in columns if c[5]);order=','.join('"'+key.replace('"','""')+'"'for _,key in keys)if keys else 'rowid'
        count=0
        for row in db.execute('SELECT * FROM '+quoted+' ORDER BY '+order):
            count+=1
            for v in row:
                # Do not parse/reformat signed JSON or expose capability URLs.
                if v is None:data=b'n'
                elif isinstance(v,bytes):data=b'b'+v
                elif isinstance(v,str):data=b's'+v.encode()
                elif isinstance(v,int):data=b'i'+str(v).encode()
                elif isinstance(v,float):data=b'f'+repr(v).encode()
                else:raise ValueError('unsupported authoritative SQLite value')
                h.update(len(data).to_bytes(8,'big'));h.update(data)
        counts[name]=count
    return dict(logical_sha256=h.hexdigest(),table_rows=counts,schema_sha256=hashlib.sha256(json.dumps(schema,separators=(',',':')).encode()).hexdigest())

def migrate(path,expected_inode,*,writer_handoff_guard,timeout_seconds=30):
    if type(timeout_seconds)not in(int,float)or not math.isfinite(timeout_seconds)or not 0<timeout_seconds<=120:raise ValueError('bounded WAL migration wait')
    _inode(path,expected_inode);writer_handoff_guard();deadline=time.monotonic()+timeout_seconds
    db=sqlite3.connect('file:'+str(path)+'?mode=rw',uri=True,timeout=.25,isolation_level=None)
    try:
        db.execute('PRAGMA synchronous=FULL')
        before=logical_snapshot(db);writer_handoff_guard();_inode(path,expected_inode)
        original=db.execute('PRAGMA journal_mode').fetchone()[0]
        while True:
            try:
                mode=db.execute('PRAGMA journal_mode=WAL').fetchone()[0]
                if mode.lower()!='wal':raise ValueError('explicit WAL journal mode refused')
                break
            except sqlite3.OperationalError as error:
                if not any(x in str(error).lower()for x in ('locked','busy')):raise
                writer_handoff_guard();_inode(path,expected_inode)
                if time.monotonic()>=deadline:raise TimeoutError('bounded quiescent WAL migration unavailable')from error
                time.sleep(min(.05,max(0,deadline-time.monotonic())))
        _inode(path,expected_inode);writer_handoff_guard()
        after=logical_snapshot(db)
        if after!=before:raise ValueError('authoritative records changed during WAL migration; preserve evidence')
        if db.execute('PRAGMA synchronous').fetchone()[0]!=2:raise ValueError('FULL synchronous required')
        return dict(version='authoritative-queue-WAL-FULL-migration-v1',original_journal_mode=original,journal_mode='wal',synchronous='FULL',queue_device_inode=list(expected_inode),all_records_and_schema_preserved=True,**after)
    finally:db.close()
