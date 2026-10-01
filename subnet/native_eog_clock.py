"""Disclosed controlled EOG runtime: fixed clock and seeded UUID stream.

Run only as the entry point of the approved derived Calendar image. Monotonic
clocks remain real, so server deadlines and event-loop scheduling still work.
Original API and SQL graders are not edited. This profile changes environmental
entropy, and must be part of the pinned environment identity.
"""
import datetime
import hashlib
import os
import sqlite3
import time
import uuid

REVISION='eog-calendar-fixed-clock-uuid-v1'

def install(seed, clock):
    if len(seed)!=64 or any(c not in '0123456789abcdef' for c in seed):raise ValueError('runtime seed')
    original=datetime.datetime
    instant=original.fromisoformat(clock)
    if instant.tzinfo!=datetime.timezone.utc:raise ValueError('runtime clock must be UTC')
    naive=instant.replace(tzinfo=None)
    class FixedDateTime(original):
        @classmethod
        def now(cls,tz=None):
            value=naive if tz is None else instant.astimezone(tz)
            return cls.fromisoformat(value.isoformat())
        @classmethod
        def utcnow(cls):return cls.fromisoformat(naive.isoformat())
        @classmethod
        def today(cls):return cls.now()
    datetime.datetime=FixedDateTime
    time.time=lambda:instant.timestamp()
    counter=0
    def uuid4():
        nonlocal counter
        counter+=1
        value=hashlib.sha256((seed+':'+str(counter)).encode()).digest()[:16]
        return uuid.UUID(bytes=value,version=4)
    uuid.uuid4=uuid4
    connect=sqlite3.connect
    def deterministic_connect(*args,**kwargs):
        connection=connect(*args,**kwargs)
        connection.create_function('current_timestamp',0,lambda:naive.strftime('%Y-%m-%d %H:%M:%S'))
        # The pinned SQL seed uses datetime('now'), not CURRENT_TIMESTAMP.
        # Delegate other SQLite date parsing/modifiers to an unmodified engine.
        reference=connect(':memory:')
        def sql_datetime(*values):
            values=list(values)
            if values and values[0]=='now':values[0]=naive.isoformat()
            query='SELECT datetime('+','.join('?' for _ in values)+')'
            return reference.execute(query,values).fetchone()[0]
        connection.create_function('datetime',-1,sql_datetime)
        return connection
    sqlite3.connect=deterministic_connect
    sqlite3.dbapi2.connect=deterministic_connect

if __name__=='__main__':
    install(os.environ['AFFINE_EOG_SEED'],os.environ['AFFINE_EOG_CLOCK'])
    import uvicorn
    uvicorn.run('main:app',host='0.0.0.0',port=8003,log_level='warning')
