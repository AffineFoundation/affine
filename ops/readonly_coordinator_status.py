"""Pinned operational copy of the reviewed pure status SELECT; no queue writes."""
import json
import random
import sqlite3
import time
from pathlib import Path


def status(self, identifier):
    deadline = time.monotonic()+30
    while True:
        db = None
        remaining = deadline-time.monotonic()
        if remaining <= 0:
            raise TimeoutError('bounded read-only status')
        try:
            db = sqlite3.connect(Path(self.path).resolve().as_uri()+'?mode=ro', uri=True,
                                 timeout=min(.25, remaining))
            db.row_factory = sqlite3.Row
            row = db.execute('SELECT status,report,report_digest,worker,attempt FROM jobs WHERE id=?', (identifier,)).fetchone()
            break
        except sqlite3.OperationalError as error:
            if not any(word in str(error).lower() for word in ('locked', 'busy')):
                raise
        finally:
            if db is not None:
                db.close()
        time.sleep(min(max(0, deadline-time.monotonic()), random.SystemRandom().uniform(.025, .05)))
    if row is None:
        raise ValueError('unknown job')
    result = dict(row)
    result['report'] = json.loads(result['report']) if result['report'] else None
    return result
