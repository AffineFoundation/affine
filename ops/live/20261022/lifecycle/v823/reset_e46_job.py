import sqlite3, sys
JOB = 'mimo-train-e5dcf2bff94195e93430c7f9a76885397c26b620462b047d'
db = sqlite3.connect(sys.argv[1], timeout=30, isolation_level=None)
db.execute('BEGIN IMMEDIATE')
rows = db.execute('SELECT status, attempt, role FROM jobs WHERE id=?', (JOB,)).fetchall()
assert rows == [('failed', 3, 'train')], rows
cur = db.execute("DELETE FROM jobs WHERE id=? AND status='failed' AND role='train'", (JOB,))
assert cur.rowcount == 1, cur.rowcount
db.execute('COMMIT')
print('deleted failed e46 train job row', rows)
