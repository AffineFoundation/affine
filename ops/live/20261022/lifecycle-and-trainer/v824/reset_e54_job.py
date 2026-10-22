import sqlite3, sys
JOB = 'mimo-train-090278a66863e486d5dab648f6149b0e1e3af0f5b40b909e'
db = sqlite3.connect(sys.argv[1], timeout=30, isolation_level=None)
db.execute('BEGIN IMMEDIATE')
rows = db.execute('SELECT status, attempt, role FROM jobs WHERE id=?', (JOB,)).fetchall()
assert rows == [('failed', 3, 'train')], rows
cur = db.execute("DELETE FROM jobs WHERE id=? AND status='failed' AND role='train'", (JOB,))
assert cur.rowcount == 1, cur.rowcount
db.execute('COMMIT')
print('deleted failed e54 train job row', rows)
