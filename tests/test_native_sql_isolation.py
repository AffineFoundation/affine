import copy
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from subnet.native_sql_isolation import BASE,REVISION,RUNNER,SOURCE,command,grade
import hashlib

class SQLIsolationControls(unittest.TestCase):
    def runtime(self):
        return dict(revision=REVISION,image='sha256:'+'1'*64,base_image=BASE,
                    original_source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                    runner_sha256=hashlib.sha256(RUNNER.encode()).hexdigest())
    def test_unapproved_runtime_rejected_before_docker(self):
        for key,value in [('image','python:latest'),('original_source_sha256','0'*64),('runner_sha256','0'*64),('revision','other')]:
            runtime=self.runtime();runtime[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):command('affine-sql-controlled-'+'a'*16,runtime)
    def test_container_identity_cannot_inject_mounts(self):
        with self.assertRaises(ValueError):command('affine-sql-controlled- --volume /:/host',self.runtime())
    def test_changed_database_rejected_without_execution(self):
        with TemporaryDirectory() as directory:
            db=Path(directory)/'original.sqlite';db.write_bytes(b'changed')
            with patch('subnet.native_sql_isolation.subprocess.run',side_effect=AssertionError('unapproved execution')):
                with self.assertRaisesRegex(ValueError,'database closure'):grade({'db_path':str(db),'database_sha256':'0'*64},'```sql SELECT 1```',self.runtime())
    def test_reply_and_timeout_bounded_before_artifact_reads(self):
        with self.assertRaisesRegex(ValueError,'reply budget'):grade({},'x'*(1024*1024+1),self.runtime())
        with self.assertRaisesRegex(ValueError,'wall-clock'):grade({},'SELECT 1',self.runtime(),timeout=999)
