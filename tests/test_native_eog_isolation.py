import unittest
import subprocess
import sys
import json
from unittest.mock import patch
from subnet.native_eog_isolation import container_command, IMAGE, NativeEOGSession, canonical

class EOGIsolationTests(unittest.TestCase):
    def test_fixed_image_and_network_boundary(self):
        cmd=container_command('affine-eog-controlled-'+'a'*16)
        self.assertEqual(cmd[-1],IMAGE)
        self.assertEqual(cmd[cmd.index('--network')+1],'none')
        self.assertIn('--read-only',cmd)
        self.assertNotIn('-v',cmd);self.assertNotIn('--mount',cmd);self.assertNotIn('-p',cmd)
        self.assertEqual(cmd[cmd.index('--cap-drop')+1],'ALL')

    def test_container_identity_rejects_shell_and_foreign_resources(self):
        for name in ('production','affine-eog-controlled-../','$(cat ~/.ssh/id_rsa)'):
            with self.assertRaises(ValueError):container_command(name)

    def test_unselected_tools_fail_before_transport(self):
        obj=object.__new__(NativeEOGSession);obj.tools={'create_calendar':{}}
        with patch.object(obj,'_rpc') as rpc:
            for name in ('sql-runner','seed-database','get_verifiers','../../api/sql-runner'):
                with self.assertRaises(ValueError):obj.call(name,{})
            rpc.assert_not_called()

    def test_canonical_rejects_nonfinite_claims(self):
        with self.assertRaises(ValueError):canonical({'reward':float('nan')})

    def test_grade_transport_route_rejects_unapproved_paths(self):
        obj=object.__new__(NativeEOGSession);obj.started=True
        with self.assertRaises(ValueError):obj._rpc('/api/delete-database',{})

    def test_fixed_clock_uuid_and_sqlite_seed_are_reproducible(self):
        code="""from subnet.native_eog_clock import install
install('a'*64,'2026-01-01T00:00:00+00:00')
import datetime,time,uuid,sqlite3,json
c=sqlite3.connect(':memory:')
print(json.dumps({'clock':datetime.datetime.now().isoformat(),'epoch':time.time(),
 'ids':[str(uuid.uuid4()),str(uuid.uuid4())],
 'sql':c.execute(\"SELECT datetime('now'), CURRENT_TIMESTAMP, datetime('2025-01-01','+1 day')\").fetchone()}))
"""
        a=subprocess.check_output([sys.executable,'-c',code]);b=subprocess.check_output([sys.executable,'-c',code])
        self.assertEqual(a,b);value=json.loads(a)
        self.assertEqual(value['sql'],['2026-01-01 00:00:00','2026-01-01 00:00:00','2025-01-02 00:00:00'])
        self.assertNotEqual(value['ids'][0],value['ids'][1])

if __name__=='__main__':unittest.main()
