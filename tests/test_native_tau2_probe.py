import json,os,pathlib,tempfile,unittest
from unittest.mock import patch
from subnet.native_tau2_probe import MockTransport,REVISION,data_inventory,digest,run_probe,sanitized_env

class NativeTau2ProbeTests(unittest.TestCase):
    def test_missing_or_wrong_revision_stops_before_process(self):
        with tempfile.TemporaryDirectory() as d:
            root=pathlib.Path(d);data=root/'data';data.mkdir()
            with patch('subnet.native_tau2_probe.subprocess.Popen') as popen:
                with self.assertRaisesRegex(ValueError,'revision'):run_probe(data,root/'out')
                (data/'.tau2_revision').write_text('798589')
                with self.assertRaisesRegex(ValueError,'revision'):run_probe(data,root/'out')
                popen.assert_not_called()
    def test_invalid_budgets_stop_before_process(self):
        for kwargs in [dict(index=-1),dict(index=2171),dict(max_steps=500),dict(wall_seconds=181)]:
            with patch('subnet.native_tau2_probe.subprocess.Popen') as popen:
                with self.assertRaises(ValueError):run_probe('/absent','/absent',**kwargs)
                popen.assert_not_called()
    def test_child_env_does_not_inherit_credentials(self):
        with patch.dict(os.environ,{'ENGY':'should-not-export','OPENAI_API_KEY':'should-not-export','OP_SERVICE_ACCOUNT_TOKEN':'should-not-export','AWS_SECRET_ACCESS_KEY':'should-not-export'},clear=False):
            env=sanitized_env('/tmp/data')
            for key in ['ENGY','OPENAI_API_KEY','OP_SERVICE_ACCOUNT_TOKEN','AWS_SECRET_ACCESS_KEY']:self.assertNotIn(key,env)
            self.assertEqual(env['TAU2_DATA_DIR'],'/tmp/data')
    def test_mock_uses_real_tool_and_labels_nonproof_receipts(self):
        mock=MockTransport();request={'model':'native-probe-agent','messages':[]}
        mock.response(request);response=mock.response(request)
        tool=response['choices'][0]['message']['tool_calls'][0]['function']
        self.assertEqual(tool['name'],'get_customer_by_phone')
        self.assertEqual(json.loads(tool['arguments']),{'phone_number':'555-123-4567'})
        self.assertFalse(mock.calls[1]['authenticated_model_receipt'])
        self.assertEqual(mock.calls[1]['request_hash'],digest(request))
        for _ in range(3):response=mock.response({'model':'native-probe-user'})
        self.assertEqual(response['choices'][0]['message']['content'],'###STOP###')
    def test_data_inventory_binds_db_and_task_changes(self):
        with tempfile.TemporaryDirectory() as d:
            p=pathlib.Path(d);(p/'db.toml').write_text('one')
            first=data_inventory(p);(p/'db.toml').write_text('two')
            self.assertNotEqual(digest(first),digest(data_inventory(p)))

if __name__=='__main__':unittest.main()
