"""Offline tests for monitor scheduling, publication and defensive rendering inputs."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from . import collector, common, build_panels, validator, submissions, scoring, evalpods

class MonitorTests(unittest.TestCase):
    def test_atomic_publication(self):
        with tempfile.TemporaryDirectory() as root:
            directory=Path(root)
            collector.publish(directory,'duel',common.panel('Test','Fixture'))
            self.assertEqual(json.loads((directory/'duel.json').read_text())['title'],'Test')
            self.assertEqual(len(list(directory.iterdir())),1)
            with self.assertRaises(ValueError):
                collector.publish(directory,'duel',{'bad':float('nan')})
            self.assertEqual(json.loads((directory/'duel.json').read_text())['title'],'Test')

    def test_failure_does_not_leak(self):
        with patch.object(collector.importlib,'import_module',side_effect=ValueError('SECRET_SENTINEL')):
            result=collector.collect_group('scoring',10)
        self.assertEqual(set(result),set(collector.GROUPS['scoring'][1]))
        self.assertNotIn('SECRET_SENTINEL',json.dumps(result))
        self.assertTrue(all(d['status']=='error' for d in result.values()))

    def test_wrong_group_keys_fail_closed(self):
        with patch.object(collector.importlib,'import_module') as module:
            module.return_value.collect.return_value={'private':{'raw':'SECRET_SENTINEL'}}
            result=collector.collect_group('validator',10)
        self.assertEqual(set(result),{'validator','weights'})
        self.assertNotIn('SECRET_SENTINEL',json.dumps(result))

    def test_missing_validator_is_unknown(self):
        with patch.object(validator,'_load',return_value=({},'Unavailable')):
            result=validator.collect()
        self.assertEqual(result['validator']['status'],'warn')
        self.assertIn('Unknown',json.dumps(result))

    def test_missing_submissions_not_empty_queue(self):
        with patch.object(submissions,'_load',return_value=({},'Unavailable')):
            result=submissions.collect()
        self.assertEqual(result['queue']['metrics'][0]['value'],'Unknown')

    def test_history_gaps_are_not_zero(self):
        result=scoring._history([{'challenge_id':'chal-1','event':'verdict'}],[])
        self.assertIsNone(result['charts'][0]['series'][0]['values'][0])

    def test_loading_chat_not_ready(self):
        result=evalpods._render('chat','Chat',9002,{'ok':True,'state':'loading'})
        self.assertEqual(result['status'],'warn')
        self.assertEqual(result['metrics'][0]['value'],'Not ready')

    def test_offline_overview_awaits_sources(self):
        result=collector.overview({})
        self.assertEqual(result['status'],'warn')
        self.assertTrue(all(n['status']=='unknown' for n in result['nodes']))

    def test_stale_overview_does_not_claim_healthy(self):
        value=common.panel('Validator','Stale fixture')
        value['collected_at']='2000-01-01T00:00:00+00:00'
        result=collector.overview({'validator':value})
        node=next(n for n in result['nodes'] if n['id']=='validator')
        self.assertEqual(node['status'],'unknown')

    def test_script_string_escapes_html(self):
        value=build_panels.script_string('</script><script>alert(1)</script>')
        self.assertNotIn('<',value)
        self.assertEqual(json.loads(value),'</script><script>alert(1)</script>')

    def test_generated_panels_current(self):
        self.assertEqual(len(build_panels.build(common.ROOT/'panels',common.ROOT/'panels/data',check=True)),23)

    def test_panels_avoid_html_injection(self):
        template=build_panels.TEMPLATE.read_text()
        self.assertNotIn('innerHTML',template)
        self.assertNotIn('<script src=',template)
        self.assertIn('a.subscribe(',template)
        self.assertIn('a.system(host',template)

if __name__=='__main__':
    unittest.main()
