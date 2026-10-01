import unittest
import tempfile
import tarfile
import io
from pathlib import Path
import hashlib
from subnet.backend_jobs import canonical
from unittest.mock import patch
from types import SimpleNamespace

from subnet.native_wikispeedia import public_path,execute,replay,verify_public_resources,public_candidate_harness


class NativeWikispeediaTests(unittest.TestCase):
    def test_candidate_tools_roundtrip_through_shared_dialect(self):
        from subnet.harness import action,turn_config
        tools=[{'function':{'name':'click_link','parameters':{'properties':{'article':{'type':'string'}}}}}]
        config=public_candidate_harness('a','c',tools,{'a':['b'],'b':['c'],'c':[]},3)
        self.assertEqual(config['max_turns'],3)
        parsed=[action(turn_config(config['harness'],i)['candidates'][0],config['harness']) for i in range(3)]
        self.assertEqual([v['tool_calls'][0]['arguments']['article'] for v in parsed[:2]],['b','c'])
        self.assertEqual(parsed[-1]['tool_calls'],[])
        self.assertTrue(config['public_starter_control_not_autonomous_search'])

    def test_candidate_harness_rejects_wrong_schema_or_false_negative(self):
        tools=[{'function':{'name':'click_link','parameters':{'properties':{'article':{'type':'string'}}}}}]
        with self.assertRaisesRegex(ValueError,'unambiguous'):
            public_candidate_harness('a','b',tools+tools,{'a':['b'],'b':[]})
        with self.assertRaisesRegex(ValueError,'unreachable'):
            public_candidate_harness('a','c',tools,{'a':['b'],'b':['c'],'c':[]},2)
        with self.assertRaisesRegex(ValueError,'unavailable'):
            public_candidate_harness('a','b',tools,{'a':['b','__affine_invalid_article__'],'b':[]})
    def test_public_directed_links_cycles_and_budget(self):
        links={'a':['b'],'b':['a','c'],'c':['d'],'d':[]}
        self.assertEqual(public_path('a','d',links,3),['b','c','d'])
        with self.assertRaisesRegex(ValueError,'unreachable'):public_path('a','d',links,2)
        with self.assertRaisesRegex(ValueError,'unreachable'):public_path('d','a',links,30)

    def test_extracted_graph_must_match_original_resource_archive(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            for archive,name in [('wikispeedia_paths-and-graph.tar.gz','graph/links.tsv'),
                                 ('wikispeedia_articles_plaintext.tar.gz','articles/page.txt')]:
                path=root/name;path.parent.mkdir(parents=True);path.write_bytes(b'original public resource')
                with tarfile.open(root/archive,'w:gz') as tar:
                    member=tarfile.TarInfo(name);member.size=path.stat().st_size
                    tar.addfile(member,io.BytesIO(path.read_bytes()))
            self.assertEqual(len(verify_public_resources(root)['extracted_files']),2)
            (root/'graph/links.tsv').write_bytes(b'forged target edge')
            with self.assertRaisesRegex(ValueError,'resource changed'):verify_public_resources(root)

    def test_unapproved_environment_or_source_never_executes_task(self):
        spec=SimpleNamespace(id='other',max_turns=30,source_hash='approved')
        with patch('subnet.native_wikispeedia.create_session') as create:
            with self.assertRaisesRegex(ValueError,'specification'):execute(spec,0,0,[{'text':'Done'}])
            spec.id='affine_wikispeedia'
            with self.assertRaisesRegex(ValueError,'source binding'):
                replay(spec,dict(environment_id=spec.id,source_hash='unapproved'))
            create.assert_not_called()

    def test_committed_tool_observation_cannot_replace_native_result(self):
        spec=SimpleNamespace(id='affine_wikispeedia',source_hash='approved',to_dict=lambda:dict(id='affine_wikispeedia',source_hash='approved',max_turns=30))
        genuine=dict(environment_id=spec.id,source_hash='approved',
            environment_definition_sha256=hashlib.sha256(canonical(spec.to_dict())).hexdigest(),index=0,seed=1,
            reward=0.,turns=[dict(action={'text':'Done'},result={'observations':[]})])
        forged=dict(genuine,reward=1.)
        with patch('subnet.native_wikispeedia.execute',return_value=genuine):
            with self.assertRaisesRegex(ValueError,'outcome mismatch'):replay(spec,forged)
            self.assertEqual(replay(spec,genuine),genuine)


if __name__=='__main__':unittest.main()
