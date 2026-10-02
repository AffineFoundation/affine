"""Actual trusted source conformance; no model or blockchain shortcuts."""
from dataclasses import replace
import unittest
from subnet.environments import build_spec, create_session, legacy_spec, source_inventory

class EnvironmentTests(unittest.TestCase):
    def ifeval_spec(self):
        from pathlib import Path
        # Bind public tasks to the code under test; archived runtime specs stay
        # immutable and are not suitable for a new adapter-source revision.
        snapshot=Path(__file__).parent/'fixtures/ifeval_public_constraint_tasks.json'
        return build_spec('affine_ifeval',{'task_snapshot':str(snapshot.resolve())},
                          num_samples=4,max_turns=1,max_output_tokens=128)
    def test_active_inventory(self):
        inventory=source_inventory()
        self.assertEqual(sum(v['active'] for v in inventory.values()),45)
        self.assertFalse(inventory['affine_gdpval']['active'])
        self.assertEqual(inventory['swerebench_v2']['module'],'swerebench_v2_v1')

    def test_source_tamper(self):
        spec=legacy_spec()
        with self.assertRaises(ValueError):create_session(replace(spec,source_hash='0'*64))

    def test_actual_verbatim_positive_negative(self):
        spec=build_spec('affine_verbatim',{'taskset':{'num_samples':2,'target_length':8,'content_type':'codes'}},num_samples=2,max_turns=1)
        first=create_session(spec)
        try:
            reset=first.reset(0,0)
            public=reset['messages'][-1]['content'].split('<text>')[-1].split('</text>')[0]
            result=first.step({'text':'<answer>'+public+'</answer>'})
            self.assertEqual(result['reward'],1)
            self.assertEqual(result['classification'],'positive')
        finally:first.close()
        second=create_session(spec)
        try:
            self.assertEqual(second.reset(0,0)['task_hash'],reset['task_hash'])
            self.assertEqual(second.step({'text':'<answer>!</answer>'})['classification'],'negative')
        finally:second.close()

    def test_actual_reasoning_gym(self):
        spec=build_spec('affine_rgym',{'taskset':{'generators':['count_bits'],'per_generator':2,'curriculum_level':1}},num_samples=2,max_turns=1)
        session=create_session(spec)
        try:
            reset=session.reset(0,0)
            number=int(reset['messages'][-1]['content'].split('number ')[1].rstrip('?'))
            result=session.step({'text':'<answer>'+str(number.bit_count())+'</answer>'})
            self.assertEqual(result['reward'],1)
        finally:session.close()

    def test_original_when2call_real_tool_reward(self):
        spec=build_spec('affine_when2call',{},num_samples=2,max_turns=4)
        session=create_session(spec)
        try:
            reset=session.reset(0,0)
            self.assertEqual(reset['task_name'],'w2c-b052e916850b')
            self.assertEqual(reset['tools'][0]['function']['name'],'get_ico_calendar')
            # Literal original task answer is a conformance oracle, never miner input.
            call={'id':'one','name':'get_ico_calendar','arguments':{'category':'_ico_cat_ecomm,_ico_cat_finance','time_utc_offset':28800,'tabname':'completed','sort':'funds_raised'}}
            result=session.step({'text':'','tool_calls':[call]})
            self.assertFalse(result['done'])
            self.assertEqual(result['observations'][0]['role'],'tool')
            result=session.step({'text':'Here are the completed ICOs.'})
            self.assertEqual(result['reward'],1)
        finally:session.close()

    def test_original_ifeval_snapshot_public_constraint(self):
        import json
        from pathlib import Path
        from subnet.environments import EnvironmentSpec
        spec=self.ifeval_spec()
        for text,reward in [('ipv6 expands the address space and routes internet traffic, allowing more devices to connect.',1),('IPv6 expands the address space and routes internet traffic, allowing more devices to connect.',0)]:
            session=create_session(spec)
            try:
                reset=session.reset(0,0)
                self.assertEqual(reset['task_name'],'ifeval-10fcca483882')
                self.assertEqual(session.step({'text':text})['reward'],reward)
            finally:session.close()

    def test_snapshot_rejects_untrusted_task_class(self):
        import json,tempfile
        from pathlib import Path
        from subnet.environments import EnvironmentSpec,_source_hash
        original=self.ifeval_spec()
        rows=json.loads(Path(original.config['task_snapshot']).read_text())
        rows[0]['task_class']='ArbitraryUploadedClass'
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'tasks.json';path.write_text(json.dumps(rows))
            spec=replace(original,config=dict(original.config,task_snapshot=str(path)))
            spec=replace(spec,source_hash=_source_hash(spec))
            session=create_session(spec)
            try:
                with self.assertRaisesRegex(RuntimeError,"task-class adapter"):session.reset(0,0)
            finally:session.close()

if __name__=='__main__':unittest.main()
