import asyncio
import unittest
from types import SimpleNamespace
from pydantic import BaseModel,TypeAdapter
from subnet.native_tool_state import bind_state_channel

class State(BaseModel):
    verifier_results:list=[]

class NativeStateBridge(unittest.TestCase):
    def test_setup_time_mutation_and_later_tool_mutation_reach_original_grader(self):
        async def run():
            trace=SimpleNamespace(state=State());tool=SimpleNamespace(_state_cls=State,_state_adapter=TypeAdapter(State))
            audit=bind_state_channel(tool,trace)
            tool.state=await tool._pull_state();before=tool._state_adapter.dump_json(tool.state)
            tool.state.verifier_results=[dict(passed=False)];await tool._push_state(before)
            self.assertEqual(trace.state.verifier_results,[dict(passed=False)])
            tool.state=await tool._pull_state();before=tool._state_adapter.dump_json(tool.state)
            tool.state.verifier_results[0]['passed']=True;await tool._push_state(before)
            self.assertTrue(all(r['passed'] for r in trace.state.verifier_results));self.assertEqual(len(audit),2)
        asyncio.run(run())
    def test_incompatible_native_state_fails_closed(self):
        with self.assertRaisesRegex(ValueError,'state class mismatch'):
            bind_state_channel(SimpleNamespace(_state_cls=State),SimpleNamespace(state=object()))

if __name__=='__main__':unittest.main()
