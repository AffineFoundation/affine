"""Prospective common EOG contract using public-only, scoped broker actors.

Not registered by the active environment dispatcher. The trusted deployment
supplies fresh actor sessions; private seed and grader fixtures never belong
in the signed public environment spec or model client.
"""
import math
from .storage import canonical
from .native_eog_split import validate_public, sha

VERSION='controlled-native-eog-broker-common-v1'

class NativeEOGAdapter:
    def __init__(self,spec,actor_factory):
        if spec.adapter!='native_eog_broker' or spec.version!=VERSION:
            raise ValueError('EOG common adapter/version')
        if spec.config.get('dependency_scope')!='controlled-cohost-public-actor-private-grader':
            raise ValueError('EOG broker isolation scope')
        if not callable(actor_factory):raise ValueError('trusted fresh actor factory required')
        self.tasks=spec.config.get('public_tasks')
        if not isinstance(self.tasks,list) or len(self.tasks)!=spec.num_samples or not self.tasks:
            raise ValueError('EOG public task index contract')
        for task in self.tasks:validate_public(task)
        if len({sha(task) for task in self.tasks})!=len(self.tasks):raise ValueError('duplicate EOG public tasks')
        self.spec=spec;self.actor_factory=actor_factory;self.actor=None;self.done=False

    def reset(self,index,seed):
        if self.actor is not None or type(index) is not int or not 0<=index<len(self.tasks) or type(seed) is not int:
            raise ValueError('EOG reset task binding')
        expected=self.tasks[index];self.actor=self.actor_factory(index,seed,sha(expected))
        initial=validate_public(self.actor.reset())
        if canonical(initial)!=canonical(expected):raise ValueError('EOG approved task mismatch')
        self.public=initial;self.events=[];self.turns=0;self.calls=0;self.done=False
        return dict(messages=initial['messages'],tools=initial['tools'],
                    task_hash=sha(dict(public_task=initial,version=self.spec.version,index=index,seed=seed)))

    def step(self,action):
        from mcp.server.fastmcp.exceptions import ToolError
        if self.actor is None or self.done:raise ValueError('EOG step after closure/uninitialized')
        if not isinstance(action,dict) or not isinstance(action.get('text',''),str):raise ValueError('EOG action schema')
        calls=action.get('tool_calls') or []
        if not isinstance(calls,list) or len(calls)>8 or self.calls+len(calls)>32:raise ValueError('EOG call budget')
        observations=[]
        for call in calls:
            if not isinstance(call,dict) or not isinstance(call.get('name'),str) or not isinstance(call.get('arguments'),dict):
                raise ValueError('EOG tool call schema')
            name,args=call['name'],call['arguments'];self.calls+=1
            try:
                value=self.actor.call(name,args)
            except ToolError as error:
                # Exact original MCP error observation; transport failures are
                # not converted into valid environment outcomes.
                value=str(error)
            else:
                self.events.append(dict(name=name,arguments=args,observation=value))
            if not isinstance(value,str):raise ValueError('EOG native observation wire type')
            observations.append(dict(role='tool',name=name,tool_call_id=call.get('id','eog-'+str(self.calls)),content=value))
        self.turns+=1;self.done=not calls or self.turns>=self.spec.max_turns
        reward=0.
        if self.done:
            terminal=self.actor.finish()
            if terminal.get('sealed') is not True or terminal.get('task_id')!=self.public['task_id'] or terminal.get('public_descriptor_sha256')!=sha(self.public) or terminal.get('transcript_sha256')!=sha(self.events):
                raise ValueError('EOG terminal native history binding')
            reward=terminal.get('reward')
            if type(reward) not in (int,float) or not math.isfinite(reward):raise ValueError('EOG native reward type/finiteness')
            reward=float(reward)
        return dict(observations=observations,done=self.done,reward=reward,
                    classification='positive' if self.done and reward>=self.spec.success_reward else 'negative' if self.done else 'neutral')

    def close(self):
        if self.actor is not None:self.actor.close();self.actor=None
