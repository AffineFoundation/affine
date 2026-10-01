"""Prospective common reset/step/replay contract for controlled Agent images.

Only operator-signed specs may supply fixture descriptors. This module is not
yet enabled by the production registry; v1 proof pilots remain historical.
"""
import hashlib
import json
from .native_agent_isolation import NativeAgentSession, canonical, validate_descriptor

VERSION='controlled-native-agent-common-v2'

class NativeAgentAdapter:
    def __init__(self, spec, session_factory=NativeAgentSession):
        self.spec=spec;self.session_factory=session_factory;self.session=None
        if spec.version!=VERSION or spec.adapter!='native_agent_controlled':raise ValueError('native adapter version')
        if spec.config.get('dependency_scope')!='immutable-controlled-images-not-full-upstream-closure':raise ValueError('native dependency scope')
        tasks=spec.config.get('native_tasks')
        if not isinstance(tasks,list) or len(tasks)!=spec.num_samples or spec.num_samples<1:raise ValueError('native task index contract')
        identities=[]
        for task in tasks:
            if set(task)!={'descriptor','instruction'}:raise ValueError('native task fields')
            descriptor=validate_descriptor(task['descriptor'])
            if hashlib.sha256(task['instruction'].encode()).hexdigest()!=descriptor['public_files']['instruction.md']:raise ValueError('native task instruction pin')
            identities.append(hashlib.sha256(canonical(task)).hexdigest())
        if len(set(identities))!=len(identities):raise ValueError('duplicate native task fixtures')
        self.tasks=tasks

    def reset(self,index,seed):
        if self.session is not None or type(index) is not int or not 0<=index<len(self.tasks) or type(seed) is not int:raise ValueError('native reset binding')
        self.index=index;self.turns=0;self.done=False;task=self.tasks[index]
        self.session=self.session_factory(task['descriptor'],task['instruction']);initial=self.session.start()
        # Agent fixture reset is deterministic. Seed is included in the signed
        # task identity; it does not pretend to seed arbitrary provider code.
        initial['task_hash']=hashlib.sha256(canonical(dict(task=task,version=self.spec.version,index=index,seed=seed))).hexdigest()
        return initial

    def step(self,action):
        if self.session is None or self.done:raise ValueError('native step after closure/uninitialized')
        if not isinstance(action,dict) or not isinstance(action.get('text',''),str):raise ValueError('native action schema')
        calls=action.get('tool_calls') or []
        if not isinstance(calls,list) or len(calls)>8:raise ValueError('native call budget')
        observations=[]
        for i,call in enumerate(calls):
            if not isinstance(call,dict) or not isinstance(call.get('name'),str) or not isinstance(call.get('arguments'),dict):raise ValueError('native call schema')
            value=self.session.call(call['name'],call['arguments'])
            content=value if isinstance(value,str) else json.dumps(value,default=str)
            observations.append(dict(role='tool',tool_call_id=call.get('id','native-'+str(self.turns)+'-'+str(i)),name=call['name'],content=content))
        self.turns+=1;self.done=not calls or self.turns>=self.spec.max_turns
        reward=float(self.session.grade()['grade']['reward']) if self.done else 0.0
        return dict(observations=observations,done=self.done,reward=reward,classification='positive' if self.done and reward>=self.spec.success_reward else 'negative' if self.done else 'neutral')

    def close(self):
        if self.session is not None:self.session.close();self.session=None
