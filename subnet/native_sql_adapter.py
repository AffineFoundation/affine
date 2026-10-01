"""Prospective original Spider common adapter with operator-owned grading.

The public actor sees database/schema/questions only. The deployment injects
fresh actors and a private terminal grader; this controlled cohost interface
is not registered in the active dispatcher or an isolation boundary against
an operator with host access.
"""
import hashlib
import json
from .storage import canonical
from .native_sql_actor import REVISION as ACTOR_VERSION
from .native_sql_isolation import REVISION as GRADER_VERSION

VERSION='controlled-native-spider-common-v1'
PUBLIC_FIELDS={'revision','db_id','database_sha256','original_source_sha256','messages','tools'}

def task_hash(task):
    return hashlib.sha256(canonical(task)).hexdigest()

def validate_public(task):
    if not isinstance(task,dict) or set(task)!=PUBLIC_FIELDS or task['revision']!=ACTOR_VERSION:
        raise ValueError('SQL public descriptor fields/version')
    for key in ('database_sha256','original_source_sha256'):
        value=task[key]
        if not isinstance(value,str) or len(value)!=64 or any(c not in '0123456789abcdef' for c in value):raise ValueError('SQL public source hash')
    if not isinstance(task['messages'],list) or not task['messages'] or not isinstance(task['tools'],list):raise ValueError('SQL public prompt/tools')
    if len(task['tools'])!=1 or task['tools'][0].get('function',{}).get('name')!='bash':raise ValueError('SQL original bash harness')
    return task

class NativeSQLAdapter:
    def __init__(self,spec,actor_factory,terminal_grade):
        if spec.adapter!='native_sql_controlled' or spec.version!=VERSION:raise ValueError('SQL adapter/version')
        if spec.config.get('dependency_scope')!='controlled-public-database-private-original-grader':raise ValueError('SQL isolation scope')
        if not callable(actor_factory) or not callable(terminal_grade):raise ValueError('trusted SQL deployment factories')
        tasks=spec.config.get('public_tasks')
        if not isinstance(tasks,list) or not tasks or len(tasks)!=spec.num_samples:raise ValueError('SQL original task index contract')
        for task in tasks:validate_public(task)
        if len({task_hash(t) for t in tasks})!=len(tasks):raise ValueError('duplicate SQL public task')
        self.grader=spec.config.get('grader_runtime')
        if not isinstance(self.grader,dict) or self.grader.get('revision')!=GRADER_VERSION:raise ValueError('SQL approved original grader')
        if any(t['original_source_sha256']!=self.grader.get('original_source_sha256') for t in tasks):raise ValueError('SQL actor/grader original source')
        self.spec=spec;self.tasks=tasks;self.actor_factory=actor_factory;self.terminal_grade=terminal_grade;self.actor=None;self.done=False

    def reset(self,index,seed):
        if self.actor is not None or type(index) is not int or not 0<=index<len(self.tasks) or type(seed) is not int:raise ValueError('SQL reset binding')
        self.index=index;self.turns=0;self.done=False;self.public=self.tasks[index]
        self.actor=self.actor_factory(index,seed,task_hash(self.public))
        try:
            if canonical(validate_public(self.actor.start()))!=canonical(self.public):raise ValueError('SQL fresh task mismatch')
            return dict(messages=self.public['messages'],tools=self.public['tools'],task_hash=task_hash(dict(public_task=self.public,version=self.spec.version,index=index,seed=seed)))
        except Exception:self.close();raise

    def step(self,action):
        if self.actor is None or self.done:raise ValueError('SQL step after closure/uninitialized')
        if not isinstance(action,dict) or not isinstance(action.get('text',''),str):raise ValueError('SQL action schema')
        calls=action.get('tool_calls') or []
        if not isinstance(calls,list) or len(calls)>8:raise ValueError('SQL call budget')
        observations=[]
        try:
            for i,call in enumerate(calls):
                if not isinstance(call,dict) or call.get('name')!='bash' or not isinstance(call.get('arguments'),dict):raise ValueError('SQL original tool schema')
                value=self.actor.call('bash',call['arguments'])
                if not isinstance(value,dict) or set(value)!={'exit_code','stdout','stderr'} or type(value['exit_code']) is not int or any(not isinstance(value[k],str) for k in ('stdout','stderr')):raise ValueError('SQL original observation schema')
                # Preserve OriginalSession's JSON observation representation.
                observations.append(dict(role='tool',tool_call_id=call.get('id',f'call-{self.turns+1}-{i}'),name='bash',content=json.dumps(value)))
            self.turns+=1;self.done=not calls or self.turns>=self.spec.max_turns
            reward=0.0
            if self.done:
                result=self.terminal_grade(self.index,action.get('text',''),task_hash(self.public))
                if result.get('runtime')!=self.grader or result.get('database_sha256')!=self.public['database_sha256']:raise ValueError('SQL terminal task/grader binding')
                isolation=result.get('isolation',{})
                if isolation.get('network')!='none' or isolation.get('read_only') is not True or isolation.get('host_mounts')!=[] or isolation.get('user')!='65534:65534':raise ValueError('SQL private grader isolation')
                reward=result.get('reward')
                if type(reward) not in (int,float) or reward not in (0,1):raise ValueError('SQL original binary reward')
                reward=float(reward)
            return dict(observations=observations,done=self.done,reward=reward,classification='positive' if self.done and reward>=self.spec.success_reward else 'negative' if self.done else 'neutral')
        except Exception:self.done=True;self.close();raise

    def close(self):
        if self.actor is not None:self.actor.close();self.actor=None
