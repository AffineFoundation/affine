"""Eligibility-only original MATH context; never a rollout or grading adapter."""
import hashlib,json,stat
from pathlib import Path
from .environments import EnvironmentSpec,EnvironmentSession,_snapshot_path,_taskset
MAX_SNAPSHOT_BYTES=128*1024**2

def fingerprint(path):
    value=path.lstat()
    if not stat.S_ISREG(value.st_mode)or value.st_nlink!=1:raise ValueError('regular immutable MATH snapshot')
    return tuple(getattr(value,k)for k in ('st_dev','st_ino','st_uid','st_mode','st_size','st_mtime_ns','st_ctime_ns'))

class NativeMathPromptSession:
    def __init__(self,raw):
        self.spec=EnvironmentSpec.from_dict(raw)if isinstance(raw,dict)else raw
        if (self.spec.id!='affine_math'or self.spec.adapter!='prime_v1'or self.spec.max_turns!=1 or
            not self.spec.config.get('task_snapshot')):
            raise ValueError('prompt-only eligibility requires original one-turn snapshot MATH')
        self.path=_snapshot_path(self.spec.config);before=fingerprint(self.path)
        if not 0<before[4]<=MAX_SNAPSHOT_BYTES:raise ValueError('bounded MATH prompt snapshot')
        data=self.path.read_bytes()
        if fingerprint(self.path)!=before:raise ValueError('MATH snapshot changed during read')
        # Original session admission authenticates complete environment source,
        # dependency closure and snapshot bytes. No reset/setup/runtime is run.
        admitted=EnvironmentSession(self.spec)
        try:
            if hashlib.sha256(data).hexdigest()!=hashlib.sha256(self.path.read_bytes()).hexdigest()or fingerprint(self.path)!=before:
                raise ValueError('MATH snapshot authentication race')
            self.taskset=_taskset(self.spec)
            self.task_cls=self.taskset.task_type();self.config_cls=self.task_cls.config_type()
            if self.task_cls.__name__!='MathTask'or self.task_cls.__module__!='affine_math_v1.taskset':
                raise ValueError('original native MathTask only')
            self.rows=json.loads(data)
            if not isinstance(self.rows,list)or len(self.rows)!=self.spec.num_samples:raise ValueError('snapshot task count')
        finally:admitted.close()
        self.snapshot_stat=before;self.closed=False
    def reset(self,index,seed):
        if self.closed:raise ValueError('closed prompt eligibility session')
        if type(index)is not int or not 0<=index<self.spec.num_samples:raise ValueError('environment index')
        if fingerprint(self.path)!=self.snapshot_stat:raise ValueError('authenticated MATH snapshot changed')
        row=self.rows[index]
        if row['task_class']!=self.task_cls.__name__:raise ValueError('original snapshot task class')
        task=self.task_cls(self.task_cls.data_type()(**row['data']),self.config_cls(**row['task_config']))
        # Match EnvironmentSession._prepare initial messages exactly. The
        # original MathTask.setup only checks grader health; it does not supply
        # prompt data. Eligibility does not claim grader health or an outcome.
        if task.toolsets(task.config)or self.taskset.toolsets(self.taskset.config):raise ValueError('prompt-only MATH requires no task tools')
        messages=[]
        if task.data.system_prompt:messages.append(dict(role='system',content=task.data.system_prompt))
        prompt=task.data.prompt
        if isinstance(prompt,str):messages.append(dict(role='user',content=prompt))
        elif prompt:messages.extend(m.model_dump(mode='json',exclude_none=True)for m in prompt)
        return dict(messages=messages,tools=[],task_hash=task.hash,task_name=task.data.name)
    def close(self):
        self.closed=True;self.rows=[];self.taskset=None
