"""Prospective positive-only objective identity, separate from preference+NLL.

Parsing this configuration does not authorize a production execution. Signed
execution routing and its independent qualification still have to accept it.
The miner's positive/negative submission quotas are not changed here.
"""

VERSION = 'task-balanced-positive-sft-v1'
COMPLETION_POLICY = 'terminal-eos-v1'
TASK_WEIGHT_RULE = 'mean-positive-within-task-then-mean-task-within-group'


def configuration(value):
    expected = dict(version=VERSION, completion_policy=COMPLETION_POLICY,
                    task_weight_rule=TASK_WEIGHT_RULE)
    if type(value) is not dict or value != expected:
        raise ValueError('exact explicit task-balanced positive-only SFT objective')
    return dict(expected)
