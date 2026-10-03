"""Paired outcome statistics after original job/report authentication.

This reducer authenticates no execution. Its caller must check the original
signed jobs, checkpoint/source/runtime bindings and complete native audits.
It refuses missing/error rows instead of silently changing a fixed cohort.
"""
import math,re


def wilson(successes,count):
    # Keep the statistics-only reducer usable without importing model runtimes.
    z=1.96;p=successes/count;denominator=1+z*z/count
    center=(p+z*z/(2*count))/denominator
    half=z*math.sqrt(p*(1-p)/count+z*z/(4*count*count))/denominator
    return [max(0.,center-half),min(1.,center+half)]


def summarize(cohort, baseline, learned):
    indices=cohort['indices'];seeds=cohort['seeds'];count=len(indices)
    if (not 1<=count<=4096 or len(seeds)!=count or len(set(indices))!=count
            or any(type(v)is not int or v<0 for v in indices+seeds)):
        raise ValueError('fixed paired cohort')
    expected=dict(zip(indices,seeds))

    def checked(records):
        if len(records)!=count:raise ValueError('complete paired population required')
        result={}
        for row in records:
            index=row.get('index');seed=row.get('seed');reward=row.get('reward')
            classification=row.get('classification');task=row.get('task_hash')
            if (type(index)is not int or index in result or index not in expected
                    or type(seed)is not int or seed!=expected[index]
                    or row.get('env_id')!='affine_math' or row.get('verified') is not True
                    or 'error' in row or 'error_type' in row
                    or not isinstance(task,str) or re.fullmatch('[0-9a-f]{64}',task)is None
                    or classification not in ('positive','negative')
                    or type(reward) not in (int,float) or not math.isfinite(reward)
                    or reward!=(1 if classification=='positive' else 0)):
                raise ValueError('paired verified index/seed/task/outcome')
            result[index]=row
        return result

    before,after=checked(baseline),checked(learned);pairs=[]
    for index in indices:
        b,a=before[index],after[index]
        if b['task_hash']!=a['task_hash']:raise ValueError('paired task identity changed')
        pairs.append(dict(index=index,seed=expected[index],task_hash=b['task_hash'],
            baseline_success=b['classification']=='positive',
            learned_success=a['classification']=='positive'))
    gains=sum(not p['baseline_success'] and p['learned_success'] for p in pairs)
    losses=sum(p['baseline_success'] and not p['learned_success'] for p in pairs)
    discordant=gains+losses
    probability=(min(1.,2*sum(math.comb(discordant,k)
        for k in range(min(gains,losses)+1))/2**discordant) if discordant else 1.)
    base_correct=sum(p['baseline_success'] for p in pairs)
    learned_correct=sum(p['learned_success'] for p in pairs)
    return dict(count=count,baseline_correct=base_correct,learned_correct=learned_correct,
        baseline_accuracy=base_correct/count,learned_accuracy=learned_correct/count,
        baseline_accuracy_interval=wilson(base_correct,count),
        learned_accuracy_interval=wilson(learned_correct,count),
        accuracy_change=(learned_correct-base_correct)/count,
        paired_gains=gains,paired_losses=losses,
        paired_exact_two_sided_p=probability,pairs=pairs,
        cohort_completeness_checked=True,execution_authenticated_here=False,
        stable_long_term_improvement_proven=False)
