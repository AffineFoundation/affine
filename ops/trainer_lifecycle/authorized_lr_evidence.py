"""CPU evidence interpretation for an explicitly ROOT-authorized effective LR.

The original attribution validator's bytecode and every other hyperparameter
are retained. No shared globals, submitted reports, jobs or model are mutated.
"""
import types
VERSIONS={
 'unaudited-training-execution-amendment-v2-effective-lr':'fp32-task-gradient-effective-lr-v1',
 'unaudited-training-execution-amendment-v3-effective-lr-genesis':'fp32-task-gradient-effective-lr-genesis-v1',
}

def expected_hyperparameters(report,job,manifest,authority):
    from subnet.training_receipts import authenticate
    from subnet.persistent_cpu_adamw import HYPERPARAMETERS,sha
    from subnet.learning_rate_transition import validate_authorization,effective_hyperparameters
    from subnet.persistent_training_protocol import validate_output
    document=job.get('unaudited_training_execution')
    if document is None:return None
    declaration=authenticate(document,authority)
    if declaration.get('version')not in VERSIONS:return None
    binding=manifest['trainer_state_binding'];parent=binding.get('parent')
    parent_sha=parent['descriptor_sha256']if parent else None
    initial=declaration['version']=='unaudited-training-execution-amendment-v3-effective-lr-genesis'
    if (authenticate(job['manifest'],authority)!=manifest or
            declaration.get('method')!=VERSIONS[declaration['version']] or
            declaration.get('job_id')!=job['job_id'] or declaration.get('epoch')!=manifest['epoch'] or
            declaration.get('original_signed_manifest_sha256')!=sha(job['manifest']) or
            declaration.get('trainer_binding_sha256')!=sha(binding) or
            declaration.get('execution_source_files')!=job['source_files'] or
            declaration.get('genesis_sha256')!=binding['genesis_sha256'] or
            declaration.get('parent_descriptor_sha256')!=parent_sha or
            declaration.get('optimizer_step_before')!=binding['global_step_before'] or
            declaration.get('steps')!=job['steps'] or initial!=(parent is None)):
        raise ValueError('effective LR evidence requires exact original ROOT job scope')
    value=validate_authorization(declaration['learning_rate_authorization'],authority,
        epoch=manifest['epoch'],job_id=job['job_id'],input_checkpoint=binding['input_checkpoint'],
        parent_descriptor_sha256=parent_sha,genesis_sha256=binding['genesis_sha256'],
        optimizer_step_before=binding['global_step_before'],steps=job['steps'],
        parameters_sha256=binding['parameters_sha256'],historical=True)
    if (value['execution_release_sha256']!=declaration['execution_release_sha256'] or
            value['effective_learning_rate']!=declaration['effective_learning_rate']):
        raise ValueError('effective LR evidence exact original release and rate')
    validate_output(report['persistent_training_state']['descriptor'],job,manifest)
    expected=effective_hyperparameters(value)
    if {k:v for k,v in expected.items()if k!='lr'}!={k:v for k,v in HYPERPARAMETERS.items()if k!='lr'}:
        raise ValueError('learning-rate-only evidence interpretation')
    diagnostics=report['training']['persistent_diagnostics']
    if diagnostics.get('effective_hyperparameters')!=expected or diagnostics.get('learning_rate_authorization_sha256')!=sha(declaration['learning_rate_authorization']):
        raise ValueError('diagnostics must report exact authorized effective hyperparameters')
    return expected

def install(authority):
    from subnet import persistent_training_evidence as evidence
    original=evidence.validate_updates
    def validate_updates(report,job,manifest):
        expected=expected_hyperparameters(report,job,manifest,authority)
        if expected is None:return original(report,job,manifest)
        namespace=dict(original.__globals__,HYPERPARAMETERS=expected)
        scoped=types.FunctionType(original.__code__,namespace,original.__name__,original.__defaults__,original.__closure__)
        scoped.__kwdefaults__=original.__kwdefaults__
        return scoped(report,job,manifest)
    evidence.validate_updates=validate_updates
    return original
