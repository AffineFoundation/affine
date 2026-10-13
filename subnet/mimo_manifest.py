"""Build a fresh prospective MiMo epoch from explicit operator inputs.

There are no legacy K/L aliases, no miner-selected nonce offsets, and no implied
calibration. The caller signs only after admitting its actual runtime evidence.
"""
import copy
import secrets

from .batch_contract import for_manifest, REQUIRED_CLIENT_FEATURES
from .complete_group_distribution import MIMO_VERSION, MIMO_TRANSPORT
from .complete_group_admission import TRANSPORT
from .complete_group_contract import GROUP_VERSION, OUTCOME_POLICY
from .complete_group_harness import MIMO_VERSION as HARNESS_VERSION
from .storage import canonical, sha


def build(*, run_id, epoch, checkpoint, source_bundle, environments, runtime_spec, vocab_size,
          backend_profile, calibration, start, duration=1200, group_size=8,
          max_batches=9, max_output_tokens=8192, randomness=None, evaluation_sets=None):
    from .mimo_runtime import validate_runtime_spec
    from .fast_prefill_audit import bind
    from .harness import source_hash as harness_hash
    from .forced_sampling import source_hash as sampling_hash
    from .mimo_answer_projection import POLICY as answer_policy, POLICY_KEY
    spec = validate_runtime_spec(runtime_spec)
    if (not isinstance(run_id, str) or not run_id or not isinstance(epoch, str) or not epoch
            or type(start) is not int or type(duration) is not int or not 1 <= duration <= 3600):
        raise ValueError('explicit run, epoch and bounded collection window')
    harness = dict(version=HARNESS_VERSION, policy='autoregressive', max_output_tokens=max_output_tokens,
                   max_context_tokens=spec.max_context_tokens, temperature=spec.temperature,
                   top_p=spec.top_p, top_k=spec.top_k, enable_thinking=spec.enable_thinking)
    definitions = copy.deepcopy(environments)
    if not definitions:
        raise ValueError('pinned math task registry required')
    for row in definitions:
        if row['spec']['max_turns'] != 1 or row['spec']['max_output_tokens'] < max_output_tokens:
            raise ValueError('native taskset must permit the signed single-turn output horizon')
        row['harness'] = copy.deepcopy(harness)
    result = dict(run_id=run_id, epoch=epoch, payable=False,
        checkpoint=copy.deepcopy(checkpoint), source_bundle=copy.deepcopy(source_bundle),
        model_id=spec.model_id, model_runtime_revision=spec.version,
        mimo_runtime=spec.to_dict(), backend_profile=copy.deepcopy(backend_profile),
        runtime_profile={}, start=start, deadline=start+duration, max_batches=max_batches,
        environments=definitions, environment_revision='trusted-adapter-registry-v1',
        harness_source_hash=harness_hash(), sampling_source_hash=sampling_hash(),
        transport_policy='direct-r2-v1', submission_transport_policy=TRANSPORT,
        probability_artifact_policy={'version':'selected-token-logprobs-v1'},
        group_transport_policy=dict(version=MIMO_TRANSPORT, vocab_size=vocab_size,
            eos_token_ids=list(spec.eos_token_ids), max_context_tokens=spec.max_context_tokens),
        group_contract=dict(version=GROUP_VERSION, group_size=group_size, min_success=1,
            min_completed_failure=1, outcome_policy=OUTCOME_POLICY),
        required_client_features=sorted(REQUIRED_CLIENT_FEATURES),
        sampling_contract=dict(version=MIMO_VERSION, randomness=randomness or secrets.token_hex(32),
            verification='prefill-cdf-calibrated', generation='cached-eager-inverse-cdf',
            support_adjudication='exact-cached-replay-v1', calibration=copy.deepcopy(calibration)))
    result[POLICY_KEY] = copy.deepcopy(answer_policy)
    if evaluation_sets is not None:
        result['evaluation_sets'] = copy.deepcopy(evaluation_sets)
    for_manifest(result, supported_features=REQUIRED_CLIENT_FEATURES)
    bind(result, harness)
    from .complete_group_sampling import binding
    binding(result, None, expected_source_hash=sampling_hash())
    from .mimo_evaluation import validate_sets
    validate_sets(result)
    return result
