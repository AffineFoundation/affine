"""Cold-process restoration of an original staged/published model update."""
import json

from .config import TrainerConfig
from .data import digest
from .optimizer import CPUAdamW
from .transaction import (Job, Transaction, RecoveryRequired, _checkpoint,
                          _inventory, _validate_report, _receipt)


def restore_transaction_state(workspace, *, expected_job, load_runtime,
                              config=TrainerConfig(), assert_trainer_lease=lambda: None):
    """Restore original exported model and optimizer; never run training.

    expected_job comes from the authenticated original dispatch, not an
    untrusted replacement journal. load_runtime(directory, descriptor) loads
    the actual output safetensors in trainable mode under the pinned runtime.
    The caller holds the sole trainer lease across restoration and subsequent
    execute_transaction. Both staged (lost ACK) and published states work.
    """
    config.validate()
    if not isinstance(expected_job, Job):
        raise ValueError('authenticated original Job required for cold recovery')
    assert_trainer_lease()
    transaction = Transaction(workspace, expected_job)
    with transaction.exclusive():
        value = transaction._read()
        if value is None or value['state'] not in ('staged', 'published'):
            raise RecoveryRequired('complete original staging required for cold recovery')
        if value.get('report_sha256') != digest(value.get('report')):
            raise ValueError('original recovery report integrity')
        _validate_report(expected_job, value['report'])
        if value['report']['config'] != config.document():
            raise ValueError('original recovery optimizer/objective settings')
        if _inventory(transaction.root, list(value['artifacts'])) != value['artifacts']:
            raise ValueError('original staged model and optimizer content required')
        checkpoint, models = _checkpoint(value['artifacts'])
        if value['state'] == 'published':
            if value.get('output_checkpoint') != checkpoint:
                raise ValueError('published recovery output binding')
            _receipt(expected_job, checkpoint, value.get('publication'))
        descriptor = json.loads((transaction.root/'optimizer/descriptor.json').read_bytes())
        if descriptor['optimizer_step'] != expected_job.optimizer_step_before + 1:
            raise ValueError('original optimizer generation counter')
        assert_trainer_lease()
        runtime = load_runtime(transaction.root/'model',
            dict(id=checkpoint, files={name: row['sha256'] for name, row in models.items()}))
        try:
            if runtime.checkpoint_id != checkpoint:
                raise ValueError('cold recovery must load actual original output model')
            optimizer = CPUAdamW(runtime.model.named_parameters(), config, input_checkpoint=checkpoint)
            optimizer.restore_local(transaction.root/'optimizer',
                descriptor_sha256=digest(descriptor), checkpoint=checkpoint)
            assert_trainer_lease()
            return runtime, optimizer, value
        except BaseException:
            close = getattr(runtime, 'close', None)
            if callable(close):
                close()
            raise
