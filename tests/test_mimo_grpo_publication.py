"""Synthetic S3 streams exercise real publication, signatures and handover.

No provider or production endpoint is contacted. Multipart bytes are fixture
payloads; trainer handover cases use actual small PyTorch optimizer updates.
"""
from dataclasses import replace
import io
import json
import threading
from pathlib import Path
from types import SimpleNamespace

from botocore.exceptions import ClientError
import pytest

from subnet.storage import Identity
from subnet.mimo_grpo import TrainerConfig, execute_transaction
from subnet.mimo_grpo.data import digest
from subnet.mimo_grpo.optimizer import CPUAdamW
from subnet.mimo_grpo.publication import R2ModelPublisher, validate_publication
from subnet.mimo_grpo.handover import initialize_private_current, private_current_callback
from subnet.mimo_runtime import file_hash
from test_mimo_grpo import Runtime, groups, binding


def error(code):
    return ClientError({'Error': {'Code': code},
                        'ResponseMetadata': {'HTTPStatusCode': 412 if code == 'PreconditionFailed' else 404}}, 'fixture')


class Body(io.BytesIO):
    def __init__(self, data, client):
        super().__init__(data)
        self.client = client
    def read(self, size=-1):
        assert 0 < size <= 4 * 1024**2 + 1, 'all readbacks must be bounded'
        self.client.read_sizes.append(size)
        return super().read(size)
    def close(self):
        self.client.closed_bodies += 1
        super().close()


class S3:
    def __init__(self):
        self.objects = {}
        self.multipart = {}
        self.uploads = []
        self.read_sizes = []
        self.closed_bodies = 0
        self.lost_complete_ack = False
        self.next_id = 0
    def get_object(self, *, Bucket, Key):
        if Key not in self.objects:
            raise error('NoSuchKey')
        return dict(ContentLength=len(self.objects[Key]), Body=Body(self.objects[Key], self))
    def put_object(self, *, Bucket, Key, Body, IfNoneMatch, **kwargs):
        assert IfNoneMatch == '*'
        if Key in self.objects:
            raise error('PreconditionFailed')
        if isinstance(Body, bytes):
            data = Body
        else:
            data = b''.join(iter(lambda: Body.read(128 * 1024), b''))
        self.objects[Key] = data
        self.uploads.append(Key)
        return {'ETag': 'fixture'}
    def create_multipart_upload(self, **kwargs):
        self.next_id += 1
        uid = str(self.next_id)
        self.multipart[uid] = {}
        return {'UploadId': uid}
    def upload_part(self, *, UploadId, PartNumber, Body, ContentLength, **kwargs):
        assert len(Body) == ContentLength <= 5 * 1024**2
        self.multipart[UploadId][PartNumber] = Body
        return {'ETag': str(PartNumber)}
    def complete_multipart_upload(self, *, Key, UploadId, MultipartUpload, IfNoneMatch, **kwargs):
        assert IfNoneMatch == '*'
        if Key in self.objects:
            raise error('PreconditionFailed')
        self.objects[Key] = b''.join(self.multipart[UploadId][row['PartNumber']] for row in MultipartUpload['Parts'])
        del self.multipart[UploadId]
        self.uploads.append(Key)
        if self.lost_complete_ack:
            self.lost_complete_ack = False
            raise TimeoutError('complete persisted; reply lost')
        return {'ETag': 'completed'}
    def abort_multipart_upload(self, *, UploadId, **kwargs):
        if UploadId not in self.multipart:
            raise error('NoSuchUpload')
        del self.multipart[UploadId]


def publisher(client, *, threshold=64 * 1024**2):
    return R2ModelPublisher(SimpleNamespace(client=client, name='private-fixture'),
        Identity(bytes([53]) * 32), namespace='private/mimo-qualification',
        source_closure_sha256='d' * 64, assert_trainer_lease=lambda: None,
        multipart_threshold=threshold, part_bytes=5 * 1024**2, read_chunk_bytes=16384)


def request(root):
    root.mkdir()
    (root/'config.json').write_text('{}')
    (root/'model.safetensors').write_bytes(b'synthetic-stream-fixture' * 300000)
    files = {p.name: dict(size=p.stat().st_size, sha256=file_hash(p)) for p in root.iterdir()}
    checkpoint = digest({name: row['sha256'] for name, row in files.items()})
    return dict(job_id='private-fixture-job', epoch='private-fixture-epoch',
        input_checkpoint='a' * 64, output_checkpoint=checkpoint, report_sha256='b' * 64,
        optimizer_step=1, model_directory=str(root), model_files=files,
        optimizer_state_uploaded=False, optimizer_state_location='trainer-local', authority_committed=False)


def test_multipart_lost_ack_retry_full_streamed_readback_never_overwrites(tmp_path):
    client = S3()
    publish = publisher(client, threshold=1024)
    original = request(tmp_path/'model')
    client.lost_complete_ack = True
    with pytest.raises(TimeoutError):
        publish(original)
    assert not client.multipart
    result = publish(original)
    uploads = list(client.uploads)
    assert publish(original) == result
    assert client.uploads == uploads
    assert result['optimizer_state_uploaded'] is False
    assert all('/optimizer/' not in name for name in client.objects)
    assert client.closed_bodies > 3
    assert publish.receipt(original['job_id']) == publish.last_receipt


def test_existing_wrong_model_bytes_are_never_overwritten(tmp_path):
    client = S3()
    publish = publisher(client)
    original = request(tmp_path/'model')
    key = publish.namespace + '/checkpoints/' + original['output_checkpoint'] + '/model/model.safetensors'
    client.objects[key] = b'corrupt-object'
    with pytest.raises(ValueError, match='length differs'):
        publish(original)
    assert client.objects[key] == b'corrupt-object'
    assert not any('/publications/' in name for name in client.objects)


def test_independent_shards_overlap_and_receipt_waits_for_every_readback(tmp_path):
    class ConcurrentS3(S3):
        def __init__(self):
            super().__init__();self.barrier=threading.Barrier(4);self.lock=threading.Lock()
            self.active=0;self.peak=0;self.completed=set()
        def put_object(self,**kwargs):
            if kwargs['Key'].endswith('.safetensors'):
                with self.lock:self.active+=1;self.peak=max(self.peak,self.active)
                self.barrier.wait(timeout=5)
                result=super().put_object(**kwargs)
                with self.lock:self.active-=1;self.completed.add(kwargs['Key'])
                return result
            if '/publications/'in kwargs['Key']:
                assert len(self.completed)==4
            return super().put_object(**kwargs)
    client=ConcurrentS3();publish=publisher(client)
    original=request(tmp_path/'model');root=Path(original['model_directory'])
    (root/'model.safetensors').unlink()
    for n in range(4):(root/f'model-{n}.safetensors').write_bytes(bytes([n])*1000)
    files={p.name:dict(size=p.stat().st_size,sha256=file_hash(p))for p in root.iterdir()}
    original.update(model_files=files,output_checkpoint=digest({n:r['sha256']for n,r in files.items()}))
    result=publish(original)
    assert client.peak==4 and result['model_full_get_verified']is True
    assert len(publish.last_receipt['payload']['objects'])==5
    assert client.closed_bodies>=7


def test_two_private_epochs_signed_r2_handover_and_checkpoint_lifetime_retention(tmp_path):
    runtime = Runtime()
    optimizer = CPUAdamW(runtime.model.named_parameters(), TrainerConfig(), input_checkpoint=runtime.checkpoint_id)
    client = S3()
    publish = publisher(client)
    initial = replace(binding(), epoch='private-epoch-1')
    current_path = tmp_path/'CURRENT.json'
    initialize_private_current(current_path, initial, source_closure_sha256='d' * 64)
    commit = private_current_callback(current_path, publish, run=initial.run, assert_trainer_lease=lambda: None)
    def step(name, pin):
        return execute_transaction(tmp_path/'history'/name, runtime, optimizer, groups(), pin,
            manifest_sha256='c' * 64, source_closure_sha256='d' * 64, assert_trainer_lease=lambda: None,
            publish_model=publish, commit_current=commit, retire_published_models=True, shard_bytes=4096)
    first = step('one', initial)
    receipt1 = publish.last_receipt
    validate_publication(receipt1, authority=publish.signer.id, transaction=first, source_closure_sha256='d' * 64)
    assert json.loads(current_path.read_text())['checkpoint'] == first['output_checkpoint']
    # Acknowledged retries reuse the same publication and CURRENT.
    assert step('one', initial) == first and optimizer.global_step == 1
    second = step('two', replace(initial, epoch='private-epoch-2',
                                 checkpoint=first['output_checkpoint'], optimizer_step=1))
    assert optimizer.global_step == 2
    assert json.loads(current_path.read_text())['checkpoint'] == second['output_checkpoint']
    assert not (tmp_path/'history/one/model').exists()
    assert not (tmp_path/'history/one/optimizer').exists()
    assert (tmp_path/'history/one/TRANSACTION.json').exists()
    assert (tmp_path/'history/two/model').exists()
    assert (tmp_path/'history/two/optimizer').exists()
    assert any(first['output_checkpoint'] in key for key in client.objects)
    with pytest.raises(ValueError, match='original completed update'):
        validate_publication(receipt1, authority=publish.signer.id, transaction=second,
                             source_closure_sha256='d' * 64)
