"""Test fixture for R2 transfers: a real S3 server on this machine, and a box that is this machine.

``R2Server`` runs the moto S3 server on a local port. The host code talks to it with its real client
(``subnet.storage.Bucket``), and the box tool talks to it over HTTP with signed URLs, as it does with R2.
``ShellBox`` stands for the ssh handle of a rented box: it runs each box command in a local shell.

``R2Server.calls`` lists every S3 request that a client of this process made (the host side). The box has
no client: it only has the signed URLs, so its part uploads are not in this list.
"""
from __future__ import annotations

import hashlib
import json
import logging
import subprocess
import urllib.request
from pathlib import Path
from typing import Any, Dict, List

import boto3
from moto.server import ThreadedMotoServer

BUCKET = 'affine-pit-r2-fixture'
MIB = 1024 * 1024


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class ShellBox:
    """The ssh handle of a box, with the box on this machine."""

    id = 'box-local'
    ip = '127.0.0.1'
    port = 22

    def __init__(self) -> None:
        self.commands: List[str] = []
        self.after: Dict[str, Any] = {}      # {text in a command: callable to run when that command has ended}

    def run(self, cmd, inp=None, timeout=600, check=True, **_kwargs):
        self.commands.append(cmd)
        proc = subprocess.run(['bash', '-c', cmd], input=inp, capture_output=True, timeout=timeout)
        out = proc.stdout.decode(errors='replace')
        if check and proc.returncode != 0:
            raise RuntimeError('box command failed: ' + proc.stderr.decode(errors='replace')[-400:])
        for text, hook in self.after.items():
            if text in cmd:
                hook()
        return out


class R2Server:
    """A local S3 server with one bucket, and the life config that points the host code at it."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.calls: List[str] = []
        self.server = ThreadedMotoServer(ip_address='127.0.0.1', port=0, verbose=False)

    def start(self) -> 'R2Server':
        logging.getLogger('werkzeug').setLevel(logging.ERROR)       # no line for each request
        self.server.start()
        host, port = self.server._server.server_address[:2]
        self.endpoint = 'http://%s:%d' % (host, port)
        self.reset()                # the server keeps its buckets in this process, also from an earlier test
        # The host client has region 'auto' (R2); this server only makes a bucket for a real region name.
        self.client = boto3.client('s3', endpoint_url=self.endpoint, region_name='us-east-1',
                                   aws_access_key_id='testing', aws_secret_access_key='testing')
        self.client.create_bucket(Bucket=BUCKET)
        creds = self.root / 'r2.env'
        creds.write_text('R2_ACCESS_KEY_ID=testing\nR2_SECRET_ACCESS_KEY=testing\n')
        self.life_cfg = self.root / 'life.json'
        self.life_cfg.write_text(json.dumps({'bucket': {
            'bucket': BUCKET, 'endpoint': self.endpoint, 'credentials_file': str(creds)}}))
        # Every client made from now on reports its requests here (the handler goes into each new client).
        self.events = boto3._get_default_session().events
        self.events.register('before-call.s3', self._record)
        return self

    def _record(self, model, **_kwargs) -> None:
        self.calls.append(model.name)

    def reset(self) -> None:
        urllib.request.urlopen(urllib.request.Request(self.endpoint + '/moto-api/reset', method='POST'), timeout=30).close()

    def stop(self) -> None:
        self.events.unregister('before-call.s3', self._record)
        self.reset()
        self.server.stop()

    def keys(self, prefix: str = '') -> List[str]:
        pages = self.client.get_paginator('list_objects_v2').paginate(Bucket=BUCKET, Prefix=prefix)
        return sorted(entry['Key'] for page in pages for entry in page.get('Contents') or [])

    def body(self, key: str) -> bytes:
        return self.client.get_object(Bucket=BUCKET, Key=key)['Body'].read()

    def etag(self, key: str) -> str:
        return self.client.head_object(Bucket=BUCKET, Key=key)['ETag'].strip('"')

    def open_uploads(self) -> List[str]:
        return sorted(u['Key'] for u in self.client.list_multipart_uploads(Bucket=BUCKET).get('Uploads') or [])
