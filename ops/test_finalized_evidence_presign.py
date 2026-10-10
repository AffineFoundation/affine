"""Only dummy static credentials and local botocore signing; no network."""
import datetime
import unittest
from unittest.mock import patch
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import boto3
from botocore.config import Config

from ops.finalized_evidence_presign import validate_get_route


class Bucket:
    def __init__(self, *, token=None, secret='synthetic-secret'):
        self.name = 'synthetic-public'
        self.client = boto3.client('s3', endpoint_url='https://storage.example.invalid',
            region_name='auto', aws_access_key_id='synthetic-access', aws_secret_access_key=secret,
            aws_session_token=token,
            config=Config(signature_version='s3v4', s3={'addressing_style':'path'}))

    def presign(self, key, operation='get_object', expires=604800):
        return self.client.generate_presigned_url(operation,
            Params={'Bucket':self.name, 'Key':key}, ExpiresIn=expires)


class PresignTests(unittest.TestCase):
    def setUp(self):
        self.bucket = Bucket()
        self.key = 'public/epoch-evidence/example/0123456789.json.gz'

    def tearDown(self):
        self.bucket.client.close()

    def change(self, url, field, value):
        parts = urlsplit(url); query = dict(parse_qsl(parts.query)); query[field] = value
        return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment))

    def test_get_signed_at_original_timestamp_passes_without_network(self):
        earlier = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(minutes=2)
        with patch('botocore.auth.get_current_datetime', return_value=earlier):
            url = self.bucket.presign(self.key)
        with patch.object(self.bucket.client, 'get_object', side_effect=AssertionError('network')):
            validate_get_route(self.bucket, url, self.key)

    def test_exact_escaped_object_path_passes(self):
        key = 'public/epoch-evidence/a space/+plus/%literal.json'
        validate_get_route(self.bucket, self.bucket.presign(key), key)

    def test_session_token_is_exact_and_signature_verified(self):
        bucket = Bucket(token='synthetic+session/token=')
        try:
            url = bucket.presign(self.key)
            validate_get_route(bucket, url, self.key)
            with self.assertRaises(ValueError):
                validate_get_route(bucket, self.change(url, 'X-Amz-Security-Token', 'other'), self.key)
        finally:bucket.client.close()

    def test_put_signature_cannot_be_labeled_get(self):
        url = self.bucket.presign(self.key, operation='put_object')
        with self.assertRaisesRegex(ValueError, 'authentic GET'):
            validate_get_route(self.bucket, url, self.key)

    def test_wrong_endpoint_key_userinfo_fragment_and_scheme_fail(self):
        url = self.bucket.presign(self.key)
        for bad in (url.replace('storage.example.invalid', 'unrelated.invalid'),
                    url.replace('/example/', '/other/'),
                    url.replace('https://', 'https://user@'), url+'#fragment', url+'#',
                    url.replace('https://', 'http://'), url.replace('https://', 'https://user:secret@')):
            with self.subTest(bad=bad.split('?')[0]), self.assertRaises(ValueError):
                validate_get_route(self.bucket, bad, self.key)

    def test_unknown_duplicate_or_encoded_query_keys_fail(self):
        url = self.bucket.presign(self.key)
        for bad in (url+'&token=synthetic', url+'&X-Amz-Expires=604800',
                    url.replace('X-Amz-Signature=', 'X%2dAmz-Signature=')):
            with self.assertRaises(ValueError):validate_get_route(self.bucket, bad, self.key)

    def test_signed_response_override_is_outside_exact_schema(self):
        url = self.bucket.client.generate_presigned_url('get_object', Params={
            'Bucket':self.bucket.name, 'Key':self.key, 'ResponseContentType':'text/plain'}, ExpiresIn=604800)
        with self.assertRaises(ValueError):validate_get_route(self.bucket, url, self.key)

    def test_algorithm_header_expiry_scope_and_signature_changes_fail(self):
        url = self.bucket.presign(self.key)
        for field, value in (('X-Amz-Algorithm','OTHER'), ('X-Amz-SignedHeaders','host;content-type'),
                ('X-Amz-Expires','604801'), ('X-Amz-Expires','0604800'),
                ('X-Amz-Credential','other/20000101/auto/s3/aws4_request'),
                ('X-Amz-Signature','0'*64), ('X-Amz-Signature','not-hex')):
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_get_route(self.bucket, self.change(url, field, value), self.key)

    def test_authentic_but_expired_or_excessively_future_signatures_fail(self):
        for offset in (datetime.timedelta(days=-8), datetime.timedelta(minutes=20)):
            at = datetime.datetime.now(datetime.timezone.utc) + offset
            with patch('botocore.auth.get_current_datetime', return_value=at):
                url = self.bucket.presign(self.key)
            with self.assertRaisesRegex(ValueError, 'clock-skew'):
                validate_get_route(self.bucket, url, self.key)

    def test_same_access_key_different_secret_fails_crypto(self):
        other = Bucket(secret='different-synthetic-secret')
        try:
            with self.assertRaisesRegex(ValueError, 'authentic GET'):
                validate_get_route(self.bucket, other.presign(self.key), self.key)
        finally:other.client.close()

    def test_malformed_query_error_never_echoes_secret_text(self):
        url = self.bucket.presign(self.key) + '&SYNTHETIC_PRIVATE_VALUE'
        with self.assertRaises(ValueError) as caught:
            validate_get_route(self.bucket, url, self.key)
        self.assertNotIn('SYNTHETIC_PRIVATE_VALUE', str(caught.exception))


if __name__ == '__main__':unittest.main()
