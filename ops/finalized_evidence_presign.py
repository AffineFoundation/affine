"""Offline validation of exact configured S3 SigV4 GET capabilities.

Only the existing Bucket's explicit signing credentials are used. No object is
read or written, and errors deliberately omit URLs, query values and credentials.
"""
import datetime
import hmac
import re
from urllib.parse import parse_qsl, urlsplit, urlunsplit

from botocore.auth import S3SigV4QueryAuth
from botocore.awsrequest import AWSRequest


FIELDS = frozenset(('X-Amz-Algorithm', 'X-Amz-Credential', 'X-Amz-Date',
                    'X-Amz-Expires', 'X-Amz-SignedHeaders', 'X-Amz-Signature'))
SIGNATURE = re.compile(r'[0-9a-f]{64}\Z')


def _route(url):
    if (type(url) is not str or not 1 <= len(url) <= 32768
            or not url.isascii() or any(ord(c) <= 32 or ord(c) == 127 for c in url)):
        raise ValueError('bounded ordinary presigned URL')
    try:
        parts = urlsplit(url)
    except ValueError:
        raise ValueError('ordinary HTTPS capability URL') from None
    if (parts.scheme != 'https' or not parts.hostname or parts.username is not None
            or parts.password is not None or '#' in url):
        raise ValueError('HTTPS capability without userinfo or fragment')
    try:
        pairs = parse_qsl(parts.query, keep_blank_values=True, strict_parsing=True,
                          max_num_fields=8)
    except ValueError:
        raise ValueError('bounded ordinary SigV4 query') from None
    query = dict(pairs)
    if (len(query) != len(pairs) or not FIELDS <= query.keys()
            or query.keys() - FIELDS - {'X-Amz-Security-Token'}
            or any(segment.partition('=')[0] not in query for segment in parts.query.split('&'))):
        raise ValueError('exact unique SigV4 query fields')
    if not SIGNATURE.fullmatch(query['X-Amz-Signature']):
        raise ValueError('SigV4 signature encoding')
    return parts, query


def validate_get_route(bucket, url, key, expires=604800):
    """Require an unexpired authentic GET for this bucket and exact object key.

    ``Bucket.presign`` performs local signing, using the same explicitly supplied
    credentials as ``bucket.client``. A fresh route supplies the configured
    endpoint, addressing style, credential scope and optional session token;
    the original route's own timestamp is used to verify its GET signature.
    """
    if type(key) is not str or not key or type(expires) is not int or not 1 <= expires <= 604800:
        raise ValueError('exact object key and bounded GET expiry')
    parts, query = _route(url)
    expected, current = _route(bucket.presign(key, operation='get_object', expires=expires))
    if (parts.scheme, parts.netloc, parts.path) != (expected.scheme, expected.netloc, expected.path):
        raise ValueError('configured bucket endpoint and exact object path')
    if (query['X-Amz-Algorithm'] != 'AWS4-HMAC-SHA256'
            or query['X-Amz-SignedHeaders'] != 'host'
            or query['X-Amz-Expires'] != str(expires)
            or query.keys() != current.keys()
            or query.get('X-Amz-Security-Token') != current.get('X-Amz-Security-Token')):
        raise ValueError('GET signing algorithm, headers, expiry and session binding')
    try:
        stamp = datetime.datetime.strptime(query['X-Amz-Date'], '%Y%m%dT%H%M%SZ').replace(tzinfo=datetime.timezone.utc)
    except ValueError:
        raise ValueError('SigV4 UTC signing timestamp') from None
    if stamp.strftime('%Y%m%dT%H%M%SZ') != query['X-Amz-Date']:
        raise ValueError('canonical SigV4 signing timestamp')
    now = datetime.datetime.now(datetime.timezone.utc)
    if stamp > now + datetime.timedelta(minutes=15) or stamp + datetime.timedelta(seconds=expires) <= now:
        raise ValueError('unexpired bounded-clock-skew GET capability')
    scope = query['X-Amz-Credential'].split('/')
    configured = current['X-Amz-Credential'].split('/')
    if (len(scope) != 5 or len(configured) != 5 or scope[0] != configured[0]
            or scope[1] != query['X-Amz-Date'][:8] or scope[2:] != configured[2:]
            or scope[3:] != ['s3', 'aws4_request']):
        raise ValueError('current credential and configured S3 signing scope')
    credentials = bucket.client._request_signer._credentials.get_frozen_credentials()
    if (credentials.access_key != scope[0]
            or credentials.token != query.get('X-Amz-Security-Token')):
        raise ValueError('current explicit signing credentials')
    unsigned = '&'.join(segment for segment in parts.query.split('&')
                        if segment.partition('=')[0] != 'X-Amz-Signature')
    request = AWSRequest(method='GET', url=urlunsplit((parts.scheme, parts.netloc, parts.path, unsigned, '')))
    request.context['timestamp'] = query['X-Amz-Date']
    signer = S3SigV4QueryAuth(credentials, scope[3], scope[2], expires=expires)
    canonical = signer.canonical_request(request)
    expected_signature = signer.signature(signer.string_to_sign(request, canonical), request)
    if not hmac.compare_digest(query['X-Amz-Signature'], expected_signature):
        raise ValueError('authentic GET capability signature')
