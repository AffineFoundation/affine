"""Bounded lossless complete-group archives, independent of old ZIP quotas.

Sizes derive from the signed N and horizon/context ceilings. Absolute resource
ceilings reject oversized configurations; they never truncate an input. This
codec alone does not qualify a longer generation harness or authenticate data.
"""
from dataclasses import dataclass
import io
import json
import math
import struct
import zipfile
import numpy as np
from .storage import canonical

POLICY = 'complete-group-bounded-selected-v1'
MAX_OBJECT_BYTES = 512 * 1024**2
MAX_CUMULATIVE_BYTES = 4 * 1024**3


def need(value, message):
    if not value:
        raise ValueError(message)


def loads(data, limit):
    need(type(data) is bytes and 0 < len(data) <= limit, 'bounded JSON bytes')
    def pairs(items):
        result = {}
        for key, value in items:
            need(key not in result, 'duplicate JSON key')
            result[key] = value
        return result
    def constant(value):
        raise ValueError('nonfinite JSON constant')
    value = json.loads(data, object_pairs_hook=pairs, parse_constant=constant)
    need(canonical(value) == data, 'canonical JSON encoding required')
    return value


@dataclass(frozen=True)
class Limits:
    group_size: int
    horizon: int
    context: int
    metadata_bytes: int
    raw_bytes: int
    compressed_bytes: int
    cumulative_bytes: int
    commitment_bytes: int


def limits(manifest):
    # Intentionally independent of harness.normalize: codecs can qualify long
    # vectors while the real Miner still fails closed on unqualified harnesses.
    group = manifest.get('group_contract', {})
    n = group.get('group_size')
    need(type(n) is int and 2 <= n <= 128, 'bounded signed group size')
    policy = manifest.get('group_transport_policy')
    from .complete_group_distribution import geometry
    geometry(policy)
    context = policy['max_context_tokens']
    need(type(context) is int and 2 <= context <= 65536, 'bounded signed context')
    cap = manifest.get('max_batches')
    need(type(cap) is int and 1 <= cap <= 256, 'bounded signed per-miner group cap')
    need(manifest.get('probability_artifact_policy') == {'version':'selected-token-logprobs-v1'},
         'complete groups require explicit compact selected-T1 artifact policy')
    definitions = manifest.get('environments')
    need(type(definitions) is list and definitions, 'explicit environment registry required')
    horizons = []
    for row in definitions:
        need(row['spec'].get('max_turns') == 1, 'initial complete-group transport is single-turn')
        raw = row['harness']
        choices = raw['by_index'].values() if raw.get('version') == 'indexed-harness-v1' else [raw]
        for h in choices:
            value = h.get('max_output_tokens')
            need(type(value) is int and 1 <= value <= 32768 and value < context,
                 'bounded signed output horizon within context')
            horizons.append(value)
    need(bool(horizons), 'nonempty signed horizon registry')
    horizon = max(horizons)
    # Token integers, both selected probability representations, escaped text,
    # framed TOPLOC proofs, observations and per-rollout bindings. Actual field
    # and token limits are checked separately by structural admission.
    metadata = 16384 + n * (98304 + context*12 + horizon*512)
    raw = metadata + n * (horizon*4 + 512)
    compressed = raw + 4096 + n*256
    cumulative = compressed * cap
    need(compressed <= MAX_OBJECT_BYTES and cumulative <= MAX_CUMULATIVE_BYTES,
         'signed complete-group configuration exceeds absolute transport resources')
    return Limits(n,horizon,context,metadata,raw,compressed,cumulative,4096+cap*1024)


def encode(records, bound, *, level=6):
    need(isinstance(bound, Limits) and type(records) is list and len(records) == 1,
         'one bounded complete-group archive')
    need(type(level) is int and 0 <= level <= 9, 'bounded compression level')
    stream = io.BytesIO(); metadata = []
    with zipfile.ZipFile(stream,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=level) as archive:
        def write(name, body):
            info=zipfile.ZipInfo(name,date_time=(1980,1,1,0,0,0))
            info.compress_type=zipfile.ZIP_DEFLATED;info.create_system=3;info.external_attr=0o600<<16
            archive.writestr(info,body,compresslevel=level)
        for batch, arrays in records:
            need(type(arrays) is list and 1 <= len(arrays) <= bound.group_size, 'original bounded rollout arrays')
            refs=[]
            for nonce, turns in enumerate(arrays):
                need(type(turns) is list and len(turns) == 1, 'one selected-T1 tensor per rollout')
                value=turns[0]
                need(isinstance(value,np.ndarray) and value.dtype == np.float32
                     and value.ndim == 2 and value.shape[1] == 1 and 0 < value.shape[0] <= bound.horizon
                     and np.isfinite(value).all() and (value <= 0).all(), 'finite selected-T1 vector within signed horizon')
                blob=io.BytesIO();np.save(blob,np.ascontiguousarray(value),allow_pickle=False)
                name=f'{nonce}.npy';write(name,blob.getvalue());refs.append([name])
            metadata.append(dict(batch=batch,arrays=refs))
        body=canonical(metadata);need(len(body) <= bound.metadata_bytes,'complete-group metadata budget')
        write('manifest.json',body)
    data=stream.getvalue();need(len(data) <= bound.compressed_bytes,'complete-group compressed budget')
    return data


def _preflight_zip(data,bound):
    """Bound directory allocation before constructing Python's ZipInfo list.

    Our writer emits ordinary single-disk ZIPs without extras, descriptors or
    comments. Reject alternate framing (including ZIP64) rather than allowing a
    tiny declared N to hide millions of central-directory records.
    """
    need(len(data)>=22,'complete ZIP end record')
    tail=struct.unpack('<4s4H2LH',data[-22:])
    signature,disk,start_disk,disk_count,count,directory_bytes,directory_offset,comment=tail
    need(signature==b'PK\x05\x06' and disk==start_disk==0 and disk_count==count
         and 2<=count<=bound.group_size+1 and comment==0
         and 0<directory_bytes<=count*128 and directory_offset+directory_bytes==len(data)-22,
         'bounded ordinary single-disk ZIP directory before allocation')
    cursor=directory_offset;local_offset=0;total_raw=0
    for _ in range(count):
        need(cursor+46<=len(data)-22,'complete bounded ZIP directory entry')
        row=struct.unpack('<4s6H3L5H2L',data[cursor:cursor+46])
        sig,made,version,flags,method,clock,date,crc,compressed,raw,name_length,extra_length,comment_length,start,internal,external,offset=row
        need(sig==b'PK\x01\x02' and version<=20 and flags==0 and method in (0,8)
             and 1<=name_length<=32 and extra_length==comment_length==start==0 and offset==local_offset,
             'canonical bounded ZIP directory framing')
        end=cursor+46+name_length;need(end<=len(data)-22,'bounded ZIP directory name')
        name=data[cursor+46:end];cursor=end
        need(local_offset+30+name_length<=directory_offset,'bounded local ZIP header')
        header=struct.unpack('<4s5H3L2H',data[local_offset:local_offset+30])
        need(header==(b'PK\x03\x04',version,flags,method,clock,date,crc,compressed,raw,name_length,0)
             and data[local_offset+30:local_offset+30+name_length]==name,
             'exact local/central ZIP framing')
        local_offset+=30+name_length+compressed
        total_raw+=raw
        need(local_offset<=directory_offset and total_raw<=bound.raw_bytes,'ZIP member data/expanded bounds')
    need(cursor==len(data)-22 and local_offset==directory_offset,'no hidden ZIP prefix/trailing records')


def decode(data, bound):
    need(isinstance(bound,Limits) and type(data) is bytes and 0 < len(data) <= bound.compressed_bytes,
         'bounded complete-group archive bytes')
    _preflight_zip(data,bound)
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        infos=archive.infolist();names=[i.filename for i in infos]
        need(2 <= len(names) <= bound.group_size+1 and len(names) == len(set(names))
             and all(not (i.flag_bits & 1) and i.compress_type in (zipfile.ZIP_STORED,zipfile.ZIP_DEFLATED) for i in infos)
             and sum(i.file_size for i in infos) <= bound.raw_bytes, 'bounded unencrypted unique ZIP members')
        info=archive.getinfo('manifest.json');need(info.file_size <= bound.metadata_bytes,'bounded archive metadata')
        records=loads(archive.read('manifest.json'),bound.metadata_bytes)
        need(type(records) is list and len(records) == 1 and type(records[0]) is dict
             and set(records[0]) == {'batch','arrays'}, 'exact single group archive framing')
        refs=records[0]['arrays'];need(type(refs) is list and 1 <= len(refs) <= bound.group_size,'bounded array references')
        need(refs == [[f'{i}.npy'] for i in range(len(refs))]
             and set(names) == {'manifest.json',*(f'{i}.npy' for i in range(len(refs)))}, 'canonical original array slots only')
        arrays=[]
        for nonce in range(len(refs)):
            info=archive.getinfo(f'{nonce}.npy');need(info.file_size <= bound.horizon*4+512,'bounded selected array before read')
            body=archive.read(info);reader=io.BytesIO(body);version=np.lib.format.read_magic(reader)
            need(version in ((1,0),(2,0)),'supported NPY framing')
            header=np.lib.format.read_array_header_1_0 if version==(1,0) else np.lib.format.read_array_header_2_0
            shape,fortran,dtype=header(reader,max_header_size=256)
            need(dtype == np.dtype(np.float32) and type(shape) is tuple and len(shape)==2
                 and shape[1]==1 and type(shape[0]) is int and 0 < shape[0] <= bound.horizon
                 and not fortran and len(body)-reader.tell()==shape[0]*4, 'exact bounded selected-vector NPY header')
            value=np.frombuffer(body,offset=reader.tell(),dtype=np.float32).reshape(shape).copy()
            need(np.isfinite(value).all() and (value<=0).all(),'finite nonpositive selected logprobs')
            arrays.append([value])
    return [(records[0]['batch'],arrays)]
