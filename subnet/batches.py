"""Bounded, non-executable batch transport."""
import io
import json
import zipfile
import zlib
import numpy as np
from .storage import canonical

MAX_UPLOAD = 100_000_000

class UploadBudgetExceeded(ValueError):
    """A complete candidate cannot fit the bounded cumulative object."""

def compression_policy(raw):
    if type(raw)is not dict or set(raw)!={'version','level'} or raw['version']!='lossless-deflate-v1' or type(raw['level'])is not int or not 0<=raw['level']<=9:raise ValueError('signed lossless artifact compression policy')
    return dict(raw)

def compression_for_manifest(manifest):
    return 6 if 'artifact_compression_policy'not in manifest else compression_policy(manifest['artifact_compression_policy'])['level']

def pack(batches, *, budget=None, stable=False, compression_level=6):
    from .artifact_budget import LEGACY,LONG
    budget=dict(LEGACY if budget is None else budget)
    if budget not in (LEGACY,LONG):raise ValueError('artifact budget')
    if type(stable)is not bool:raise ValueError('stable framing flag')
    if type(compression_level)is not int or not 0<=compression_level<=9:raise ValueError('lossless DEFLATE compression level')
    out = io.BytesIO()
    manifest = []
    def write(archive,name,data):
        if not stable:return archive.writestr(name,data,compresslevel=compression_level)
        info=zipfile.ZipInfo(name,date_time=(1980,1,1,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED
        info.create_system=3;info.external_attr=0o600<<16
        return archive.writestr(info,data,compresslevel=compression_level)
    with zipfile.ZipFile(out, 'w', compression=zipfile.ZIP_DEFLATED,compresslevel=compression_level) as z:
        for bi, (batch, arrays) in enumerate(batches):
            if stable and (bi>=32 or len(arrays)>32):raise ValueError('stable batch/rollout budget')
            refs = []
            for ri, turns in enumerate(arrays):
                if stable and len(turns)>32:raise ValueError('stable turn budget')
                row = []
                for ti, tensor in enumerate(turns):
                    if stable and (not isinstance(tensor,np.ndarray) or tensor.dtype!=np.dtype(np.float32) or tensor.ndim!=2 or any(n<=0 for n in tensor.shape) or tensor.shape[0]>budget['tensor_rows'] or tensor.shape[1]>200000):raise ValueError('stable tensor shape or dtype')
                    name = f'{bi}-{ri}-{ti}.npy'
                    buf = io.BytesIO(); np.save(buf, tensor, allow_pickle=False)
                    write(z,name,buf.getvalue()); row.append(name)
                refs.append(row)
            manifest.append(dict(batch=batch, arrays=refs))
        metadata=canonical(manifest)
        if stable and (len(metadata)>2_000_000 or len(z.infolist())+1>4096):raise ValueError('stable manifest/archive entry budget')
        write(z,'manifest.json',metadata)
    with zipfile.ZipFile(io.BytesIO(out.getvalue())) as archive:
        if sum(e.file_size for e in archive.infolist())>budget['raw_bytes']:
            raise UploadBudgetExceeded('raw upload exceeds budget')
    if out.tell() > budget['compressed_bytes']:
        raise UploadBudgetExceeded('upload exceeds budget')
    return out.getvalue()

def bounded_tensor(data, *, max_rows=512):
    """Validate NPY framing before NumPy can allocate its declared shape."""
    stream=io.BytesIO(data)
    version=np.lib.format.read_magic(stream)
    if version==(1,0):reader=np.lib.format.read_array_header_1_0
    elif version==(2,0):reader=np.lib.format.read_array_header_2_0
    else:raise ValueError('unsupported tensor NPY version')
    shape,fortran_order,dtype=reader(stream,max_header_size=10000)
    if type(max_rows)is not int or max_rows not in (512,2048):raise ValueError('tensor row budget')
    if dtype!=np.dtype(np.float32) or len(shape)!=2 or any(type(n) is not int or n<=0 for n in shape) or shape[0]>max_rows or shape[1]>200000:
        raise ValueError('tensor header shape or dtype')
    expected=shape[0]*shape[1]*dtype.itemsize
    if len(data)-stream.tell()!=expected:
        raise ValueError('tensor payload length mismatch')
    stream.seek(0)
    value=np.load(stream,allow_pickle=False)
    if value.dtype!=dtype or value.shape!=shape:
        raise ValueError('tensor decoded shape mismatch')
    return value


def unpack(data, *, max_upload=MAX_UPLOAD, budget=None):
    from .artifact_budget import LEGACY,LONG
    if budget is not None:
        budget=dict(budget)
        if budget not in (LEGACY,LONG) or max_upload!=MAX_UPLOAD:raise ValueError('artifact budget')
        max_upload=budget['compressed_bytes']
    else:budget=dict(LEGACY)
    if type(max_upload) is not int or not 0 < max_upload <= (LONG['compressed_bytes'] if budget==LONG else 250_000_000):
        raise ValueError('compressed budget bounds')
    if len(data) > max_upload:
        raise ValueError('compressed upload budget')
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        entries = z.infolist()
        if any(e.flag_bits & 1 for e in entries):raise ValueError('encrypted archive refused')
        if any(e.compress_type not in (zipfile.ZIP_STORED,zipfile.ZIP_DEFLATED,zipfile.ZIP_BZIP2,zipfile.ZIP_LZMA) for e in entries):raise ValueError('unsupported archive compression')
        names = [e.filename for e in entries]
        if len(names) != len(set(names)) or len(names) > 4096 or sum(e.file_size for e in entries) > budget['raw_bytes']:
            raise ValueError('archive budget or duplicate entries')
        if any('/' in n or '..' in n for n in names):
            raise ValueError('archive paths')
        if z.getinfo('manifest.json').file_size > 2_000_000:
            raise ValueError('manifest budget')
        records = json.loads(z.read('manifest.json'))
        if not isinstance(records, list) or len(records) > 32:
            raise ValueError('batch budget')
        result, referenced = [], {'manifest.json'}
        for record in records:
            arrays = []
            for turns in record['arrays']:
                if len(turns) > 32:
                    raise ValueError('turn budget')
                row = []
                for name in turns:
                    if name in referenced or name not in names:
                        raise ValueError('tensor reference')
                    referenced.add(name)
                    value = bounded_tensor(z.read(name),max_rows=budget['tensor_rows'])
                    if value.dtype != np.float32 or value.ndim != 2 or value.shape[0] > budget['tensor_rows'] or value.shape[1] > 200000:
                        raise ValueError('tensor shape')
                    row.append(value)
                arrays.append(row)
            result.append((record['batch'], arrays))
        if referenced != set(names):
            raise ValueError('unexpected entries')
        return result


class SubmissionRejected(ValueError):
    """Untrusted artifact framing/quota refusal, not a worker failure."""


def submission_records(data, *, budget, max_batches):
    # Approved policy errors remain outside the untrusted decoder boundary.
    from .artifact_budget import LEGACY,LONG
    if budget not in (LEGACY,LONG):raise ValueError('artifact budget')
    if type(max_batches) is not int or max_batches < 1:raise ValueError('operator batch quota')
    try:
        records=unpack(data,budget=budget)
        if len(records)>max_batches:raise ValueError('batch quota')
        return records
    except (zipfile.BadZipFile,zipfile.LargeZipFile,zlib.error,ValueError,KeyError,TypeError,IndexError,AttributeError,EOFError,UnicodeError) as error:
        raise SubmissionRejected(type(error).__name__+': '+str(error)[:300]) from error
