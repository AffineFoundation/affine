"""Bounded, non-executable batch transport."""
import io
import json
import zipfile
import numpy as np
from .storage import canonical

MAX_UPLOAD = 100_000_000

def pack(batches):
    out = io.BytesIO()
    manifest = []
    with zipfile.ZipFile(out, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for bi, (batch, arrays) in enumerate(batches):
            refs = []
            for ri, turns in enumerate(arrays):
                row = []
                for ti, tensor in enumerate(turns):
                    name = f'{bi}-{ri}-{ti}.npy'
                    buf = io.BytesIO(); np.save(buf, tensor, allow_pickle=False)
                    z.writestr(name, buf.getvalue()); row.append(name)
                refs.append(row)
            manifest.append(dict(batch=batch, arrays=refs))
        z.writestr('manifest.json', canonical(manifest))
    if out.tell() > MAX_UPLOAD:
        raise ValueError('upload exceeds budget')
    return out.getvalue()

def bounded_tensor(data):
    """Validate NPY framing before NumPy can allocate its declared shape."""
    stream=io.BytesIO(data)
    version=np.lib.format.read_magic(stream)
    if version==(1,0):reader=np.lib.format.read_array_header_1_0
    elif version==(2,0):reader=np.lib.format.read_array_header_2_0
    else:raise ValueError('unsupported tensor NPY version')
    shape,fortran_order,dtype=reader(stream,max_header_size=10000)
    if dtype!=np.dtype(np.float32) or len(shape)!=2 or any(type(n) is not int or n<=0 for n in shape) or shape[0]>512 or shape[1]>200000:
        raise ValueError('tensor header shape or dtype')
    expected=shape[0]*shape[1]*dtype.itemsize
    if len(data)-stream.tell()!=expected:
        raise ValueError('tensor payload length mismatch')
    stream.seek(0)
    value=np.load(stream,allow_pickle=False)
    if value.dtype!=dtype or value.shape!=shape:
        raise ValueError('tensor decoded shape mismatch')
    return value


def unpack(data):
    if len(data) > MAX_UPLOAD:
        raise ValueError('compressed upload budget')
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        entries = z.infolist()
        names = [e.filename for e in entries]
        if len(names) != len(set(names)) or len(names) > 4096 or sum(e.file_size for e in entries) > 500_000_000:
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
                    value = bounded_tensor(z.read(name))
                    if value.dtype != np.float32 or value.ndim != 2 or value.shape[0] > 512 or value.shape[1] > 200000:
                        raise ValueError('tensor shape')
                    row.append(value)
                arrays.append(row)
            result.append((record['batch'], arrays))
        if referenced != set(names):
            raise ValueError('unexpected entries')
        return result
