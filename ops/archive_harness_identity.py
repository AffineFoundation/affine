"""Derive a reviewed harness identity from authenticated source archive bytes.

The caller verifies the signed source descriptor and complete reported inventory.
This reader independently hashes the required modules and recognizes only the
published source_hash construction. It never imports the archived modules.
"""
import ast
import hashlib
import io
import tarfile

MODULES=('subnet/harness.py','subnet/native_mrcr_public_policy.py','subnet/sample_harness.py')

def _shape(expression):
    return ast.dump(ast.parse(expression,mode='eval').body,include_attributes=False)

READS={_shape('Path(__file__).read_bytes()'):MODULES[0]}
for filename in MODULES[1:]:
    READS[_shape('(Path(__file__).parent / '+repr(filename.split('/')[-1])+').read_bytes()')]=filename


def identity(body,expected_files):
    if not isinstance(body,bytes) or not 0<len(body)<=32*1024**2:
        raise ValueError('bounded authenticated source archive')
    if not isinstance(expected_files,dict) or MODULES[0] not in expected_files:
        raise ValueError('pinned archived harness required')
    values={}
    with tarfile.open(fileobj=io.BytesIO(body),mode='r:gz') as archive:
        for member in archive:
            name=member.name[2:] if member.name.startswith('./') else member.name
            if name not in MODULES or name not in expected_files:continue
            if (name in values or not member.isfile() or member.issym() or
                    not 0<member.size<=1024**2):
                raise ValueError('exact bounded archived module')
            with archive.extractfile(member) as source:content=source.read(1024**2+1)
            if len(content)!=member.size or hashlib.sha256(content).hexdigest()!=expected_files.get(name):
                raise ValueError('archived module inventory binding')
            values[name]=content
    if MODULES[0] not in values:raise ValueError('archived harness missing')
    tree=ast.parse(values[MODULES[0]].decode('utf-8'))
    functions=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='source_hash']
    if len(functions)!=1:raise ValueError('exact archived source_hash function')
    fn=functions[0]
    statements=list(fn.body)
    if statements and isinstance(statements[0],ast.Expr) and isinstance(statements[0].value,ast.Constant) and isinstance(statements[0].value.value,str):statements.pop(0)
    if (fn.decorator_list or fn.args.args or fn.args.posonlyargs or fn.args.kwonlyargs
            or fn.args.vararg or fn.args.kwarg or len(statements)!=1
            or not isinstance(statements[0],ast.Return)):
        raise ValueError('reviewed archived source_hash construction')
    returned=statements[0].value
    if (not isinstance(returned,ast.Call) or returned.args or returned.keywords
            or not isinstance(returned.func,ast.Attribute) or returned.func.attr!='hexdigest'):
        raise ValueError('reviewed hexadecimal source hash')
    digest=returned.func.value
    if (not isinstance(digest,ast.Call) or len(digest.args)!=1 or digest.keywords
            or ast.dump(digest.func,include_attributes=False)!=_shape('hashlib.sha256')):
        raise ValueError('reviewed SHA256 source hash')
    def leaves(node):
        if isinstance(node,ast.BinOp) and isinstance(node.op,ast.Add):
            return leaves(node.left)+leaves(node.right)
        name=READS.get(ast.dump(node,include_attributes=False))
        if name is None:raise ValueError('unreviewed archived hash input')
        return [name]
    inputs=leaves(digest.args[0])
    if inputs[0]!=MODULES[0] or len(inputs)!=len(set(inputs)):
        raise ValueError('exact archived source hash inputs')
    if any(name not in values for name in inputs):raise ValueError('archived hash input missing')
    return hashlib.sha256(b''.join(values[name] for name in inputs)).hexdigest()
