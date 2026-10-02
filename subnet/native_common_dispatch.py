"""Prospective qualified native dispatch; no default provider fallback on bad pins."""
import hashlib
from pathlib import Path
from .native_prolog_session import VERSION,NativePrologSession
from .native_prolog_actor import isolation_command

def validate_prolog_binding(spec):
    if spec.id!='affine_prolog' or spec.adapter!='prime_v1' or spec.version!='prime-native-prolog-dispatch-v2' or spec.config.get('prolog_session_revision')!=VERSION:
        raise ValueError('exact qualified native Prolog dispatch')
    pins=spec.config.get('prolog_source_files')
    expected={'subnet/native_prolog_actor.py','subnet/native_prolog_session.py'}
    if not isinstance(pins,dict)or set(pins)!=expected:raise ValueError('exact native Prolog source membership')
    root=Path(__file__).resolve().parents[1]
    for name,digest in pins.items():
        path=root/name
        if path.is_symlink()or not path.is_file()or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:raise ValueError('native Prolog source pin')
    isolation_command('affine-prolog-native-'+'a'*16,spec.config.get('prolog_runtime',{}))
    if spec.max_turns!=2 or not 512<=spec.max_output_tokens<=32768 or spec.num_samples!=3 or type(spec.success_reward)not in(int,float)or spec.success_reward!=1.:
        raise ValueError('qualified original three-fixture Prolog geometry')
    if spec.config.get('taskset')!={'tasks':['prolog-nqueens-0005','prolog-nqueens-0014','prolog-nqueens-0023'],'num_examples':32,'difficulty':'medium'}:
        raise ValueError('qualified original Prolog fixture population')

def prolog_session(spec):
    validate_prolog_binding(spec)
    from .environments import _source_hash
    if spec.source_hash!=_source_hash(spec):raise ValueError('native Prolog trusted environment source')
    return NativePrologSession(spec)
