"""Reject malformed native TOPLOC inputs before crossing the extension boundary."""
import base64


def validate_framing(proofs, expected, topk=128):
    if not isinstance(proofs,list) or len(proofs)!=expected:
        raise ValueError('proof count')
    for proof in proofs:
        if not isinstance(proof,str) or not 0<len(proof)<=16384:
            raise ValueError('proof encoding budget')
        raw=base64.b64decode(proof,validate=True)
        if len(raw)!=2+2*topk or not 32769<=int.from_bytes(raw[:2],'big')<=65497:
            raise ValueError('TOPLOC proof framing')
