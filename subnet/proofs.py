"""Reject malformed native TOPLOC inputs before crossing the extension boundary."""
import base64


def verify_mapped_proofs(activations, proofs, decode_batching_size, topk, *, num_threads=2):
    """Replay TOPLOC 0.1.6's encoded index map with exact error checks.

    Its builder and native verifier remap selected positions by the proof's
    modulus before interpolation/evaluation in the fixed 65497 coefficient
    field. The Python list verifier omits that remapping. Large prefill chunks
    can therefore reject their own honest proof. Keep the proof bytes and
    error thresholds unchanged, and use the native verifier's index semantics.
    """
    import math
    from statistics import mean, median
    import torch
    from toploc.poly import batch_activations
    from toploc.C.csrc.poly import ProofPoly, VerificationResult
    from toploc.C.csrc.ndd import evaluate_polynomials
    from toploc.C.csrc.utils import get_fp_parts
    if not isinstance(activations, list) or not activations:
        raise ValueError('TOPLOC activation framing')
    validate_framing(proofs, 1+math.ceil((len(activations)-1)/decode_batching_size), topk)
    results=[]
    for encoded, chunk in zip(proofs, batch_activations(activations, decode_batching_size, skip_prefill=False)):
        proof=ProofPoly.from_base64(encoded)
        chunk=chunk.view(-1).cpu()
        indices=chunk.abs().topk(k=topk).indices.tolist()
        mapped=[index % proof.modulus for index in indices]
        if len(set(mapped)) != topk:
            raise ValueError('TOPLOC noninjective index map')
        values=evaluate_polynomials(proof.coeffs, mapped)
        recovered=torch.tensor(values,dtype=torch.uint16).view(torch.bfloat16)
        exps,mants=get_fp_parts(recovered,num_threads=num_threads)
        actual_exps,actual_mants=get_fp_parts(chunk[indices],num_threads=num_threads)
        mismatches=[a != b for a,b in zip(exps,actual_exps)]
        errors=[abs(a-b) for a,b,bad in zip(mants,actual_mants,mismatches) if not bad]
        results.append(VerificationResult(sum(mismatches),mean(errors) if errors else 2**64,median(errors) if errors else 2**64))
    return results


def validate_framing(proofs, expected, topk=128):
    if not isinstance(proofs,list) or len(proofs)!=expected:
        raise ValueError('proof count')
    for proof in proofs:
        if not isinstance(proof,str) or not 0<len(proof)<=16384:
            raise ValueError('proof encoding budget')
        raw=base64.b64decode(proof,validate=True)
        if len(raw)!=2+2*topk or not 32769<=int.from_bytes(raw[:2],'big')<=65497:
            raise ValueError('TOPLOC proof framing')
