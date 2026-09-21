"""Skip the identity temperature division, retaining full upstream tensor geometry.

Pinned upstream performs in-place logits.div_(1.0) before slicing responses.
Its full-vocabulary backward can allocate another full-context logits gradient.
At the approved T=1, division and its derivative are identities. Compile the
exact pinned forward with only this operation removed. Keep the entire actor
loss, batching and optimizer unchanged; reject any upstream source drift.
"""
import hashlib
import inspect
import textwrap
from types import MethodType

FORWARD_SHA256 = '5804c443d7b94490aad2769bb3637194eefdbedd3a6767af512d4e5576d9ee2d'


def enable_t1_identity_repair(actor):
    if getattr(actor, '_t1_identity_repair_enabled', False):
        raise ValueError('Identity repair already enabled')
    if actor.use_remove_padding or actor.use_fused_kernels or actor.use_ulysses_sp:
        raise ValueError('Only the approved padded non-fused single-rank actor is supported')
    original=actor._forward_micro_batch.__func__
    source=textwrap.dedent(inspect.getsource(original))
    if hashlib.sha256(source.encode()).hexdigest()!=FORWARD_SHA256:
        raise ValueError('Pinned upstream forward differs from verified source')
    needle='logits.div_(temperature)'
    if source.count(needle)!=1:raise ValueError('Expected exactly one padded identity division')
    rewritten=source.replace(needle,'assert temperature == 1.0, "T1 memory repair requires temperature1"')
    namespace=dict(original.__globals__)
    exec(compile(rewritten, __file__+'::pinned_forward_identity_only', 'exec'),namespace)
    patched=namespace['_forward_micro_batch']
    actor._forward_micro_batch=MethodType(patched,actor)
    actor._t1_identity_repair_enabled=True
