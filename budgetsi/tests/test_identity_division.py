import ast,hashlib,inspect,textwrap,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
from budgetsi import identity_division as repair

class ToyActor:
    use_remove_padding=use_fused_kernels=use_ulysses_sp=False
    def __init__(self):self.weight=torch.nn.Parameter(torch.arange(12,dtype=torch.float64).reshape(3,4)/10)
    def _forward_micro_batch(self,batch,temperature):
        logits=batch @ self.weight
        logits.div_(temperature)
        return logits[-2:].log_softmax(-1)

class IdentityDivisionTests(unittest.TestCase):
    def test_pinned_source_only_one_identity_division_is_removed(self):
        path=Path(__file__).resolve().parents[2]/'verl/verl/workers/actor/dp_actor.py'
        source=path.read_text();node=next(n for n in ast.walk(ast.parse(source)) if isinstance(n,ast.FunctionDef) and n.name=='_forward_micro_batch')
        method=textwrap.dedent('\n'.join(source.splitlines()[node.lineno-1:node.end_lineno])+'\n')
        self.assertEqual(hashlib.sha256(method.encode()).hexdigest(),repair.FORWARD_SHA256)
        self.assertEqual(method.count('logits.div_(temperature)'),1)
    def test_identity_has_exact_probabilities_and_gradient_and_preserves_shape(self):
        actor=ToyActor();batch=torch.arange(18,dtype=torch.float64).reshape(6,3)/10
        expected=actor._forward_micro_batch(batch,1.);expected.sum().backward();grad=actor.weight.grad.clone();actor.weight.grad=None
        source=textwrap.dedent(inspect.getsource(actor._forward_micro_batch.__func__))
        with patch.object(repair,'FORWARD_SHA256',hashlib.sha256(source.encode()).hexdigest()):repair.enable_t1_identity_repair(actor)
        got=actor._forward_micro_batch(batch,1.);got.sum().backward()
        self.assertTrue(torch.equal(expected,got));self.assertTrue(torch.equal(grad,actor.weight.grad))
        with self.assertRaises(AssertionError):actor._forward_micro_batch(batch,0.5)
    def test_source_drift_and_repeat_install_rejected(self):
        with self.assertRaisesRegex(ValueError,'Pinned upstream'):repair.enable_t1_identity_repair(ToyActor())
        actor=ToyActor();actor.use_remove_padding=True
        with self.assertRaises(ValueError):repair.enable_t1_identity_repair(actor)

if __name__=='__main__':unittest.main()
