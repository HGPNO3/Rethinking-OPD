import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from omegaconf import OmegaConf

from budgetsi.audit.source_probe import function, numerical_functions
from budgetsi.top16 import SelectedAction, _binding
from budgetsi.variant_bridge import prepare_batch
from budgetsi.variant_spec import DEFAULT, VARIANTS, from_config, get_variant


class Data:
    def __init__(self, tensors, meta_info=None):
        self.batch, self.meta_info, self.non_tensor_batch = tensors, meta_info or {}, {}

    @classmethod
    def from_dict(cls, tensors):
        return cls(tensors)

    def select(
        self, batch_keys, non_tensor_batch_keys=None, non_tensor_select_keys=None
    ):
        return Data({k: self.batch[k] for k in batch_keys}, self.meta_info)

    def split(self, size):
        return [self]  # numerical fixture is one complete action

    def to(self, device):
        return self


def fixture(variant):
    torch.manual_seed(42)
    p = torch.randn(1, 3, 32, dtype=torch.float64).log_softmax(-1)
    q = torch.randn(1, 3, 32, dtype=torch.float64).log_softmax(-1)
    sid = torch.arange(16).repeat(3, 1)
    tid = torch.stack(
        [torch.arange(8, 24).flip(0), torch.arange(16, 32), torch.arange(16).flip(0)]
    )
    a = SelectedAction(
        "a", "s", (28,), (30, 28), (4, 5, 2), "tok", "tok", "prompt", "ref", 0.7
    )
    sample = p[0].gather(-1, torch.tensor(a.target_ids)[:, None]).squeeze(-1)
    s = {
        "binding": _binding(a),
        "actor_revision": "r",
        "variant": variant.contract(),
        "teacher_temperature": 1.0,
        "sampled": sample,
    }
    if variant.k:
        s.update(
            ids=sid,
            student=p[0].gather(-1, sid),
            teacher=q[0].gather(-1, sid),
            teacher_ids=tid,
            teacher_topk=q[0].gather(-1, tid),
        )
    else:
        s["teacher_sampled"] = (
            q[0].gather(-1, torch.tensor(a.target_ids)[:, None]).squeeze(-1)
        )
    return a, s, p, q


class Variants(unittest.TestCase):
    def test_cross_variant_resume_and_unbound_approval_rejected(self):
        from budgetsi.formal_gate import check_launch
        from budgetsi.run_state import Ledger

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            base = {"target_nodes": 1000, "opd_variant": "union_student"}
            Ledger(path, base, "same_commit", False)
            with self.assertRaisesRegex(ValueError, "binding changed"):
                Ledger(
                    path,
                    dict(base, opd_variant="intersection_student"),
                    "same_commit",
                    True,
                )
            config = Path(__file__).resolve().parents[1] / "variants/union_student.json"
            approval = path / "approval.json"
            approval.write_text("{}")
            with self.assertRaisesRegex(ValueError, "explicitly bind"):
                check_launch(config, approval, Path(__file__).resolve().parents[2])

    def test_config_catalog_and_baseline_unchanged_settings(self):
        root = Path(__file__).resolve().parents[1]
        base = json.loads((root / "formal_config.json").read_text())
        self.assertEqual(from_config(base), DEFAULT)
        for name, variant in VARIANTS.items():
            cfg = json.loads((root / "variants" / (name + ".json")).read_text())
            self.assertEqual(from_config(cfg), variant)
            for k in base:
                if k not in {"schema_version", "scope", "authorization"}:
                    self.assertEqual(cfg[k], base[k], k)
            cfg["opd"]["k"] = 999
            with self.assertRaises(ValueError):
                from_config(cfg)
        with self.assertRaises(ValueError):
            get_variant("invented")
        base["opd_variant"] = "union_student"
        with self.assertRaises(ValueError):
            from_config(base)

    def test_upstream_union_and_intersection_reward_and_gradient(self):
        for name in ["union_student", "intersection_student", "union_teacher"]:
            with self.subTest(name=name):
                v = VARIANTS[name]
                a, score, p, q = fixture(v)
                tensors, meta = prepare_batch(
                    [a], [score], variant=v, snapshot_id="s", eos_ids={2}, pad_id=0
                )
                self.assertEqual(
                    tensors["overlap_mask"][0].sum(-1).tolist(), [8, 0, 16]
                )
                # Reversed teacher ordering ensures the two mask axes aren't swapped.
                expected_mask = (
                    score["teacher_ids"][..., None] == score["ids"][:, None, :]
                ).any(-1)
                torch.testing.assert_close(
                    tensors["teacher_in_student_mask"][0], expected_mask
                )
                ns = numerical_functions()
                ns.update(DataProto=Data, get_device_id=lambda: "cpu")
                reward_fn = function(
                    "verl/verl/workers/actor/dp_actor.py",
                    "compute_distillation_reward",
                    ns,
                    True,
                    parent="DataParallelPPOActor",
                )

                def forward(inputs, probabilities=p, **kw):
                    return (
                        None,
                        None,
                        None,
                        probabilities.gather(-1, kw["student_top_k_ids"]),
                    )

                actor = SimpleNamespace(
                    actor_module=torch.nn.Linear(1, 1), _forward_micro_batch=forward
                )
                result = reward_fn(actor, Data(tensors, meta)).batch
                adv, _ = ns["compute_token_reward_direct_advantage"](
                    result["rm_scores"], tensors["response_mask"]
                )
                ids = result.get("union_top_k_ids", tensors["student_top_k_ids"])
                expected = torch.zeros_like(adv)
                for t in range(3):
                    stu, tea = (
                        set(score["ids"][t].tolist()),
                        set(score["teacher_ids"][t].tolist()),
                    )
                    support = stu & tea if v.strategy == "intersection" else stu | tea
                    if not support:
                        continue
                    den = sum(
                        (p if v.weight == "student_p" else q)[0, t, k].exp()
                        for k in support
                    )
                    seen = set()
                    for j, k in enumerate(ids[0, t].tolist()):
                        if k in support and k not in seen:
                            expected[0, t, j] = (
                                (q[0, t, k] - p[0, t, k])
                                * (p if v.weight == "student_p" else q)[0, t, k].exp()
                                / den
                            )
                            seen.add(k)
                torch.testing.assert_close(adv, expected)
                self.assertTrue(torch.isfinite(adv).all())
                if v.strategy == "intersection":
                    self.assertTrue((adv[0, 1] == 0).all())
                z = p.clone().requires_grad_()
                lp = z.log_softmax(-1).gather(-1, ids)
                cfg = OmegaConf.create(
                    {
                        "clip_ratio": 0.2,
                        "clip_ratio_low": 0.2,
                        "clip_ratio_high": 0.2,
                        "clip_ratio_c": 3.0,
                    }
                )
                loss = ns["compute_policy_loss_vanilla"](
                    lp.detach(), lp, adv, tensors["response_mask"], config=cfg
                )[0]
                gradient = torch.autograd.grad(loss, z)[0]
                lp = z.log_softmax(-1).gather(-1, ids)
                reference = -(expected.detach() * lp).sum() / 3
                torch.testing.assert_close(
                    gradient, torch.autograd.grad(reference, z)[0]
                )

    def test_sampled_path_has_no_candidate_tensors_or_extra_probability_weight(self):
        v = VARIANTS["sampled_token"]
        a, s, _p, _q = fixture(v)
        tensors, meta = prepare_batch(
            [a], [s], variant=v, snapshot_id="s", eos_ids={2}, pad_id=0
        )
        self.assertEqual(meta["top_k"], 0)
        self.assertFalse(any("top_k" in key for key in tensors))
        expected = s["teacher_sampled"] - s["sampled"]
        actual = -(tensors["old_log_probs"] - tensors["teacher_sampled_log_probs"])
        torch.testing.assert_close(actual[0], expected)
        self.assertFalse(torch.allclose(actual[0], expected * s["sampled"].exp()))
        self.assertEqual(tensors["response_mask"].tolist(), [[1, 1, 1]])  # includes EOS
        self.assertFalse((tensors["input_ids"] == 30).any())  # teacher reference absent

    def test_variant_and_context_cache_mixing_rejected(self):
        v = VARIANTS["union_student"]
        a, s, _, _ = fixture(v)
        for variant in [
            VARIANTS["union_teacher"],
            VARIANTS["intersection_student"],
            DEFAULT,
        ]:
            with self.assertRaises(ValueError):
                prepare_batch(
                    [a], [s], variant=variant, snapshot_id="s", eos_ids={2}, pad_id=0
                )
        bad = copy.deepcopy(s)
        bad["binding"] = "wrong"
        with self.assertRaises(ValueError):
            prepare_batch([a], [bad], variant=v, snapshot_id="s", eos_ids={2}, pad_id=0)
        bad = copy.deepcopy(s)
        bad["teacher_ids"][:, 1] = bad["teacher_ids"][:, 0]
        with self.assertRaises(ValueError):
            prepare_batch([a], [bad], variant=v, snapshot_id="s", eos_ids={2}, pad_id=0)
        bad = copy.deepcopy(s)
        bad["teacher_topk"][0, 0] = float("nan")
        with self.assertRaises(ValueError):
            prepare_batch([a], [bad], variant=v, snapshot_id="s", eos_ids={2}, pad_id=0)


if __name__ == "__main__":
    unittest.main()
