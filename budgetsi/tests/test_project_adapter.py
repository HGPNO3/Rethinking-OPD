import contextlib
import importlib.util
import io
import sys
import types
import unittest
from unittest.mock import patch

import torch
from omegaconf import OmegaConf

from budgetsi.adapter import actor_overrides, build_update, validate_actor
from budgetsi.audit.source_probe import ROOT, function, numerical_functions, read_source


def load_extension():
    """Load our real plugin against the exact pinned registration/aggregation API."""
    ns = numerical_functions()
    ns.update(POLICY_LOSS_REGISTRY={}, PolicyLossFn=object)
    function("verl/verl/trainer/ppo/core_algos.py", "register_policy_loss", ns, True)
    module = types.ModuleType("verl.trainer.ppo.core_algos")
    module.agg_loss = ns["agg_loss"]
    module.register_policy_loss = ns["register_policy_loss"]
    spec = importlib.util.spec_from_file_location("budgetsi_plugin_test", ROOT / "budgetsi/register_loss.py")
    plugin = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"verl.trainer.ppo.core_algos": module}):
        spec.loader.exec_module(plugin)
    return ns["POLICY_LOSS_REGISTRY"]["budgetsi_raw_sampled"]


LOSS = load_extension()


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.arange(64, dtype=torch.float64).reshape(2, 4, 8) / 37)


def fixture():
    logits = Toy().logits.detach()
    raw = logits.log_softmax(-1)
    behavior = (logits / 0.7).log_softmax(-1)
    rows = []
    scores = []
    for i, (prefix, target) in enumerate([([7], [3, 2]), ([6, 7], [4, 5, 6, 2])]):
        idx = torch.tensor(target)
        n = len(target)
        old = raw[i, :n].gather(-1, idx[:, None]).flatten().tolist()
        mu = behavior[i, :n].gather(-1, idx[:, None]).flatten().tolist()
        q = [v + (i + 1) * 0.2 for v in old]
        rows.append(
            dict(
                id=str(i),
                snapshot_id="toy-snapshot",
                prompt_binding="toy-binding",
                student_prefix=prefix,
                target_ids=target,
                loss_mask=[True] * n,
                original=dict(
                    prompt_token_ids=prefix,
                    generated_token_ids=target,
                    behavior_logprobs=mu,
                    behavior_temperature=0.7,
                    behavior_probability_mode="processed_logprobs",
                ),
                teacher_logprobs=q,
                teacher_score=dict(target_ids=target, raw_logprobs=q),
                behavior_logprobs=mu,
                reference_action={"argument": "illustrative private-to-teacher reference"},
            )
        )
        scores.append(
            dict(
                snapshot_id="toy-snapshot",
                prompt_ids=prefix,
                target_ids=target,
                raw_logprobs=old,
                backend="training_actor_raw",
            )
        )
    return rows, scores


def prepare(rows=None, scores=None):
    if rows is None:
        rows, scores = fixture()
    return build_update(rows, scores, snapshot_id="toy-snapshot", prompt_binding="toy-binding", pad_id=0, eos_ids={2})


class Batch:
    def __init__(self, tensors, metadata):
        self.batch = tensors
        self.meta_info = metadata
        self.non_tensor_batch = {}

    def select(self, batch_keys, non_tensor_batch_keys):
        return Batch({k: self.batch[k] for k in batch_keys}, self.meta_info)

    def split(self, n):
        return [
            Batch({k: v[i : i + n] for k, v in self.batch.items()}, self.meta_info)
            for i in range(0, len(self.batch["responses"]), n)
        ]

    def to(self, device):
        return self


class Actor:
    def __init__(self, dynamic=False, group_size=1):
        self.actor_module = Toy()
        self.actor_optimizer = torch.optim.AdamW(self.actor_module.parameters(), lr=0.001, weight_decay=0.01)
        self.config = OmegaConf.create(
            dict(
                policy_loss={"loss_mode": "budgetsi_raw_sampled"},
                loss_agg_mode="seq-mean-token-sum",
                ppo_mini_batch_size=2,
                ppo_micro_batch_size_per_gpu=1,
                ppo_epochs=1,
                use_rollout_log_probs=True,
                use_kl_loss=False,
                entropy_coeff=0.0,
                ulysses_sequence_parallel_size=1,
                use_dynamic_bsz=dynamic,
                ppo_max_token_len_per_gpu=100,
                grad_clip=1.0,
            )
        )
        self.ulysses_sequence_parallel_size = 1
        self.gradients = []
        ns = numerical_functions()
        ns.update(
            get_device_id=lambda: "cpu",
            get_policy_loss_fn=lambda name: LOSS,
            prepare_dynamic_batch=lambda b, max_token_len: (b.split(group_size), None),
            FSDP=type("FSDP", (), {}),
            FSDPModule=type("FSDPModule", (), {}),
            DTensor=type("DTensor", (), {}),
        )
        function("verl/verl/utils/py_functional.py", "append_to_dict", ns, True)
        self.update = function(
            "verl/verl/workers/actor/dp_actor.py", "update_policy", ns, True, parent="DataParallelPPOActor"
        )
        step = function(
            "verl/verl/workers/actor/dp_actor.py", "_optimizer_step", ns, True, parent="DataParallelPPOActor"
        )

        def record_step():
            self.gradients.append(self.actor_module.logits.grad.detach().clone())
            return step(self)

        self._optimizer_step = record_step

    def _forward_micro_batch(self, inputs, temperature, calculate_entropy=False):
        row = (inputs["input_ids"][:, 0] != 0).long()
        lp = (
            (self.actor_module.logits[row] / temperature)
            .log_softmax(-1)
            .gather(-1, inputs["responses"][..., None])
            .squeeze(-1)
        )
        return None, lp, None, None

    def run(self, p):
        validate_actor(self.config, len(p.node_ids), 1)
        with contextlib.redirect_stdout(io.StringIO()):
            metrics = self.update(self, Batch(p.tensors, p.metadata))
        return self.gradients[0], self.actor_module.logits.detach().clone(), sum(metrics["actor/pg_loss"])


class ProjectAdapter(unittest.TestCase):
    def test_framework_files_are_identical_to_pin(self):
        for path in [
            "verl/verl/workers/actor/dp_actor.py",
            "verl/verl/workers/config/actor.py",
            "verl/verl/trainer/config/actor/dp_actor.yaml",
            "on_policy_distillation.sh",
        ]:
            self.assertEqual((ROOT / path).read_text(), read_source(path, True))

    def test_original_tokens_reference_isolation_and_eos_padding(self):
        rows, scores = fixture()
        a = prepare(rows, scores)
        rows[0]["reference_action"]["argument"] = "another teacher-only reference, never student input"
        b = prepare(rows, scores)
        torch.testing.assert_close(a.tensors["input_ids"], b.tensors["input_ids"])
        self.assertEqual(a.tensors["response_mask"].tolist(), [[1, 1, 0, 0], [1, 1, 1, 1]])
        self.assertEqual(a.tensors["responses"][0].tolist(), [3, 2, 0, 0])
        self.assertNotIn("reference_action", a.tensors)
        self.assertEqual(a.metadata["temperature"], 1.0)

    def test_actor_loop_matches_direct_original_surrogate(self):
        p = prepare()
        actual = Actor().run(p)
        model = Toy()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        logp = model.logits.log_softmax(-1).gather(-1, p.tensors["responses"][..., None]).squeeze(-1)
        mask = p.tensors["response_mask"]
        b, t = 2, 6
        raw_adv = p.tensors["advantages"] * (t / b)
        expected = (-(logp - p.tensors["old_log_probs"]).exp() * raw_adv * mask).sum() / t
        expected.backward()
        gradient = model.logits.grad.detach().clone()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
        optimizer.step()
        torch.testing.assert_close(actual[0], gradient, atol=1e-8, rtol=1e-7)
        torch.testing.assert_close(actual[1], model.logits.detach(), atol=1e-8, rtol=1e-7)
        self.assertAlmostEqual(actual[2], expected.item(), places=7)

    def test_dynamic_partition_gradient_and_update_invariance(self):
        p = prepare()
        a = Actor(True, 1).run(p)
        b = Actor(True, 2).run(p)
        for x, y in zip(a[:2], b[:2], strict=True):
            torch.testing.assert_close(x, y, rtol=1e-12, atol=1e-12)
        self.assertAlmostEqual(a[2], b[2], places=12)

    def test_stale_retargeted_or_serving_scores_are_rejected(self):
        for change in ["snapshot", "target", "prefix", "score", "duplicate", "eos", "prompt", "hf_source"]:
            r, h = fixture()
            if change == "snapshot":
                h[0]["snapshot_id"] = "old"
            if change == "target":
                r[0]["target_ids"] = [4, 2]
            if change == "prefix":
                r[0]["student_prefix"] = [5]
            if change == "score":
                r[0]["teacher_logprobs"] = [float("nan"), -0.1]
            if change == "duplicate":
                r[1]["id"] = r[0]["id"]
            if change == "eos":
                r[0]["target_ids"] = [2, 3]
            if change == "prompt":
                r[0]["prompt_binding"] = "old"
            if change == "hf_source":
                h[0]["backend"] = "historical_serving_raw"
            with self.subTest(change=change), self.assertRaises(ValueError):
                prepare(r, h)

    def test_unverified_actor_settings_rejected(self):
        for key, value in [
            ("ppo_epochs", 2),
            ("use_rollout_log_probs", False),
            ("ppo_mini_batch_size", 1),
            ("ppo_micro_batch_size_per_gpu", 2),
            ("use_kl_loss", True),
            ("entropy_coeff", 0.1),
        ]:
            actor = Actor()
            actor.config[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_actor(actor.config, 2, 1)
        with self.assertRaises(ValueError):
            validate_actor(Actor().config, 2, 2)

    def test_no_hyperparameter_recipe_is_invented(self):
        keys = actor_overrides(2)
        self.assertFalse(any("optim" in k or "lora" in k or "max_response" in k for k in keys))


if __name__ == "__main__":
    unittest.main()
