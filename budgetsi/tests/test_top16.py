import copy
import unittest
from dataclasses import replace
from types import SimpleNamespace

import torch
from omegaconf import OmegaConf

from budgetsi.audit.source_probe import function, numerical_functions
from budgetsi.top16 import (
    LocalActorService,
    SelectedAction,
    actor_overrides,
    from_collector_record,
    prepare_batch,
    response_logits,
    score_action,
    update_actor,
    validate_actor,
)


class Container:
    def __init__(self, tensors, meta_info=None):
        self.batch, self.meta_info = tensors, meta_info or {}

    @classmethod
    def from_dict(cls, tensors):
        return cls(tensors)


def upstream():
    ns = numerical_functions()
    ns.update(DataProto=Container, get_device_id=lambda: "cpu")
    fn = function(
        "verl/verl/workers/actor/dp_actor.py",
        "compute_distillation_reward",
        ns,
        True,
        parent="DataParallelPPOActor",
    )
    return ns, fn


class CausalToy(torch.nn.Module):
    def __init__(self, seed):
        super().__init__()
        torch.manual_seed(seed)
        self.embedding = torch.nn.Embedding(32, 12)
        self.head = torch.nn.Linear(12, 32)
        self.config = SimpleNamespace(vocab_size=32)

    def forward(self, input_ids, **kwargs):
        return SimpleNamespace(logits=self.head(self.embedding(input_ids).cumsum(1)))


def fixture():
    student, teacher = CausalToy(1), CausalToy(2)
    actions = [
        SelectedAction(
            str(i),
            "snapshot",
            p,
            (25, 26, *p),
            y,
            "tokenizer",
            "tokenizer",
            "prompt-v1",
            "reference",
            0.7,
        )
        for i, (p, y) in enumerate([((3, 4), (5, 2)), ((6,), (7, 8, 2))])
    ]
    scores = [
        score_action(student, teacher, a, snapshot_id="snapshot", eos_ids={2})
        for a in actions
    ]
    return student, teacher, actions, scores


def prepare(actions, scores):
    return prepare_batch(actions, scores, snapshot_id="snapshot", eos_ids={2}, pad_id=0)


class Top16(unittest.TestCase):
    def test_collector_bridge_rerenders_original_context_and_discards_cached_scores(
        self,
    ):
        class Tokenizer:
            def apply_chat_template(self, messages, **kwargs):
                return messages

            def get_vocab(self):
                return {str(i): i for i in range(32)}

        record = {
            "id": "real-record-shape-fixture",
            "snapshot_id": "snapshot",
            "visible_messages": [3, 4],
            "teacher_messages": [25, 26, 3, 4],
            "original": {
                "prompt_token_ids": [3, 4],
                "generated_token_ids": [5, 2],
                "behavior_temperature": 0.7,
            },
            "student_prefix": [3, 4],
            "target_ids": [5, 2],
            "loss_mask": [True, True],
            "teacher_score": {"prompt_token_ids": [25, 26, 3, 4]},
            "prompt_binding": "bound-protocol",
            "reference_action_sha256": "reference",
            "teacher_logprobs": [-1.0, -2.0],
        }
        calls = []
        kwargs = {
            "student_tokenizer": Tokenizer(),
            "teacher_tokenizer": Tokenizer(),
            "validate_record": lambda r: calls.append(r["id"]),
            "snapshot_id": "snapshot",
            "eos_ids": {2},
        }
        a = from_collector_record(record, **kwargs)
        record["teacher_logprobs"] = [-5.0, -6.0]
        self.assertEqual(a, from_collector_record(record, **kwargs))
        self.assertEqual(len(calls), 2)
        record["visible_messages"] = [9, 4]
        with self.assertRaisesRegex(ValueError, "Rendered student"):
            from_collector_record(record, **kwargs)

    def test_teacher_scores_same_ids_at_different_prompt_offsets(self):
        s, teacher, actions, scores = fixture()
        for a, receipt in zip(actions, scores):
            for t in range(len(a.target_ids)):
                # Independent prefix-by-prefix forwards catch target shift and future leakage.
                ids = torch.tensor([[*a.teacher_prompt_ids, *a.target_ids[:t]]])
                q = teacher(ids).logits[0, -1].log_softmax(-1)
                torch.testing.assert_close(receipt["teacher"][t], q[receipt["ids"][t]])
                pids = torch.tensor([[*a.student_prompt_ids, *a.target_ids[:t]]])
                p = (s(pids).logits[0, -1] / 0.7).log_softmax(-1)
                torch.testing.assert_close(receipt["student"][t], p[receipt["ids"][t]])

    def test_teacher_reference_changes_scores_not_student_input(self):
        s, t, actions, scores = fixture()
        changed = [
            replace(
                a,
                teacher_prompt_ids=(28, *a.teacher_prompt_ids),
                reference_hash="changed",
            )
            for a in actions
        ]
        new = [
            score_action(s, t, a, snapshot_id="snapshot", eos_ids={2}) for a in changed
        ]
        a, _ = prepare(actions, scores)
        b, _ = prepare(changed, new)
        for key in (
            "input_ids",
            "responses",
            "student_top_k_ids",
            "student_top_k_log_probs",
        ):
            torch.testing.assert_close(a[key], b[key])
        self.assertFalse(
            torch.allclose(
                a["teacher_on_student_log_probs"], b["teacher_on_student_log_probs"]
            )
        )
        self.assertEqual(a["response_mask"].tolist(), [[1, 1, 0], [1, 1, 1]])
        self.assertFalse((a["input_ids"] == 25).any())

    def test_actual_upstream_reward_advantage_and_gradient(self):
        student, teacher, actions, scores = fixture()
        tensors, meta = prepare(actions, scores)
        ns, reward_fn = upstream()
        reward = reward_fn(
            SimpleNamespace(actor_module=student), Container(tensors, meta)
        ).batch["rm_scores"]
        expected = (
            tensors["teacher_on_student_log_probs"] - tensors["student_top_k_log_probs"]
        ) * tensors["student_top_k_log_probs"].softmax(-1)
        torch.testing.assert_close(reward, expected)
        adv, _ = ns["compute_token_reward_direct_advantage"](
            reward, tensors["response_mask"]
        )
        self.assertTrue((adv[0, 2] == 0).all())
        self.assertFalse(adv.requires_grad)
        cfg = OmegaConf.create(
            {
                "clip_ratio": 0.2,
                "clip_ratio_low": 0.2,
                "clip_ratio_high": 0.2,
                "clip_ratio_c": 3.0,
            }
        )
        # Independent surrogate at the same policy point must have identical gradients.
        z = torch.randn(2, 3, 32, requires_grad=True)
        lp = z.log_softmax(-1).gather(-1, tensors["student_top_k_ids"])
        actual = ns["compute_policy_loss_vanilla"](
            lp.detach(), lp, adv, tensors["response_mask"], config=cfg
        )[0]
        grad = torch.autograd.grad(actual, z)[0]
        lp = z.log_softmax(-1).gather(-1, tensors["student_top_k_ids"])
        expected_loss = (
            -(adv * lp * tensors["response_mask"][..., None]).sum()
            / tensors["response_mask"].sum()
        )
        torch.testing.assert_close(grad, torch.autograd.grad(expected_loss, z)[0])
        self.assertTrue((grad[0, 2] == 0).all())
        self.assertIsNone(next(teacher.parameters()).grad)

    def test_actual_model_parameter_update_and_teacher_frozen(self):
        student, teacher, actions, scores = fixture()
        ns, reward_fn = upstream()
        tensors, meta = prepare(actions, scores)
        r = reward_fn(
            SimpleNamespace(actor_module=student), Container(tensors, meta)
        ).batch["rm_scores"]
        adv, _ = ns["compute_token_reward_direct_advantage"](
            r, tensors["response_mask"]
        )
        before = copy.deepcopy(student.state_dict())
        teacher_before = copy.deepcopy(teacher.state_dict())
        optim = torch.optim.AdamW(student.parameters(), lr=1e-4)
        cfg = OmegaConf.create(
            {
                "clip_ratio": 0.2,
                "clip_ratio_low": 0.2,
                "clip_ratio_high": 0.2,
                "clip_ratio_c": 3.0,
            }
        )
        for i, a in enumerate(actions):
            n = len(a.target_ids)
            lp = (
                response_logits(student, a.student_prompt_ids, a.target_ids) / 0.7
            ).log_softmax(-1)
            lp = lp.gather(-1, scores[i]["ids"])[None]
            loss = ns["compute_policy_loss_vanilla"](
                lp.detach(), lp, adv[i : i + 1, :n], torch.ones(1, n), config=cfg
            )[0]
            (loss / len(actions)).backward()  # upstream static microbatch=1 semantics
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in student.parameters()))
        optim.step()
        self.assertTrue(
            any(not torch.equal(before[k], v) for k, v in student.state_dict().items())
        )
        for k, v in teacher.state_dict().items():
            torch.testing.assert_close(v, teacher_before[k])

    def test_reject_stale_scores_changed_targets_and_mismatched_vocab(self):
        _, _, actions, scores = fixture()
        changes = [
            {"snapshot_id": "old"},
            {"target_ids": (8, 2)},
            {"student_tokenizer_hash": "wrong"},
            {"teacher_prompt_ids": (9,)},
            {"sampling_temperature": 1.0},
            {"target_ids": (2, 5)},
        ]
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ValueError):
                prepare([replace(actions[0], **change), actions[1]], scores)
        corrupted = copy.deepcopy(scores)
        corrupted[0]["teacher"][0, 0] = float("nan")
        with self.assertRaises(ValueError):
            prepare(actions, corrupted)

    def test_contract_uses_vanilla_and_rejects_legacy(self):
        cfg = OmegaConf.from_dotlist(
            [
                f"{k}={str(v).lower() if isinstance(v, bool) else v}"
                for k, v in actor_overrides(2).items()
            ]
        )
        validate_actor(cfg, 2)
        cfg.policy_loss.loss_mode = "budgetsi_raw_sampled"
        with self.assertRaises(ValueError):
            validate_actor(cfg, 2)

    def test_live_revision_rejects_scores_after_parameter_change(self):
        student, _, actions, scores = fixture()
        service = LocalActorService(SimpleNamespace(actor_module=student), 0.7)
        revision = service.revision()
        for score in scores:
            score["actor_revision"] = revision
        with torch.no_grad():
            next(student.parameters()).add_(0.01)
        self.assertNotEqual(service.revision(), revision)
        cfg = OmegaConf.from_dotlist(
            [
                f"{k}={str(v).lower() if isinstance(v, bool) else v}"
                for k, v in actor_overrides(2).items()
            ]
        )
        with self.assertRaisesRegex(ValueError, "live actor revision"):
            update_actor(
                service,
                actions,
                scores,
                actor_config=cfg,
                snapshot_id="snapshot",
                eos_ids={2},
                pad_id=0,
            )

    def test_actor_and_reward_receive_their_respective_k_keys(self):
        _, _, actions, scores = fixture()
        _, meta = prepare(actions, scores)
        self.assertEqual(
            meta["top_k"], 16
        )  # actual actor.compute_log_prob reads this key
        self.assertEqual(meta["log_prob_top_k"], 16)  # reward method reads this key


if __name__ == "__main__":
    unittest.main()
