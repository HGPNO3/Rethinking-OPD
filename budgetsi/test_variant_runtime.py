"""Actual installed verl actor + tiny local Transformers models, strictly CPU.

Run with CUDA_VISIBLE_DEVICES='' in the existing dependency environment.
This validates full imports and updates, not 4B/14B GPU resource feasibility.
"""

import copy
import tempfile
import unittest
from unittest.mock import patch

import torch
from transformers import GPT2Config, GPT2LMHeadModel

from budgetsi import top16
from budgetsi.social_loop import actor_config
from budgetsi.top16 import LocalActorService, SelectedAction
from budgetsi.variant_bridge import (
    build_update_data,
    prepare_batch,
    score_with_actor,
    update_actor,
)
from budgetsi.variant_spec import DEFAULT, VARIANTS


class VariantRuntime(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if torch.cuda.is_available():
            raise RuntimeError(
                "Set CUDA_VISIBLE_DEVICES='' to avoid active training GPUs"
            )
        torch.set_num_threads(2)
        from budgetsi.model_runtime import upstream_actor_class
        cls.Actor = upstream_actor_class()
        cls.folder = tempfile.TemporaryDirectory()
        torch.distributed.init_process_group(
            "gloo", init_method=f"file://{cls.folder.name}/dist", rank=0, world_size=1
        )

    @classmethod
    def tearDownClass(cls):
        torch.distributed.destroy_process_group()
        cls.folder.cleanup()

    def models(self):
        cfg = GPT2Config(
            vocab_size=64,
            n_positions=32,
            n_embd=16,
            n_layer=1,
            n_head=2,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
            eos_token_id=2,
            pad_token_id=0,
        )
        torch.manual_seed(71)
        student = GPT2LMHeadModel(cfg).eval()
        torch.manual_seed(91)
        teacher = GPT2LMHeadModel(cfg).eval().requires_grad_(False)
        actions = [
            SelectedAction(
                str(i),
                "snapshot",
                p,
                (50, 51, *p),
                y,
                "tok",
                "tok",
                "protocol",
                "reference",
                0.7,
            )
            for i, (p, y) in enumerate([((3, 4), (7, 2)), ((8,), (9, 10, 2))])
        ]
        optimizer = torch.optim.AdamW(student.parameters(), lr=1e-3, weight_decay=0.01)
        cfg = actor_config(len(actions))
        worker = LocalActorService(self.Actor(cfg, student, optimizer), 0.7)
        return student, teacher, actions, optimizer, cfg, worker

    def test_all_five_actual_scoring_reward_update_and_staleness(self):
        for variant in VARIANTS.values():
            with self.subTest(variant=variant.name):
                student, teacher, actions, optimizer, cfg, worker = self.models()
                before = copy.deepcopy(student.state_dict())
                teacher_before = copy.deepcopy(teacher.state_dict())
                with patch.object(
                    worker.actor,
                    "compute_log_prob",
                    wraps=worker.actor.compute_log_prob,
                ) as observe:
                    scores = score_with_actor(
                        worker,
                        teacher,
                        actions,
                        variant=variant,
                        snapshot_id="snapshot",
                        eos_ids={2},
                        pad_id=0,
                    )
                    self.assertEqual(
                        observe.call_args.args[0].meta_info["top_k"], variant.k
                    )
                # Explicit prefix-by-prefix teacher forwards catch prompt-offset/future leakage.
                for a, s in zip(actions, scores):
                    for t in range(len(a.target_ids)):
                        ids = torch.tensor([[*a.teacher_prompt_ids, *a.target_ids[:t]]])
                        q = teacher(ids).logits[0, -1].float().log_softmax(-1)
                        if variant.k:
                            torch.testing.assert_close(
                                s["teacher"][t], q[s["ids"][t]], atol=2e-6, rtol=1e-5
                            )
                            if variant != DEFAULT:
                                expected_q, expected_ids = q.topk(16)
                                torch.testing.assert_close(
                                    s["teacher_topk"][t],
                                    expected_q,
                                    atol=2e-6,
                                    rtol=1e-5,
                                )
                                self.assertEqual(
                                    s["teacher_ids"][t].tolist(), expected_ids.tolist()
                                )
                        else:
                            torch.testing.assert_close(
                                s["teacher_sampled"][t],
                                q[a.target_ids[t]],
                                atol=2e-6,
                                rtol=1e-5,
                            )
                data = build_update_data(
                    worker,
                    actions,
                    scores,
                    variant=variant,
                    snapshot_id="snapshot",
                    eos_ids={2},
                    pad_id=0,
                )
                adv = data.batch["advantages"]
                self.assertEqual(adv.ndim, 2 if variant.k == 0 else 3)
                if variant.strategy == "union":
                    self.assertEqual(adv.shape[-1], 32)
                self.assertTrue((adv[0, 2] == 0).all())
                self.assertFalse(adv.requires_grad)
                update_actor(
                    worker,
                    actions,
                    scores,
                    variant=variant,
                    actor_config=cfg,
                    snapshot_id="snapshot",
                    eos_ids={2},
                    pad_id=0,
                )
                self.assertTrue(
                    any(
                        not torch.equal(before[k], v)
                        for k, v in student.state_dict().items()
                    )
                )
                self.assertTrue(
                    all(torch.isfinite(v).all() for v in student.state_dict().values())
                )
                self.assertEqual(
                    {int(s["step"]) for s in optimizer.state.values()}, {1}
                )
                for k, value in teacher.state_dict().items():
                    torch.testing.assert_close(value, teacher_before[k], rtol=0, atol=0)
                with self.assertRaises(ValueError):
                    update_actor(
                        worker,
                        actions,
                        scores,
                        variant=variant,
                        actor_config=cfg,
                        snapshot_id="snapshot",
                        eos_ids={2},
                        pad_id=0,
                    )
                if variant != DEFAULT:
                    with self.assertRaises(ValueError):
                        prepare_batch(
                            actions,
                            scores,
                            variant=DEFAULT,
                            snapshot_id="snapshot",
                            eos_ids={2},
                            pad_id=0,
                        )

    def test_baseline_parameters_are_bitwise_identical_to_original_path(self):
        states = []
        for original in [True, False]:
            student, teacher, actions, _optimizer, cfg, worker = self.models()
            module = (
                top16
                if original
                else __import__("budgetsi.variant_bridge", fromlist=["update_actor"])
            )
            scores = module.score_with_actor(
                worker, teacher, actions, snapshot_id="snapshot", eos_ids={2}, pad_id=0
            )
            module.update_actor(
                worker,
                actions,
                scores,
                actor_config=cfg,
                snapshot_id="snapshot",
                eos_ids={2},
                pad_id=0,
            )
            states.append(copy.deepcopy(student.state_dict()))
        for k in states[0]:
            torch.testing.assert_close(states[0][k], states[1][k], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
