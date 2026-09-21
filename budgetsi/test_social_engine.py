"""Exercise actual Transformers generation/scoring with tiny local CPU weights.

No pretrained weights or network; this isolates sampling fallback regressions.
Run in the GPU acceptance dependency environment (fork_rng sees two GPUs).
"""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import GPT2Config, GPT2LMHeadModel

from budgetsi.social_loop import Engine


class EngineTest(unittest.TestCase):
    def test_actual_sampling_ignores_pretrained_nucleus_default(self):
        torch.manual_seed(5)
        model = GPT2LMHeadModel(
            GPT2Config(
                vocab_size=17,
                n_positions=16,
                n_embd=16,
                n_layer=1,
                n_head=2,
                bos_token_id=1,
                eos_token_id=16,
                pad_token_id=0,
            )
        ).eval()
        model.generation_config.top_p = 0.95
        model.generation_config.top_k = 20
        observed = []
        original = model._prepare_generation_config

        def prepare(*args, **kwargs):
            cfg, extras = original(*args, **kwargs)
            observed.append((cfg.top_p, cfg.top_k, cfg.temperature))
            return cfg, extras

        model._prepare_generation_config = prepare
        with tempfile.TemporaryDirectory() as folder:
            engine = Engine(
                model,
                model,
                {"student": SimpleNamespace(pad_token_id=0)},
                7,
                Path(folder),
            )
            with self.assertRaises(RuntimeError):
                engine.call({"model": "student"})
            engine.accepting = True
            engine.snapshot = "tiny-test"
            prompt = [1, 3, 4]
            result = engine.call(
                {
                    "model": "student",
                    "prompt": prompt,
                    "max_tokens": 4,
                    "temperature": 0.7,
                    "top_p": 1.0,
                    "top_k": -1,
                    "seed": 123,
                    "stop_token_ids": [16],
                }
            )
            self.assertEqual(observed, [(1.0, 0, 0.7)])
            choice = result["choices"][0]
            target = choice["token_ids"]
            for i, tid in enumerate(target):
                with torch.no_grad():
                    logits = (
                        model(torch.tensor([prompt + target[:i]])).logits[0, -1].float()
                    )
                expected = (logits / 0.7).log_softmax(-1)[tid].item()
                self.assertAlmostEqual(
                    choice["logprobs"]["token_logprobs"][i], expected, places=5
                )
            # Raw scoring must use temperature 1 and the original target prefix.
            engine.max_context = 16
            score = engine.call(
                {
                    "model": "student",
                    "operation": "score",
                    "prompt_ids": prompt,
                    "target_ids": target,
                }
            )
            for i, tid in enumerate(target):
                with torch.no_grad():
                    expected = (
                        model(torch.tensor([prompt + target[:i]]))
                        .logits[0, -1]
                        .float()
                        .log_softmax(-1)[tid]
                        .item()
                    )
                self.assertAlmostEqual(score["raw_logprobs"][i], expected, places=5)


if __name__ == "__main__":
    unittest.main()
