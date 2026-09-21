import unittest

import torch
from omegaconf import OmegaConf

from budgetsi.audit.source_probe import numerical_functions

CFG = OmegaConf.create(dict(clip_ratio=0.2, clip_ratio_low=0.2, clip_ratio_high=0.2, clip_ratio_c=3.0))
F = numerical_functions()


def upstream_loss(logp, old, q, mask, weights=None):
    advantages, _ = F["compute_token_reward_direct_advantage"](q - old, mask)
    return F["compute_policy_loss_vanilla"](
        logp.detach(), logp, advantages, mask, config=CFG, rollout_is_weights=weights
    )[0]


class Numerics(unittest.TestCase):
    def test_raw_surrogate_matches_upstream_first_step_with_explicit_is(self):
        z = torch.tensor(
            [[[0.8, -0.4, 0.2], [0.1, 1.2, -0.3]]],
            dtype=torch.float64,
            requires_grad=True,
        )
        old = z.detach().log_softmax(-1)[..., 0]
        behavior = (z.detach() / 0.7).log_softmax(-1)[..., 0]
        q = torch.tensor([[-1.2, -0.5]], dtype=torch.float64)
        mask = torch.ones_like(q)
        logp = z.log_softmax(-1)[..., 0]
        legacy = -(torch.exp(logp - behavior) * (q - old)).mean()
        expected = torch.autograd.grad(legacy, z)[0]
        loss = upstream_loss(z.log_softmax(-1)[..., 0], old, q, mask, (old - behavior).exp())
        actual = torch.autograd.grad(loss, z)[0]
        torch.testing.assert_close(actual, expected, atol=1e-8, rtol=1e-8)

    def test_same_point_seven_does_not_mean_same_training_objective(self):
        z = torch.tensor([[[1.2, -0.2, 0.3]]], dtype=torch.float64, requires_grad=True)
        raw = z.log_softmax(-1)[..., 0]
        old = raw.detach()
        behavior = (z.detach() / 0.7).log_softmax(-1)[..., 0]
        q = torch.tensor([[-1.5]], dtype=torch.float64)
        legacy = -(torch.exp(raw - behavior) * (q - old)).mean()
        g1 = torch.autograd.grad(legacy, z)[0]
        tempered = (z / 0.7).log_softmax(-1)[..., 0]
        candidate = upstream_loss(tempered, behavior, q, torch.ones_like(q))
        g2 = torch.autograd.grad(candidate, z)[0]
        self.assertGreater((g1 - g2).abs().max().item(), 0.05)

    def test_microbatch_token_mean_is_not_global_token_mean(self):
        values = torch.tensor([[1.0, 0.0, 0.0], [3.0, 3.0, 3.0]])
        mask = torch.tensor([[1.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        whole = F["agg_loss"](values, mask, "token-mean")
        split = sum(F["agg_loss"](v[None], m[None], "token-mean") / 2 for v, m in zip(values, mask, strict=True))
        self.assertAlmostEqual(whole.item(), 2.5)
        self.assertAlmostEqual(split.item(), 2.0)

    def test_eos_is_included_padding_excluded_and_teacher_detached(self):
        logp = torch.tensor([[-0.5, -0.4, -0.7]], requires_grad=True)
        old = logp.detach()
        q = torch.tensor([[-0.2, -0.1, -0.3]], requires_grad=True)
        loss = upstream_loss(logp, old, q, torch.tensor([[1.0, 1.0, 0.0]]))
        loss.backward()
        self.assertNotEqual(logp.grad[0, 1].item(), 0)  # second token stands for EOS
        self.assertEqual(logp.grad[0, 2].item(), 0)
        self.assertIsNone(q.grad)

    def test_multiple_update_ppo_clipping_cannot_be_called_equivalent(self):
        old = torch.tensor([[-1.0]])
        new = torch.tensor([[-0.5]], requires_grad=True)
        adv = torch.tensor([[1.0]])
        loss, _ = F["compute_policy_loss_vanilla"](old, new, adv, torch.ones_like(adv), config=CFG)
        grad = torch.autograd.grad(loss, new)[0]
        self.assertEqual(grad.item(), 0.0)
        unclipped = -(new - old).exp()
        self.assertNotEqual(torch.autograd.grad(unclipped, new)[0].item(), 0.0)


if __name__ == "__main__":
    unittest.main()
