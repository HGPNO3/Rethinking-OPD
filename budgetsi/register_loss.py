"""Register through verl's existing extension API; do not patch framework files.

Import this module in the actor process before constructing/using the worker:
VERL_USE_EXTERNAL_MODULES=budgetsi.register_loss
Ensure both the repo root and the installed pinned verl package are importable.
"""

import torch

from verl.trainer.ppo.core_algos import agg_loss, register_policy_loss


@register_policy_loss("budgetsi_raw_sampled")
def compute_budgetsi_loss(
    old_log_prob,
    log_prob,
    advantages,
    response_mask,
    loss_agg_mode="seq-mean-token-sum",
    config=None,
    rollout_is_weights=None,
    format_mask=None,
):
    """Exact recorded raw-policy surrogate; old_log_prob contains BEHAVIOR logp.

    advantages already contains detached (teacher_raw - old_HF_raw) * B/T.
    The actor must retain old_log_probs and use raw temperature 1.0. This is not a
    full-trajectory KL estimator, and the samples remain selected by our collector.
    """
    if loss_agg_mode != "seq-mean-token-sum" or rollout_is_weights is not None or format_mask is not None:
        raise ValueError("Unverified loss aggregation or duplicate weighting/masking")
    if config is None or not config.use_rollout_log_probs or config.ppo_epochs != 1:
        raise ValueError("Require fixed behavior denominator and one epoch")
    if bool((response_mask.sum(-1) <= 0).any()):
        raise ValueError("Empty actions would change sequence normalization")
    valid = response_mask.bool()
    if not bool(torch.isfinite(log_prob[valid]).all() and torch.isfinite(old_log_prob[valid]).all()):
        raise ValueError("Nonfinite valid-token probability")
    # Mask before exponentiation so padded positions cannot overflow or affect loss.
    delta = torch.where(response_mask.bool(), log_prob - old_log_prob.detach(), 0.0)
    ratio = delta.exp()
    if not bool(torch.isfinite(ratio).all()):
        raise ValueError("Nonfinite importance ratio; review instead of silent clipping")
    loss = agg_loss(-ratio * advantages.detach(), response_mask, loss_agg_mode)
    return loss, {}
