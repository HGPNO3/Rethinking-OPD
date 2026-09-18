"""Adapt validated BudgetSI records to the unchanged verl actor update interface.

No models, services, network calls, or optimizer steps are started here. HF scores
must be freshly computed by the training actor at the named frozen snapshot.
"""

import hashlib
import json
import math
from dataclasses import dataclass

import torch

LOSS_NAME = "budgetsi_raw_sampled"


def _probabilities(values, length, label):
    if len(values) != length or any(not math.isfinite(v) or v > 0 for v in values):
        raise ValueError(f"Invalid {label} probabilities")
    return torch.tensor(values, dtype=torch.float32)


def _ids(values, label):
    if not values or any(type(v) is not int or v < 0 for v in values):
        raise ValueError(f"Invalid {label} token IDs")


@dataclass
class PreparedUpdate:
    tensors: dict
    metadata: dict
    node_ids: list[str]
    contract: dict

    def as_dataproto(self, *, actor_config, world_size):
        """Requires an installed pinned verl; does not invoke update_actor."""
        validate_actor(actor_config, len(self.node_ids), world_size)
        from verl import DataProto

        return DataProto.from_dict(tensors=self.tensors, meta_info=self.metadata)


def actor_overrides(batch_size):
    """Update-service contract, not a new baseline or a complete run configuration."""
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be a positive node count")
    return {
        "actor_rollout_ref.actor.policy_loss.loss_mode": LOSS_NAME,
        "actor_rollout_ref.actor.loss_agg_mode": "seq-mean-token-sum",
        "actor_rollout_ref.actor.ppo_mini_batch_size": batch_size,
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu": 1,
        "actor_rollout_ref.actor.ppo_epochs": 1,
        "actor_rollout_ref.actor.use_rollout_log_probs": True,
        "actor_rollout_ref.actor.use_kl_loss": False,
        "actor_rollout_ref.actor.entropy_coeff": 0.0,
        "actor_rollout_ref.actor.ulysses_sequence_parallel_size": 1,
        # Worker initialization multiplies configured minibatch size by rollout.n.
        "actor_rollout_ref.rollout.n": 1,
    }


def validate_actor(actor_config, batch_size, world_size):
    """Validate actual post-initialization actor settings, not just shell values."""
    if world_size != 1:
        raise ValueError("Only the single-rank update contract has been verified")
    expected = actor_overrides(batch_size)
    for key, value in expected.items():
        prefix = "actor_rollout_ref.actor."
        if not key.startswith(prefix):
            continue
        actual = actor_config
        for part in key[len(prefix) :].split("."):
            actual = actual[part]
        if actual != value:
            raise ValueError(f"Unverified actor setting: {key}={actual!r}, expected {value!r}")


def build_update(records, hf_scores, *, snapshot_id, prompt_binding, pad_id, eos_ids):
    """Build one whole frozen-policy minibatch from original student actions.

    The collector owns teacher selection and target-time visibility. This bridge
    checks record/probability identity and never forwards reference/partner text to
    the actor. Historical serving raw scores are not substituted for HF scores.
    """
    if not records or len(records) != len(hf_scores):
        raise ValueError("A nonempty batch needs one fresh HF score receipt per node")
    if type(pad_id) is not int or pad_id < 0 or not eos_ids:
        raise ValueError("Explicit tokenizer padding/EOS contract required")
    rows = []
    seen = set()
    for r, hf in zip(records, hf_scores, strict=True):
        if r["id"] in seen:
            raise ValueError("Duplicate selected node")
        seen.add(r["id"])
        if r["snapshot_id"] != snapshot_id or hf["snapshot_id"] != snapshot_id:
            raise ValueError("Stale probability snapshot")
        if r["prompt_binding"] != prompt_binding:
            raise ValueError("Incompatible teacher prompt protocol")
        prefix, target = r["student_prefix"], r["target_ids"]
        _ids(prefix, "prefix")
        _ids(target, "target")
        if target[-1] not in eos_ids or any(v in eos_ids for v in target[:-1]):
            raise ValueError("Original action must end at its first EOS")
        if prefix != r["original"]["prompt_token_ids"] or target != r["original"]["generated_token_ids"]:
            raise ValueError("Student input/target must be the original rollout tokens")
        if r["loss_mask"] != [True] * len(target):
            raise ValueError("Only complete original actions including EOS are supported")
        if hf["prompt_ids"] != prefix or hf["target_ids"] != target or hf["backend"] != "training_actor_raw":
            raise ValueError("HF score receipt does not match original actor input/target")
        if r["teacher_score"]["target_ids"] != target:
            raise ValueError("Teacher must rescore the original target, not its own candidate")
        if r["teacher_logprobs"] != r["teacher_score"]["raw_logprobs"]:
            raise ValueError("Teacher probability receipt mismatch")
        if r["behavior_logprobs"] != r["original"]["behavior_logprobs"]:
            raise ValueError("Behavior probability receipt mismatch")
        if r["original"].get("behavior_temperature") != 0.7:
            raise ValueError("Unexpected behavior temperature for this recorded protocol")
        if r["original"].get("behavior_probability_mode") != "processed_logprobs":
            raise ValueError("Behavior probabilities must represent actual sampling")
        n = len(target)
        behavior = _probabilities(r["behavior_logprobs"], n, "behavior")
        teacher = _probabilities(r["teacher_logprobs"], n, "teacher raw")
        old = _probabilities(hf["raw_logprobs"], n, "training actor raw")
        rows.append((prefix, target, behavior, teacher, old))
    b, total = len(rows), sum(len(row[1]) for row in rows)
    pmax, tmax = max(len(row[0]) for row in rows), max(len(row[1]) for row in rows)
    inputs = torch.full((b, pmax + tmax), pad_id, dtype=torch.long)
    responses = torch.full((b, tmax), pad_id, dtype=torch.long)
    attention = torch.zeros_like(inputs)
    mask = torch.zeros((b, tmax), dtype=torch.long)
    behavior = torch.zeros((b, tmax))
    advantages = torch.zeros((b, tmax))
    for i, (prefix, target, mu, q, old) in enumerate(rows):
        n = len(target)
        inputs[i, pmax - len(prefix) : pmax] = torch.tensor(prefix)
        inputs[i, pmax : pmax + n] = torch.tensor(target)
        responses[i, :n] = torch.tensor(target)
        attention[i, pmax - len(prefix) : pmax + n] = 1
        mask[i, :n] = 1
        behavior[i, :n] = mu
        # seq-mean-token-sum + microbatch sequence weighting gives sum(loss)/T.
        advantages[i, :n] = (q - old) * (b / total)
    positions = (attention.cumsum(-1) - 1).clamp_min(0)
    tensors = dict(
        input_ids=inputs,
        responses=responses,
        attention_mask=attention,
        position_ids=positions,
        response_mask=mask,
        old_log_probs=behavior,
        advantages=advantages.detach(),
    )
    receipt = dict(
        snapshot_id=snapshot_id,
        prompt_binding=prompt_binding,
        nodes=b,
        target_tokens=total,
        teacher_reference_in_actor_input=False,
        record_ids_sha256=hashlib.sha256(json.dumps([r["id"] for r in records]).encode()).hexdigest(),
    )
    metadata = dict(temperature=1.0, global_token_num=attention.sum(-1).tolist(), budgetsi=receipt)
    return PreparedUpdate(tensors, metadata, [r["id"] for r in records], actor_overrides(b))
