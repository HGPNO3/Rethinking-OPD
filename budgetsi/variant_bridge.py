"""Project-side inputs for unchanged upstream OPD reward/advantage/actor paths.

Baseline delegates exactly to top16.py. Teacher supports use the teacher's own
reference-conditioned prefix; all actor tensors use student-visible prefixes.
"""

import torch
from budgetsi.model_runtime import text_config

from budgetsi import top16
from budgetsi.variant_spec import DEFAULT, get_variant


def _placeholder(action):
    n = len(action.target_ids)
    return {
        "binding": top16._binding(action),
        "ids": torch.arange(16).expand(n, -1),
        "student": torch.zeros(n, 16),
        "teacher": torch.zeros(n, 16),
        "sampled": torch.zeros(n),
        "teacher_temperature": 1.0,
    }


def _check_variant(variant):
    if get_variant(variant.name) != variant:
        raise ValueError("Variant differs from registered contract")


def _inputs(actions, snapshot_id, eos_ids, pad_id, variant):
    tensors, meta = top16.prepare_batch(
        actions,
        [_placeholder(a) for a in actions],
        snapshot_id=snapshot_id,
        eos_ids=eos_ids,
        pad_id=pad_id,
    )
    meta.update(
        top_k=variant.k,
        log_prob_top_k=variant.k,
        top_k_strategy=variant.strategy,
        reward_weight_mode=variant.weight,
        opd_variant=variant.contract(),
    )
    if variant.k == 0:
        # Placeholders above are used solely for common input validation/packing.
        # No Top-K model calculation or candidate supervision occurs in this path.
        for key in [
            "student_top_k_ids",
            "student_top_k_log_probs",
            "teacher_on_student_log_probs",
        ]:
            del tensors[key]
    return tensors, meta


@torch.no_grad()
def score_with_actor(
    worker, teacher, actions, *, variant=DEFAULT, snapshot_id, eos_ids, pad_id
):
    _check_variant(variant)
    if variant == DEFAULT:
        return top16.score_with_actor(
            worker,
            teacher,
            actions,
            snapshot_id=snapshot_id,
            eos_ids=eos_ids,
            pad_id=pad_id,
        )
    from verl import DataProto

    if not isinstance(worker, top16.LocalActorService):
        raise TypeError("The verified local upstream actor service is required")
    tensors, meta = _inputs(actions, snapshot_id, eos_ids, pad_id, variant)
    device = next(worker.actor.actor_module.parameters()).device
    data = DataProto.from_dict(
        tensors={k: v.to(device) for k, v in tensors.items()}, meta_info=meta
    )
    worker._check(data)
    sampled, _, ids, student_topk = worker.actor.compute_log_prob(
        data, calculate_entropy=False
    )
    revision = worker.revision()
    results = []
    remote = hasattr(teacher, "score_support")
    mode = None if remote else teacher.training
    if not remote:
        teacher.eval()
    try:
        for i, action in enumerate(actions):
            n = len(action.target_ids)
            receipt = {
                "binding": top16._binding(action),
                "actor_revision": revision,
                "variant": variant.contract(),
                "sampled": sampled[i, :n].cpu(),
                "teacher_temperature": 1.0,
            }
            si = ids[i, :n] if variant.k else None
            if remote:
                receipt.update(teacher.score_support(action, si, variant.k))
            else:
                q = (
                    top16.response_logits(
                        teacher, action.teacher_prompt_ids, action.target_ids
                    )
                    .float()
                    .log_softmax(-1)
                )
                if q.shape[-1] != text_config(worker.actor.actor_module).vocab_size:
                    raise ValueError("Vocabulary mismatch")
                if variant.k:
                    tq, ti = q.topk(variant.k, dim=-1)
                    receipt.update(
                        teacher=q.gather(-1, si.to(q.device)).cpu(),
                        teacher_ids=ti.cpu(),
                        teacher_topk=tq.cpu(),
                    )
                else:
                    target = torch.tensor(action.target_ids, device=q.device)
                    receipt["teacher_sampled"] = (
                        q.gather(-1, target[:, None]).squeeze(-1).cpu()
                    )
            if variant.k:
                receipt.update(ids=si.cpu(), student=student_topk[i, :n].cpu())
            results.append(receipt)
    finally:
        if not remote:
            teacher.train(mode)
    return results


def _logprob(value, shape):
    if (
        value.shape != shape
        or not value.is_floating_point()
        or not torch.isfinite(value).all()
        or (value > 0).any()
    ):
        raise ValueError("Invalid log-probability tensor")


def prepare_batch(actions, scores, *, variant=DEFAULT, snapshot_id, eos_ids, pad_id):
    _check_variant(variant)
    if variant == DEFAULT:
        if any("variant" in s for s in scores):
            raise ValueError("Variant scores cannot enter legacy baseline")
        return top16.prepare_batch(
            actions, scores, snapshot_id=snapshot_id, eos_ids=eos_ids, pad_id=pad_id
        )
    if not actions or len(actions) != len(scores):
        raise ValueError("One score per action required")
    for a, s in zip(actions, scores, strict=True):
        if (
            s.get("variant") != variant.contract()
            or s.get("binding") != top16._binding(a)
            or s.get("teacher_temperature") != 1.0
        ):
            raise ValueError("Score/variant/context binding mismatch")
        _logprob(s["sampled"], (len(a.target_ids),))
        if s["sampled"].dtype != scores[0]["sampled"].dtype:
            raise ValueError("Mixed sampled log-probability dtypes")
    if variant.k:
        tensors, meta = top16.prepare_batch(
            actions, scores, snapshot_id=snapshot_id, eos_ids=eos_ids, pad_id=pad_id
        )
        b, t, k = tensors["student_top_k_ids"].shape
        tensors["teacher_top_k_ids"] = torch.zeros(b, t, k, dtype=torch.long)
        tensors["teacher_top_k_log_probs"] = torch.zeros(
            b, t, k, dtype=scores[0]["teacher_topk"].dtype
        )
        for i, (a, s) in enumerate(zip(actions, scores, strict=True)):
            n = len(a.target_ids)
            ids = s["teacher_ids"]
            if (
                ids.shape != (n, k)
                or ids.dtype != torch.long
                or (ids < 0).any()
                or any(len(set(row.tolist())) != k for row in ids)
            ):
                raise ValueError("Invalid teacher support")
            _logprob(s["teacher_topk"], (n, k))
            if s["teacher_topk"].dtype != scores[0]["teacher_topk"].dtype:
                raise ValueError("Mixed teacher score dtypes")
            tensors["teacher_top_k_ids"][i, :n] = ids
            tensors["teacher_top_k_log_probs"][i, :n] = s["teacher_topk"].detach()
        # Separate orientations: student-in-teacher versus teacher-in-student.
        matches = tensors["student_top_k_ids"].unsqueeze(-1) == tensors[
            "teacher_top_k_ids"
        ].unsqueeze(-2)
        valid = tensors["response_mask"].bool().unsqueeze(-1)
        tensors["overlap_mask"] = matches.any(-1) & valid
        tensors["teacher_in_student_mask"] = matches.any(-2) & valid
    else:
        tensors, meta = _inputs(actions, snapshot_id, eos_ids, pad_id, variant)
        shape = tensors["response_mask"].shape
        tensors["old_log_probs"] = torch.zeros(shape, dtype=scores[0]["sampled"].dtype)
        tensors["teacher_sampled_log_probs"] = torch.zeros(
            shape, dtype=scores[0]["teacher_sampled"].dtype
        )
        for i, (a, s) in enumerate(zip(actions, scores, strict=True)):
            n = len(a.target_ids)
            _logprob(s["teacher_sampled"], (n,))
            if s["teacher_sampled"].dtype != scores[0]["teacher_sampled"].dtype:
                raise ValueError("Mixed teacher score dtypes")
            tensors["old_log_probs"][i, :n] = s["sampled"].detach()
            tensors["teacher_sampled_log_probs"][i, :n] = s["teacher_sampled"].detach()
    meta.update(
        top_k=variant.k,
        log_prob_top_k=variant.k,
        top_k_strategy=variant.strategy,
        reward_weight_mode=variant.weight,
        opd_variant=variant.contract(),
    )
    return tensors, meta


def build_update_data(
    worker, actions, scores, *, variant=DEFAULT, snapshot_id, eos_ids, pad_id
):
    """Construct real DataProto using upstream reward/advantage; inspectable by tests."""
    from verl.trainer.ppo.core_algos import compute_token_reward_direct_advantage

    from verl import DataProto

    if not isinstance(worker, top16.LocalActorService):
        raise TypeError("The verified local upstream actor service is required")
    if any(s.get("actor_revision") != worker.revision() for s in scores):
        raise ValueError("Stale actor scores")
    tensors, meta = prepare_batch(
        actions,
        scores,
        variant=variant,
        snapshot_id=snapshot_id,
        eos_ids=eos_ids,
        pad_id=pad_id,
    )
    device = next(worker.actor.actor_module.parameters()).device
    data = DataProto.from_dict(
        tensors={k: v.to(device) for k, v in tensors.items()}, meta_info=meta
    )
    if variant.k:
        data = data.union(worker.compute_distillation_reward(data))
    else:
        # Exactly the arithmetic in upstream FSDP reward worker's K=0 branch
        # (fsdp_workers.py:2717-2721); no legacy loss, IS correction, or extra p weight.
        reverse_kl = (
            data.batch["old_log_probs"] - data.batch["teacher_sampled_log_probs"]
        )
        data.batch["rm_scores"] = -reverse_kl
    advantages, returns = compute_token_reward_direct_advantage(
        data.batch["rm_scores"], data.batch["response_mask"]
    )
    if not torch.isfinite(advantages).all():
        raise ValueError("Nonfinite upstream advantage")
    data.batch["advantages"], data.batch["returns"] = advantages, returns
    return data


def update_actor(
    worker,
    actions,
    scores,
    *,
    variant=DEFAULT,
    actor_config,
    snapshot_id,
    eos_ids,
    pad_id,
    world_size=1,
):
    _check_variant(variant)
    if variant == DEFAULT:
        if any("variant" in s for s in scores):
            raise ValueError("Variant scores cannot enter legacy baseline")
        return top16.update_actor(
            worker,
            actions,
            scores,
            actor_config=actor_config,
            snapshot_id=snapshot_id,
            eos_ids=eos_ids,
            pad_id=pad_id,
            world_size=world_size,
        )
    top16.validate_actor(actor_config, len(actions), world_size)
    data = build_update_data(
        worker,
        actions,
        scores,
        variant=variant,
        snapshot_id=snapshot_id,
        eos_ids=eos_ids,
        pad_id=pad_id,
    )
    return worker.update_actor(data)
