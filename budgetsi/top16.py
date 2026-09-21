"""Student Top-16 project bridge. No replacement reward or policy loss.

Collector supplies separately rendered student/teacher prompts. Teacher prompt
construction and selection are owned by the existing social collector, not here.
Old sampled-token probability caches are deliberately not accepted.
"""

import hashlib
import json
import math
from dataclasses import dataclass

import torch
from budgetsi.model_runtime import text_config


@dataclass(frozen=True)
class SelectedAction:
    node_id: str
    snapshot_id: str
    student_prompt_ids: tuple[int, ...]
    teacher_prompt_ids: tuple[int, ...]
    target_ids: tuple[int, ...]
    student_tokenizer_hash: str
    teacher_tokenizer_hash: str
    prompt_binding: str
    reference_hash: str
    sampling_temperature: float

    def validate(self, snapshot_id, eos_ids):
        if self.snapshot_id != snapshot_id:
            raise ValueError("Stale student snapshot")
        if not all(
            (
                self.node_id,
                self.prompt_binding,
                self.reference_hash,
                self.student_tokenizer_hash,
                self.teacher_tokenizer_hash,
            )
        ):
            raise ValueError("Missing selection/tokenizer/prompt binding")
        if self.student_tokenizer_hash != self.teacher_tokenizer_hash:
            raise ValueError("Teacher and student token ID mappings must match")
        for ids in (self.student_prompt_ids, self.teacher_prompt_ids, self.target_ids):
            if not ids or any(type(v) is not int or v < 0 for v in ids):
                raise ValueError("Invalid prompt/target IDs")
        if (
            not eos_ids
            or self.target_ids[-1] not in eos_ids
            or any(v in eos_ids for v in self.target_ids[:-1])
        ):
            raise ValueError("Complete original action with first EOS required")
        if (
            not math.isfinite(self.sampling_temperature)
            or self.sampling_temperature <= 0
        ):
            raise ValueError("Invalid sampling temperature")


def _binding(action):
    return hashlib.sha256(
        json.dumps(action.__dict__, sort_keys=True).encode()
    ).hexdigest()


def from_collector_record(
    record,
    *,
    student_tokenizer,
    teacher_tokenizer,
    validate_record,
    snapshot_id,
    eos_ids,
):
    """Convert the existing reference-OPD collector record; discard old scores.

    The caller provides the matching collector's validate_reference_record.
    We additionally re-render both contexts using the actual tokenizers. This
    validates serialization, not the truth of the upstream IG selection.
    """
    validate_record(record)
    original = record["original"]
    student_ids = student_tokenizer.apply_chat_template(
        record["visible_messages"],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_dict=False,
    )
    teacher_ids = teacher_tokenizer.apply_chat_template(
        record["teacher_messages"],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_dict=False,
    )
    if (
        list(student_ids) != original["prompt_token_ids"]
        or list(student_ids) != record["student_prefix"]
    ):
        raise ValueError("Rendered student prompt differs from original rollout")
    if list(teacher_ids) != record["teacher_score"]["prompt_token_ids"]:
        raise ValueError("Rendered teacher reference prompt differs from collector")
    if record["target_ids"] != original["generated_token_ids"] or record[
        "loss_mask"
    ] != [True] * len(record["target_ids"]):
        raise ValueError("Only complete original student targets are supported")

    def fingerprint(tok):
        return hashlib.sha256(
            json.dumps(tok.get_vocab(), sort_keys=True).encode()
        ).hexdigest()

    action = SelectedAction(
        record["id"],
        record["snapshot_id"],
        tuple(student_ids),
        tuple(teacher_ids),
        tuple(record["target_ids"]),
        fingerprint(student_tokenizer),
        fingerprint(teacher_tokenizer),
        record["prompt_binding"],
        record["reference_action_sha256"],
        original["behavior_temperature"],
    )
    action.validate(snapshot_id, eos_ids)
    return action


def response_logits(model, prompt_ids, target_ids):
    """Teacher-forced original targets, using this model's own prompt length.

    The logit before target[0] predicts target[0]. No candidate continuation or
    post-action partner reply is appended. One action per forward limits batch
    memory; Qwen3.5 materializes only the target prediction positions.
    """
    device = next(model.parameters()).device
    ids = torch.tensor([(*prompt_ids, *target_ids)], device=device)
    start = len(prompt_ids) - 1
    if getattr(model.config, 'model_type', '') in {'qwen3_5', 'qwen3_5_text'}:
        # Do not allocate a full prefix-length x 248K vocabulary tensor merely
        # to score the target. This leaves the probability calculation unchanged.
        # An integer slice works even when Accelerate places the final hidden
        # states on another GPU. A tensor of positions is moved to the input
        # GPU by its hooks and cannot index those remote hidden states.
        return model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False,
                     logits_to_keep=len(target_ids) + 1).logits[0, :-1]
    output = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
    return output.logits[0, start : start + len(target_ids)]


@torch.no_grad()
def score_action(
    student, teacher, action, *, snapshot_id, eos_ids, teacher_temperature=1.0
):
    """Fresh scores on the two explicitly bound input contexts; all K IDs fixed.

    Like upstream, actor scoring uses rollout temperature; teacher uses its
    separately supplied temperature. No old raw-temperature IS correction.
    Caller must bind the actual loaded weights to snapshot_id and verify the
    tokenizer fingerprints. A string receipt alone does not prove provenance.
    """
    action.validate(snapshot_id, eos_ids)
    if not math.isfinite(teacher_temperature) or teacher_temperature <= 0:
        raise ValueError("Invalid teacher temperature")
    modes = student.training, teacher.training
    student.eval()
    teacher.eval()
    try:
        p = (
            response_logits(
                student, action.student_prompt_ids, action.target_ids
            ).float()
            / action.sampling_temperature
        ).log_softmax(-1)
        if p.shape[-1] < 16:
            raise ValueError("Student vocabulary smaller than K=16")
        p16, ids = p.topk(16, dim=-1)
        sampled = p.gather(
            -1, torch.tensor(action.target_ids, device=p.device)[:, None]
        ).squeeze(-1)
        del p
        q = (
            response_logits(
                teacher, action.teacher_prompt_ids, action.target_ids
            ).float()
            / teacher_temperature
        ).log_softmax(-1)
        # vocab-size equality is necessary but fingerprint equality above is also required.
        if q.shape[-1] != text_config(student).vocab_size:
            raise ValueError("Vocabulary size mismatch")
        q16 = q.gather(-1, ids.to(q.device))
        for value in (p16, q16, sampled):
            if not torch.isfinite(value).all():
                raise ValueError("Nonfinite model probabilities")
        return {
            "binding": _binding(action),
            "ids": ids.cpu(),
            "student": p16.cpu(),
            "teacher": q16.cpu(),
            "sampled": sampled.cpu(),
            "teacher_temperature": teacher_temperature,
        }
    finally:
        student.train(modes[0])
        teacher.train(modes[1])


def actor_overrides(batch_size):
    """Single student rank, one fixed minibatch; upstream vanilla update semantics."""
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("Nonempty action batch required")
    return {
        "policy_loss.loss_mode": "vanilla",
        "loss_agg_mode": "token-mean",
        "ppo_mini_batch_size": batch_size,
        "ppo_micro_batch_size_per_gpu": 1,
        "ppo_epochs": 1,
        "use_rollout_log_probs": False,
        "use_kl_loss": False,
        "entropy_coeff": 0.0,
        "ulysses_sequence_parallel_size": 1,
    }


def school_actor_config(batch_size, max_context=40960, max_token_len_per_gpu=None):
    """Use the pinned upstream dynamic-batch/vanilla recipe without replacing loss.

    The upstream actor weights each microbatch token mean by its answer fraction;
    this is not generally the token mean over the entire optimizer minibatch.
    Keep the token budget explicit and identical across the two school controls.
    """
    from omegaconf import OmegaConf

    if type(max_context) is not int or max_context < 1:
        raise ValueError("Positive context budget required")
    budget = max_context if max_token_len_per_gpu is None else max_token_len_per_gpu
    if type(budget) is not int or budget < max_context:
        raise ValueError("Dynamic token budget must cover one complete context")
    values = actor_overrides(batch_size)
    values.update({
        "use_dynamic_bsz": True,
        "ppo_max_token_len_per_gpu": budget,
        "school_actor_recipe": "upstream_dynamic_v1",
        "school_context_length": max_context,
        "use_remove_padding": False,
        "use_fused_kernels": False,
        "use_torch_compile": False,
        "entropy_from_logits_with_chunking": False,
        "entropy_checkpointing": False,
        "grad_clip": 1.0,
        "clip_ratio": 0.2,
        "clip_ratio_low": 0.2,
        "clip_ratio_high": 0.2,
        "clip_ratio_c": 3.0,
    })
    return OmegaConf.from_dotlist([
        f"{key}={str(value).lower() if isinstance(value, bool) else value}"
        for key, value in values.items()
    ])


def validate_actor(config, batch_size, world_size=1):
    if world_size != 1:
        raise ValueError("Student update currently requires one rank")
    if config.get("school_actor_recipe") is not None:
        if config.get("school_actor_recipe") != "upstream_dynamic_v1" or config.get("use_dynamic_bsz") is not True:
            raise ValueError("School actor must use the verified upstream dynamic recipe")
        context = config.get("school_context_length")
        budget = config.get("ppo_max_token_len_per_gpu")
        if type(context) is not int or context < 1 or type(budget) is not int or budget < context:
            raise ValueError("School dynamic token budget cannot cover context")
    for key, expected in actor_overrides(batch_size).items():
        actual = config
        for part in key.split("."):
            actual = actual[part]
        if actual != expected:
            raise ValueError(f"Actor contract mismatch: {key}={actual!r}")


class LocalActorService:
    """Single-process service around the actual upstream DataParallelPPOActor.

    GPU 0 holds the student; a separate GPU can hold the frozen HF teacher.
    This uses real DataProto and actor methods, without Ray/FSDP orchestration.
    """

    def __init__(self, actor, temperature):
        self.actor = actor
        self.temperature = temperature

    def _check(self, data):
        if data.meta_info["temperature"] != self.temperature:
            raise ValueError("Worker/scoring temperature mismatch")

    def revision(self):
        # Process-local freshness guard. A model reload deliberately invalidates
        # receipts; checkpoint provenance remains the driver's responsibility.
        return hashlib.sha256(
            repr(
                [
                    (name, id(p), p._version)
                    for name, p in self.actor.actor_module.named_parameters()
                ]
            ).encode()
        ).hexdigest()

    def compute_log_prob(self, data):
        from verl import DataProto

        self._check(data)
        sampled, _, ids, topk = self.actor.compute_log_prob(
            data, calculate_entropy=False
        )
        return DataProto.from_dict(
            tensors={
                "old_log_probs": sampled,
                "student_top_k_ids": ids,
                "student_top_k_log_probs": topk,
            }
        )

    def compute_distillation_reward(self, data):
        self._check(data)
        return self.actor.compute_distillation_reward(data)

    def update_actor(self, data):
        self._check(data)
        return self.actor.update_policy(data)


@torch.no_grad()
def score_with_actor(
    worker, teacher, actions, *, snapshot_id, eos_ids, pad_id, teacher_temperature=1.0
):
    """Score student with the very same upstream actor used for the update.

    Avoid an HF-vs-actor autocast/rounding mismatch in the frozen Top-16 support.
    Placeholder scores below only construct masks/inputs, never rewards.
    """
    from verl import DataProto

    if not math.isfinite(teacher_temperature) or teacher_temperature <= 0:
        raise ValueError("Invalid teacher temperature")
    placeholders = [
        {
            "binding": _binding(a),
            "ids": torch.arange(16).expand(len(a.target_ids), -1),
            "student": torch.zeros(len(a.target_ids), 16),
            "teacher": torch.zeros(len(a.target_ids), 16),
            "sampled": torch.zeros(len(a.target_ids)),
            "teacher_temperature": teacher_temperature,
        }
        for a in actions
    ]
    tensors, meta = prepare_batch(
        actions, placeholders, snapshot_id=snapshot_id, eos_ids=eos_ids, pad_id=pad_id
    )
    device = next(worker.actor.actor_module.parameters()).device
    batch = DataProto.from_dict(
        tensors={k: v.to(device) for k, v in tensors.items()}, meta_info=meta
    )
    scored = worker.compute_log_prob(batch).batch
    revision = worker.revision()
    results = []
    remote = hasattr(teacher, "score_support")
    if remote and teacher_temperature != 1.:
        raise ValueError("Remote teacher temperature must be 1")
    was_training = None if remote else teacher.training
    if not remote:
        teacher.eval()
    try:
        for i, action in enumerate(actions):
            n = len(action.target_ids)
            ids = scored["student_top_k_ids"][i, :n]
            if remote:
                teacher_values = teacher.score_support(action, ids, 16)['teacher']
            else:
                q = (response_logits(teacher, action.teacher_prompt_ids, action.target_ids).float()
                     / teacher_temperature).log_softmax(-1)
                if q.shape[-1] != text_config(worker.actor.actor_module).vocab_size:
                    raise ValueError('Vocabulary size mismatch')
                teacher_values = q.gather(-1, ids.to(q.device)).cpu()
            results.append(
                {
                    "binding": _binding(action),
                    "ids": ids.cpu(),
                    "actor_revision": revision,
                    "student": scored["student_top_k_log_probs"][i, :n].cpu(),
                    "sampled": scored["old_log_probs"][i, :n].cpu(),
                    "teacher": teacher_values,
                    "teacher_temperature": teacher_temperature,
                }
            )
    finally:
        if not remote:
            teacher.train(was_training)
    return results


def prepare_batch(actions, scores, *, snapshot_id, eos_ids, pad_id):
    """Pack student-only inputs. Teacher tokens/reference never enter actor tensors."""
    if (
        not actions
        or len(actions) != len(scores)
        or len({a.node_id for a in actions}) != len(actions)
    ):
        raise ValueError("Nonempty unique actions with one score each required")
    if type(pad_id) is not int or pad_id < 0:
        raise ValueError("Invalid pad ID")
    for action in actions:
        action.validate(snapshot_id, eos_ids)
    if len({a.sampling_temperature for a in actions}) != 1:
        raise ValueError("Mixed student temperatures")
    if len({a.prompt_binding for a in actions}) != 1:
        raise ValueError("Mixed prompt protocol")
    if len({a.student_tokenizer_hash for a in actions}) != 1:
        raise ValueError("Mixed tokenizer mappings")
    if len({s["teacher_temperature"] for s in scores}) != 1:
        raise ValueError("Mixed teacher temperatures")
    b, p, t = (
        len(actions),
        max(len(a.student_prompt_ids) for a in actions),
        max(len(a.target_ids) for a in actions),
    )
    tensors = {
        "input_ids": torch.full((b, p + t), pad_id, dtype=torch.long),
        "responses": torch.full((b, t), pad_id, dtype=torch.long),
        "attention_mask": torch.zeros((b, p + t), dtype=torch.long),
        "response_mask": torch.zeros((b, t), dtype=torch.long),
        "old_log_probs": torch.zeros((b, t), dtype=scores[0]["sampled"].dtype),
        "student_top_k_ids": torch.zeros((b, t, 16), dtype=torch.long),
        "student_top_k_log_probs": torch.zeros(
            (b, t, 16), dtype=scores[0]["student"].dtype
        ),
        "teacher_on_student_log_probs": torch.zeros(
            (b, t, 16), dtype=scores[0]["teacher"].dtype
        ),
    }
    for i, (a, s) in enumerate(zip(actions, scores, strict=True)):
        n = len(a.target_ids)
        if s["binding"] != _binding(a):
            raise ValueError("Score/context/snapshot binding mismatch")
        for key in ("ids", "student", "teacher"):
            if s[key].shape != (n, 16):
                raise ValueError("Expected T x 16 aligned scores")
        if s["ids"].dtype != torch.long or (s["ids"] < 0).any():
            raise ValueError("Invalid candidate IDs")
        if any(len(set(row.tolist())) != 16 for row in s["ids"]):
            raise ValueError("Duplicate candidate IDs")
        if s["sampled"].shape != (n,):
            raise ValueError("Invalid sampled probability shape")
        for key in ("student", "teacher", "sampled"):
            if not s[key].is_floating_point() or s[key].dtype != scores[0][key].dtype:
                raise ValueError("Mixed or non-floating score dtypes")
            if not torch.isfinite(s[key]).all() or (s[key] > 0).any():
                raise ValueError("Invalid score probabilities")
        tensors["input_ids"][i, p - len(a.student_prompt_ids) : p] = torch.tensor(
            a.student_prompt_ids
        )
        tensors["input_ids"][i, p : p + n] = torch.tensor(a.target_ids)
        tensors["attention_mask"][i, p - len(a.student_prompt_ids) : p + n] = 1
        tensors["responses"][i, :n] = torch.tensor(a.target_ids)
        tensors["response_mask"][i, :n] = 1
        tensors["old_log_probs"][i, :n] = s["sampled"]
        for dest, src in [
            ("student_top_k_ids", "ids"),
            ("student_top_k_log_probs", "student"),
            ("teacher_on_student_log_probs", "teacher"),
        ]:
            tensors[dest][i, :n] = s[src].detach()
    tensors["position_ids"] = (tensors["attention_mask"].cumsum(-1) - 1).clamp_min(0)
    metadata = {
        "temperature": actions[0].sampling_temperature,
        "teacher_temperature": scores[0]["teacher_temperature"],
        "top_k": 16,
        "log_prob_top_k": 16,
        "top_k_strategy": "only_stu",
        "reward_weight_mode": "student_p",
        "micro_batch_size": 1,
        "use_dynamic_bsz": False,
        "global_token_num": tensors["attention_mask"].sum(-1).tolist(),
        "budgetsi": {
            "snapshot_id": snapshot_id,
            "nodes": [a.node_id for a in actions],
            "bindings": [s["binding"] for s in scores],
        },
    }
    return tensors, metadata


def update_actor(
    worker, actions, scores, *, actor_config, snapshot_id, eos_ids, pad_id, world_size=1
):
    """Installed-verl path, through the single-process LocalActorService.

    Ray/FSDP orchestration is not validated by this project bridge. The driver
    owns selection/gate verification; the service checks live score freshness.
    """
    if not isinstance(worker, LocalActorService):
        raise TypeError("Only the verified local actor service is supported")
    validate_actor(actor_config, len(actions), world_size)
    revision = worker.revision()
    if any(s.get("actor_revision") != revision for s in scores):
        raise ValueError("Scores were not computed by the current live actor revision")
    from verl.trainer.ppo.core_algos import compute_token_reward_direct_advantage

    from verl import DataProto

    tensors, metadata = prepare_batch(
        actions, scores, snapshot_id=snapshot_id, eos_ids=eos_ids, pad_id=pad_id
    )
    if isinstance(worker, LocalActorService):
        device = next(worker.actor.actor_module.parameters()).device
        tensors = {k: v.to(device) for k, v in tensors.items()}
    data = DataProto.from_dict(tensors=tensors, meta_info=metadata)
    rewards = worker.compute_distillation_reward(data)
    data = data.union(rewards)
    advantages, returns = compute_token_reward_direct_advantage(
        data.batch["rm_scores"], data.batch["response_mask"]
    )
    data.batch["advantages"] = advantages
    data.batch["returns"] = returns
    return worker.update_actor(data)
