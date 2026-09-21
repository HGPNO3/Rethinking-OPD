"""Bounded two-GPU acceptance: real 4B/14B, actual upstream actor, synthetic prompts.

Not a social-effect experiment or a replacement collector. No external APIs.
Uses local model files only; saves to a new output directory, never old runs.
"""

import argparse
import copy
import hashlib
import importlib.metadata
import json
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
from transformers import AutoModelForCausalLM, AutoTokenizer

from budgetsi.top16 import (
    LocalActorService,
    SelectedAction,
    actor_overrides,
)

from budgetsi.variant_bridge import score_with_actor, update_actor
from budgetsi.variant_spec import VARIANTS, get_variant


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def adapter_state(model):
    return {
        k: v.detach().cpu().clone() for k, v in get_peft_model_state_dict(model).items()
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--student", required=True)
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--variant", choices=tuple(VARIANTS), default="student_top16")
    parser.add_argument("--remote-config")
    parser.add_argument("--approval")
    args = parser.parse_args()
    variant = get_variant(args.variant)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    report = {
        "scope": "synthetic two-action real-model GPU acceptance, not selected social training data",
        "formal_training": False,
        "opd_variant": variant.contract(),
        "ray_fsdp_tested": False,
        "status": "running",
    }
    started = time.monotonic()

    def event(name, **values):
        report.update(values)
        report["stage"] = name
        report["elapsed_seconds"] = time.monotonic() - started
        (out / "report.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(dict(stage=name, **values)), flush=True)

    try:
        if torch.cuda.device_count() != (1 if args.remote_config else 2):
            raise RuntimeError("Unexpected GPU count")
        if args.remote_config:
            from budgetsi.formal_gate import check_launch
            from budgetsi.remote_teacher import RemoteTeacher, contract
            from budgetsi.social_loop import file_hash
            config = json.loads(Path(args.remote_config).read_text())
            gate = check_launch(args.remote_config, args.approval, Path(__file__).resolve().parents[1])
            if config["formal_training"] or config["opd_variant"] != variant.name:
                raise ValueError("Smoke requires the selected variant engineering config")
            if (args.student, args.teacher) != (config["student"], config["teacher"]):
                raise ValueError("Smoke model paths mismatch")
            teacher = RemoteTeacher(config["remote_teacher"]["endpoint"],
                                    contract(config, gate["verified_bindings"]["git_commit"]))
            for path, expected in config["files_sha256"].items():
                if Path(path).parent == Path(config["teacher"]):
                    assert teacher.info["assets"][path] == expected
                else:
                    assert file_hash(path) == expected
            report["gate"] = gate
        torch.cuda.set_device(0)
        torch.manual_seed(37)
        torch.distributed.init_process_group(
            "nccl",
            init_method=f"file://{out.resolve()}/dist_init",
            rank=0,
            world_size=1,
        )
        from budgetsi.model_runtime import upstream_actor_class
        DataParallelPPOActor = upstream_actor_class()

        st = AutoTokenizer.from_pretrained(args.student, local_files_only=True)
        tt = AutoTokenizer.from_pretrained(args.teacher, local_files_only=True)
        sf, tf = digest(st.get_vocab()), digest(tt.get_vocab())
        if sf != tf:
            raise RuntimeError("Tokenizer vocab/token-ID mapping differs")
        event(
            "tokenizers_checked",
            tokenizer_fingerprint=sf,
            devices=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            model_paths={"student": args.student, "teacher": args.teacher},
            runtime_versions={
                name: importlib.metadata.version(name)
                for name in (
                    "torch",
                    "transformers",
                    "peft",
                    "ray",
                    "tensordict",
                    "omegaconf",
                )
            },
        )
        from budgetsi.model_runtime import load_model, lora_targets, text_config, eos_contract
        student = load_model(args.student, {'': 0})
        targets = lora_targets(student)
        student = get_peft_model(
            student,
            LoraConfig(
                r=32,
                lora_alpha=64,
                lora_dropout=0.0,
                target_modules=targets,
                task_type="CAUSAL_LM",
            ),
        )
        student.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        student.config.use_cache = False
        text_config(student).use_cache = False
        event(
            "student_loaded",
            trainable_parameters=sum(
                p.numel() for p in student.parameters() if p.requires_grad
            ),
        )
        if not args.remote_config:
            teacher = AutoModelForCausalLM.from_pretrained(
                args.teacher,
                local_files_only=True,
                torch_dtype=torch.bfloat16,
                attn_implementation="sdpa",
                device_map={"": 1},
            )
            teacher.requires_grad_(False)
            teacher.eval()
        event("teacher_loaded")

        # Deliberately short examples for correctness, not memory/throughput sizing.
        examples = [
            (
                "A colleague cannot meet today. Offer a meeting tomorrow in one short sentence.",
                '{"action_type":"speak","argument":"Would tomorrow morning work for you?"}',
            ),
            (
                "A colleague asks you to share a private password. Decline politely in one short sentence.",
                '{"action_type":"speak","argument":"I cannot share passwords, but I can help request your own access."}',
            ),
        ]
        actions = []
        student.eval()
        eos_ids = eos_contract(st, args.student)
        for i, (question, reference) in enumerate(examples):
            messages = [
                {
                    "role": "system",
                    "content": 'Return one short action JSON with action_type="speak" and argument. No explanations.',
                },
                {"role": "user", "content": question},
            ]
            prompt = tuple(
                st.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    enable_thinking=False,
                    return_dict=False,
                )
            )
            ids = torch.tensor([prompt], device="cuda:0")
            with torch.no_grad():
                generated = student.generate(
                    input_ids=ids,
                    attention_mask=torch.ones_like(ids),
                    max_new_tokens=128,
                    do_sample=True,
                    temperature=0.7,
                    top_p=1.0,
                    top_k=0,
                    pad_token_id=st.pad_token_id,
                    eos_token_id=list(eos_ids),
                    use_cache=True,
                )[0, len(prompt) :].tolist()
            if not generated or generated[-1] not in eos_ids:
                raise RuntimeError(
                    "Synthetic rollout did not finish; no forced EOS or fabricated target"
                )
            teacher_messages = [
                {
                    "role": "system",
                    "content": "Guide the role toward the reference action using only the visible situation. The reference is an unexecuted suggestion, not an event. Continue the original student prefix.",
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {"visible_messages": messages, "reference_action": reference}
                    ),
                },
            ]
            tp = tuple(
                tt.apply_chat_template(
                    teacher_messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    enable_thinking=False,
                    return_dict=False,
                )
            )
            actions.append(
                SelectedAction(
                    f"synthetic-{i}",
                    "gpu-smoke-initial",
                    prompt,
                    tp,
                    tuple(generated),
                    sf,
                    tf,
                    "synthetic-reference-v1",
                    digest(reference),
                    0.7,
                )
            )
        event(
            "rollouts_generated",
            target_lengths=[len(a.target_ids) for a in actions],
            student_prompt_lengths=[len(a.student_prompt_ids) for a in actions],
            teacher_prompt_lengths=[len(a.teacher_prompt_ids) for a in actions],
        )
        # Store IDs for reproducibility; no real user/private conversation is used.
        (out / "synthetic_actions.json").write_text(
            json.dumps([a.__dict__ for a in actions])
        )
        cfg = OmegaConf.from_dotlist(
            [
                f"{k}={str(v).lower() if isinstance(v, bool) else v}"
                for k, v in actor_overrides(len(actions)).items()
            ]
        )
        cfg.update(
            {
                "use_remove_padding": False,
                "use_fused_kernels": False,
                "use_torch_compile": False,
                "entropy_from_logits_with_chunking": False,
                "entropy_checkpointing": False,
                "use_dynamic_bsz": False,
                "grad_clip": 1.0,
                "clip_ratio": 0.2,
                "clip_ratio_low": 0.2,
                "clip_ratio_high": 0.2,
                "clip_ratio_c": 3.0,
            }
        )
        # Engineering smoke LR only; not a new formal training recipe.
        optimizer = torch.optim.AdamW(
            [p for p in student.parameters() if p.requires_grad],
            lr=1e-6,
            weight_decay=0.01,
        )
        actor = DataParallelPPOActor(cfg, student, optimizer)
        service = LocalActorService(actor, 0.7)
        scores = score_with_actor(
            service,
            teacher,
            actions,
            variant=variant,
            snapshot_id="gpu-smoke-initial",
            eos_ids=eos_ids,
            pad_id=st.pad_token_id,
        )
        event("opd_scored", score_shapes=[list(s["student" if variant.k else "sampled"].shape) for s in scores])
        before = adapter_state(student)
        frozen_versions = {
            k: p._version for k, p in student.named_parameters() if not p.requires_grad
        }
        teacher_versions = {} if args.remote_config else {k: p._version for k, p in teacher.named_parameters()}
        metrics = update_actor(
            service,
            actions,
            scores,
            actor_config=cfg,
            variant=variant,
            snapshot_id="gpu-smoke-initial",
            eos_ids=eos_ids,
            pad_id=st.pad_token_id,
        )
        after = adapter_state(student)
        changed = [k for k in before if not torch.equal(before[k], after[k])]
        if not changed:
            raise RuntimeError("No LoRA parameter changed")
        if any(
            p._version != frozen_versions[k] or p.grad is not None
            for k, p in student.named_parameters()
            if not p.requires_grad
        ):
            raise RuntimeError("Frozen student parameter modified")
        if args.remote_config:
            report["teacher_status"] = teacher.assert_frozen()
        else:
            if any(
                p._version != teacher_versions[k] or p.grad is not None
                for k, p in teacher.named_parameters()
            ):
                raise RuntimeError("Frozen teacher modified")
        if not all(torch.isfinite(v).all() for v in after.values()):
            raise RuntimeError("Nonfinite updated adapter")
        event(
            "upstream_update_passed",
            changed_adapter_tensors=len(changed),
            metrics=metrics,
            max_adapter_delta=max(
                (after[k] - before[k]).abs().max().item() for k in before
            ),
        )
        student.save_pretrained(out / "adapter", safe_serialization=True)
        torch.save(optimizer.state_dict(), out / "optimizer.pt")
        # Perturb and restore to prove actual restoration, not simply saving files.
        expected_optimizer = copy.deepcopy(optimizer.state_dict())
        with torch.no_grad():
            next(p for p in student.parameters() if p.requires_grad).add_(1.0)
        optimizer.state.clear()
        from safetensors.torch import load_file

        restored = load_file(str(out / "adapter/adapter_model.safetensors"))
        set_peft_model_state_dict(student, restored)
        optimizer.load_state_dict(
            torch.load(out / "optimizer.pt", map_location="cpu", weights_only=True)
        )
        for k, v in adapter_state(student).items():
            torch.testing.assert_close(v, after[k], rtol=0, atol=0)
        for key, state in expected_optimizer["state"].items():
            for name, value in state.items():
                actual = optimizer.state_dict()["state"][key][name]
                if isinstance(value, torch.Tensor):
                    torch.testing.assert_close(
                        value.cpu(), actual.cpu(), rtol=0, atol=0
                    )
                elif actual != value:
                    raise RuntimeError("Optimizer restore mismatch")
        event(
            "completed",
            status="passed",
            adapter_restore_exact=True,
            optimizer_restore_exact=True,
            peak_allocated_gib=[
                torch.cuda.max_memory_allocated(i) / 2**30 for i in range(torch.cuda.device_count())
            ],
            peak_reserved_gib=[
                torch.cuda.max_memory_reserved(i) / 2**30 for i in range(torch.cuda.device_count())
            ],
        )
    except Exception as error:
        event("failed", status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
