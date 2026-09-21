"""Bounded real-social acceptance: collect, update, restore, collect, update.

Reuses the pinned collector protocol and the actual upstream OPD actor. No
external APIs, no effect evaluation, and no automatic formal-training phase.
"""

import argparse
import copy
import hashlib
import json
import os
import random
import subprocess
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer, ThreadingHTTPServer
from pathlib import Path

import torch
from omegaconf import OmegaConf
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from budgetsi.gpu_smoke import adapter_state
from budgetsi.top16 import (
    LocalActorService,
    actor_overrides,
    from_collector_record,
    response_logits,
    score_with_actor,
    update_actor,
)

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "social_protocol"))
from runner import (
    prompt_binding,
    seed_for,
    select,
    validate_reference_record,
)


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024**2), b""):
            result.update(chunk)
    return result.hexdigest()


def state_hash(state):
    result = hashlib.sha256()
    for name, value in sorted(state.items()):
        result.update(name.encode())
        result.update(value.contiguous().view(torch.uint8).numpy().tobytes())
    return result.hexdigest()


def atomic(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    tmp.replace(path)


def same_state(expected, actual):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(expected.cpu(), actual.cpu(), rtol=0, atol=0)
    elif isinstance(expected, dict):
        assert expected.keys() == actual.keys()
        for key in expected:
            same_state(expected[key], actual[key])
    elif isinstance(expected, (list, tuple)):
        assert len(expected) == len(actual)
        for a, b in zip(expected, actual):
            same_state(a, b)
    else:
        assert expected == actual


class Engine:
    def __init__(self, student, teacher, tokenizers, max_context, out, *, temperature=0.7):
        self.models = {"student": student, "teacher": teacher}
        self.tokenizers = tokenizers
        self.max_context = max_context
        self.temperature = temperature
        self.out = out
        self.snapshot = None
        self.accepting = False
        self.requests = []
        self.lock = threading.Lock()

    @torch.no_grad()
    def call(self, data):
        if not self.accepting:
            raise RuntimeError("Collection is closed during policy update")
        if data["model"] == "teacher" and hasattr(self.models["teacher"], "score_support"):
            started = time.monotonic()
            result = self.models["teacher"].call("collect", data)
            receipt = dict(model="teacher", operation=data.get("operation", "generate"),
                           snapshot=self.snapshot, seconds=time.monotonic() - started, **result["usage"])
            self.requests.append(receipt)
            with (self.out / "engine_requests.jsonl").open("a") as handle:
                handle.write(json.dumps(receipt) + "\n")
            return result
        model = self.models[data["model"]]
        tok = self.tokenizers[data["model"]]
        model.eval()
        started = time.monotonic()
        if data.get("operation") == "score":
            prompt, target = data["prompt_ids"], data["target_ids"]
            if len(prompt) + len(target) + 1 > self.max_context:
                raise ValueError("Context overflow")
            logits = response_logits(model, prompt, target).float()
            ids = torch.tensor(target, device=logits.device)
            values = logits.log_softmax(-1).gather(-1, ids[:, None]).squeeze(-1)
            result = {"raw_logprobs": values.cpu().tolist()}
            operation = "score"
        else:
            prompt = data["prompt"]
            remaining = self.max_context - len(prompt)
            if remaining < 1 or data["max_tokens"] != remaining:
                raise ValueError("Unexpected generation context limit")
            if data["temperature"] != self.temperature or data["top_p"] != 1 or data["top_k"] != -1:
                raise ValueError("Unexpected decoding protocol")
            inputs = torch.tensor([prompt], device=model.device)
            # Explicit neutral penalties; do not inherit model sampling defaults.
            generation = GenerationConfig(
                do_sample=True,
                temperature=self.temperature,
                top_p=1.0,
                top_k=0,
                repetition_penalty=1.0,
                max_new_tokens=remaining,
                eos_token_id=data["stop_token_ids"],
                pad_token_id=tok.pad_token_id,
                use_cache=True,
                return_dict_in_generate=True,
                output_scores=True,
            )
            with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
                torch.manual_seed(data["seed"])
                output = model.generate(
                    input_ids=inputs,
                    attention_mask=torch.ones_like(inputs),
                    generation_config=generation,
                    # Transformers fills neutral GenerationConfig values from the
                    # model's defaults. Explicit kwargs prevent top_p=0.95 fallback.
                    top_p=1.0,
                    top_k=0,
                    temperature=self.temperature,
                    repetition_penalty=1.0,
                )
            target = output.sequences[0, len(prompt) :].tolist()
            values = [
                float(score[0].float().log_softmax(-1)[tid].item())
                for score, tid in zip(output.scores, target)
            ]
            finished = bool(target and target[-1] in data["stop_token_ids"])
            result = {
                "choices": [
                    {
                        "token_ids": target,
                        "finish_reason": "stop" if finished else "length",
                        "logprobs": {"token_logprobs": values},
                    }
                ]
            }
            operation = "generate"
        result["usage"] = {
            "prompt_tokens": len(prompt),
            "completion_tokens": len(target),
        }
        receipt = {
            "model": data["model"],
            "operation": operation,
            "snapshot": self.snapshot,
            "seconds": time.monotonic() - started,
            **result["usage"],
        }
        self.requests.append(receipt)
        with (self.out / "engine_requests.jsonl").open("a") as handle:
            handle.write(json.dumps(receipt) + "\n")
        return result


def start_server(engine):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):
            try:
                if self.path != "/v1/completions":
                    raise ValueError("Unknown local endpoint")
                data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if getattr(engine, "parallel", False):
                    result = engine.call(data)
                else:
                    with engine.lock:
                        result = engine.call(data)
                status = 200
            except Exception as error:  # noqa: BLE001 - HTTP boundary records and returns every failure
                traceback.print_exc()
                status, result = 500, {"error": repr(error)}
            body = json.dumps(result).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    from budgetsi.http_runtime import BurstHTTPServer

    server_class = BurstHTTPServer if getattr(engine, "parallel", False) else HTTPServer
    server = server_class(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def actor_config(count):
    cfg = OmegaConf.from_dotlist(
        [
            f"{k}={str(v).lower() if isinstance(v, bool) else v}"
            for k, v in actor_overrides(count).items()
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
    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "social_acceptance.json"))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = {
        "status": "running",
        "scope": config["scope"],
        "formal_training": False,
        "updates": [],
        "batches": [],
        "prompt_binding": prompt_binding(),
    }
    server = child = None

    def event(stage, **values):
        report.update(values, stage=stage, elapsed_seconds=time.monotonic() - started)
        atomic(out / "report.json", report)
        print(json.dumps({"stage": stage, **values}), flush=True)

    try:
        assert config["max_updates"] == 2 and not config["external_api"]
        assert torch.cuda.device_count() == 2
        # Fixed approved local assets. Fail before loading if any changed.
        for name, expected in config["source_provenance"].items():
            assert file_hash(ROOT / "social_protocol" / name) == expected, name
        event("verifying_assets")
        verified = {}
        for name, expected in config["files_sha256"].items():
            actual = file_hash(name)
            assert actual == expected, name
            verified[name] = actual
        atomic(out / "verified_assets.json", verified)
        atomic(out / "config.json", config)
        torch.cuda.set_device(0)
        torch.manual_seed(config["seed"])
        torch.distributed.init_process_group(
            "nccl", init_method=f"file://{out}/dist_init", rank=0, world_size=1
        )
        from verl.workers.actor.dp_actor import DataParallelPPOActor

        st = AutoTokenizer.from_pretrained(config["student"], local_files_only=True)
        tt = AutoTokenizer.from_pretrained(config["teacher"], local_files_only=True)
        assert st.get_vocab() == tt.get_vocab()
        student = AutoModelForCausalLM.from_pretrained(
            config["student"],
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            device_map={"": 0},
        )
        student = get_peft_model(
            student,
            LoraConfig(
                r=config["optimizer"]["r"],
                lora_alpha=config["optimizer"]["alpha"],
                lora_dropout=0.0,
                target_modules="all-linear",
                task_type="CAUSAL_LM",
            ),
        )
        student.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        student.config.use_cache = False
        event("student_loaded")
        teacher = AutoModelForCausalLM.from_pretrained(
            config["teacher"],
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            device_map={"": 1},
        )
        teacher.requires_grad_(False)
        teacher.eval()
        event("teacher_loaded")
        eos = student.generation_config.eos_token_id
        eos = set(eos if isinstance(eos, list) else [eos])
        eos.add(st.convert_tokens_to_ids("<|im_end|>"))
        optimizer = torch.optim.AdamW(
            [p for p in student.parameters() if p.requires_grad],
            lr=config["optimizer"]["lr"],
            weight_decay=config["optimizer"]["weight_decay"],
        )
        frozen = {
            k: p._version for k, p in student.named_parameters() if not p.requires_grad
        }
        teacher_versions = {k: p._version for k, p in teacher.named_parameters()}
        engine = Engine(
            student, teacher, {"student": st, "teacher": tt}, config["context"], out
        )
        server = start_server(engine)
        pool = json.loads((ROOT / "social_protocol/inputs.json").read_text())["scenes"]
        pool = sorted(
            pool,
            key=lambda s: hashlib.sha256(
                ("split20260915" + s["id"]).encode()
            ).hexdigest(),
        )[:80]
        random.Random(config["seed"]).shuffle(pool)
        used = set()
        previous_snapshot = None
        for batch_index in range(config["max_scene_batches"]):
            if len(report["updates"]) == 2:
                break
            batch = out / f"batch_{batch_index:02d}"
            batch.mkdir()
            current = state_hash(adapter_state(student))
            if previous_snapshot is not None:
                assert current == previous_snapshot
            engine.snapshot = current
            scene = copy.deepcopy(pool[batch_index])
            scene["id"] = f"top16-acceptance:b{batch_index}:" + scene["id"]
            scene["seed"] = config["seed"] + batch_index
            atomic(batch / "inputs.json", {"scenes": [scene]})
            event("collecting", batch=batch_index, snapshot=current)
            engine.accepting = True
            request_start = len(engine.requests)
            collection_start = time.monotonic()
            command = [
                config["collector_python"],
                str(ROOT / "social_collect.py"),
                "--inputs",
                str(batch / "inputs.json"),
                "--output",
                str(batch / "rollout"),
                "--student-tokenizer",
                config["student"],
                "--teacher-tokenizer",
                config["teacher"],
                "--endpoint",
                f"http://127.0.0.1:{server.server_port}",
                "--max-context",
                str(config["context"]),
                "--seed",
                str(config["seed"]),
                "--goal-instruction",
                config["goal_instruction"],
                "--snapshot-id",
                current,
            ]
            env = os.environ.copy()
            # Same local-storage setting as the original collector launcher.
            env["SOTOPIA_STORAGE_BACKEND"] = "local"
            env.pop(
                "PYTHONPATH", None
            )  # Collector retains its own compatible Sotopia dependencies.
            with (batch / "collector.log").open("w") as log:
                child = subprocess.Popen(
                    command, stdout=log, stderr=subprocess.STDOUT, env=env
                )
                child.wait(timeout=3600)
            if child.returncode:
                raise RuntimeError(f"Collector failed; see {batch}/collector.log")
            child = None
            with engine.lock:
                engine.accepting = False
            assert state_hash(adapter_state(student)) == current
            summary = json.loads((batch / "rollout/summary.json").read_text())
            if summary["status"] != "completed":
                raise RuntimeError(
                    "Incomplete real dialogue; preserve failure, do not silently train"
                )
            records = json.loads((batch / "rollout/selected_records.json").read_text())
            nodes = {
                r["id"]: r
                for r in json.loads((batch / "rollout/nodes.json").read_text())
            }
            for record in records:
                node = nodes[record["id"]]
                selected, _ = select(
                    node["candidates"], seed_for(config["seed"], record["id"], "select")
                )
                assert selected == record["specialty"] == node["selected"]
                reference = next(
                    c["generation"]["action"]
                    for c in node["candidates"]
                    if c["specialty"] == selected
                )
                assert reference == record["reference_action"]
                assert node["original"] == record["original"]
            report["batches"].append(
                {
                    "batch": batch_index,
                    "selected_nodes": len(records),
                    "A_actions": summary["A_actions"],
                    "snapshot": current,
                    "collection_seconds": time.monotonic() - collection_start,
                    "requests": len(engine.requests) - request_start,
                }
            )
            atomic(out / "report.json", report)
            if not records:
                continue
            actions = [
                from_collector_record(
                    r,
                    student_tokenizer=st,
                    teacher_tokenizer=tt,
                    validate_record=validate_reference_record,
                    snapshot_id=current,
                    eos_ids=eos,
                )
                for r in records
            ]
            assert not used.intersection(a.node_id for a in actions)
            cfg = actor_config(len(actions))
            service = LocalActorService(
                DataParallelPPOActor(cfg, student, optimizer), 0.7
            )
            event("scoring_top16", batch=batch_index, selected_nodes=len(actions))
            update_started = time.monotonic()
            scores = score_with_actor(
                service,
                teacher,
                actions,
                snapshot_id=current,
                eos_ids=eos,
                pad_id=st.pad_token_id,
            )
            event("updating", batch=batch_index)
            before = adapter_state(student)
            metrics = update_actor(
                service,
                actions,
                scores,
                actor_config=cfg,
                snapshot_id=current,
                eos_ids=eos,
                pad_id=st.pad_token_id,
            )
            after = adapter_state(student)
            changed = sum(not torch.equal(before[k], after[k]) for k in before)
            assert changed and all(torch.isfinite(v).all() for v in after.values())
            assert all(
                p._version == frozen[k] and p.grad is None
                for k, p in student.named_parameters()
                if not p.requires_grad
            )
            assert all(
                p._version == teacher_versions[k] and p.grad is None
                for k, p in teacher.named_parameters()
            )
            update_dir = batch / "update"
            update_dir.mkdir()
            student.save_pretrained(update_dir / "adapter", safe_serialization=True)
            torch.save(optimizer.state_dict(), update_dir / "optimizer.pt")
            expected_optimizer = copy.deepcopy(optimizer.state_dict())
            # Exercise loading, then actually collect and update from restored state.
            with torch.no_grad():
                next(p for p in student.parameters() if p.requires_grad).add_(1.0)
            optimizer.state.clear()
            set_peft_model_state_dict(
                student,
                load_file(str(update_dir / "adapter/adapter_model.safetensors")),
            )
            optimizer.load_state_dict(
                torch.load(
                    update_dir / "optimizer.pt", map_location="cpu", weights_only=True
                )
            )
            same_state(after, adapter_state(student))
            same_state(expected_optimizer, optimizer.state_dict())
            steps = {
                int(s["step"].item()) for s in optimizer.state.values() if "step" in s
            }
            assert steps == {len(report["updates"]) + 1}
            previous_snapshot = state_hash(after)
            assert previous_snapshot != current
            receipt = {
                "status": "passed",
                "batch": batch_index,
                "snapshot_before": current,
                "snapshot_after": previous_snapshot,
                "nodes": [a.node_id for a in actions],
                "changed_adapter_tensors": changed,
                "optimizer_step": next(iter(steps)),
                "adapter_restore_exact": True,
                "optimizer_restore_exact": True,
                "update_and_restore_seconds": time.monotonic() - update_started,
                "student_prompt_lengths": [len(a.student_prompt_ids) for a in actions],
                "teacher_prompt_lengths": [len(a.teacher_prompt_ids) for a in actions],
                "target_lengths": [len(a.target_ids) for a in actions],
                "metrics": metrics,
                "adapter_file_sha256": file_hash(
                    update_dir / "adapter/adapter_model.safetensors"
                ),
                "optimizer_file_sha256": file_hash(update_dir / "optimizer.pt"),
                "peak_allocated_gib": [
                    torch.cuda.max_memory_allocated(i) / 2**30 for i in range(2)
                ],
                "peak_reserved_gib": [
                    torch.cuda.max_memory_reserved(i) / 2**30 for i in range(2)
                ],
            }
            atomic(update_dir / "result.json", receipt)
            report["updates"].append(receipt)
            used.update(receipt["nodes"])
            del scores, before, after, expected_optimizer
            torch.cuda.empty_cache()
            event("batch_passed", batch=batch_index)
        assert len(report["updates"]) == 2, (
            "Insufficient selected batches within predeclared bound"
        )
        assert (
            report["updates"][1]["snapshot_before"]
            == report["updates"][0]["snapshot_after"]
        )
        event(
            "completed",
            status="passed",
            used_nodes=len(used),
            restored_checkpoint_used_for_next_collection=True,
            optimizer_continued_to_step_2=True,
            peak_allocated_gib=[
                torch.cuda.max_memory_allocated(i) / 2**30 for i in range(2)
            ],
            peak_reserved_gib=[
                torch.cuda.max_memory_reserved(i) / 2**30 for i in range(2)
            ],
        )
    except BaseException as error:
        event("failed", status="failed", error=repr(error))
        raise
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        if server:
            server.shutdown()
            server.server_close()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
