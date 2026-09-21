"""Gated online OPD training: exact used-node quota and transactional resume."""

import argparse
from contextlib import nullcontext
import fcntl
import signal
from budgetsi.run_state import Ledger, choose_quota, batch_scenes
from budgetsi.formal_gate import check_launch
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
from http.server import BaseHTTPRequestHandler, HTTPServer
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
)

from budgetsi.variant_bridge import score_with_actor, update_actor
from budgetsi.variant_spec import from_config

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "social_protocol"))
from runner import (
    PROMPT_VERSION,
    prompt_binding,
    teacher_context_binding,
    seed_for,
    select,
    validate_reference_record,
)


from budgetsi.social_loop import (file_hash, state_hash, atomic, same_state, Engine,
                                 start_server, actor_config)


def profile_mark(out, stage, batch):
    """Synchronized stage boundary; numerical inputs and outputs are untouched."""
    if torch.cuda.is_initialized():
        torch.cuda.synchronize()
    with (out / 'phase_profile.jsonl').open('a') as handle:
        handle.write(json.dumps(dict(stage=stage, batch=batch, monotonic=time.monotonic(), epoch=time.time()))+'\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "formal_config.json"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--approval", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    variant = from_config(config)
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=args.resume)
    lock = (out / "run.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    gate = check_launch(Path(args.config), Path(args.approval), ROOT.parent)
    ledger = Ledger(out, config, gate["verified_bindings"]["git_commit"], args.resume)
    atomic(out / "gate.json", gate)
    atomic(out / "approval.json", json.loads(Path(args.approval).read_text()))
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Signal {signum}; resume from last committed batch")
    signal.signal(signal.SIGTERM, interrupted)
    started = time.monotonic()
    report = {
        "status": "running",
        "scope": config["scope"],
        "formal_training": config["formal_training"],
        "target_nodes": config["target_nodes"],
        "opd_variant": variant.contract(),
        "used_nodes": len(ledger.used),
        "updates": list(ledger.state["updates"]),
        "batches": list(ledger.state["batches"]),
        "prompt_binding": prompt_binding(),
    }
    server = child = telemetry = None
    student_replica = None
    student_score_replica = None
    teacher = None
    phased = bool(config.get("teacher_inference"))

    def event(stage, **values):
        report.update(values, stage=stage, elapsed_seconds=time.monotonic() - started)
        atomic(out / "report.json", report)
        print(json.dumps({"stage": stage, **values}), flush=True)

    try:
        if config.get('logging'):
            from budgetsi.telemetry import Telemetry
            from budgetsi.diagnostic_run import publish
            telemetry = Telemetry(out/'telemetry', **config['logging'], config=dict(
                variant=variant.contract(),model_pair=[Path(config['student']).name,Path(config['teacher']).name],
                optimizer=config['optimizer'], diagnostics=config['diagnostics'], git_commit=gate['verified_bindings']['git_commit'],
                teacher_context_mode=config.get("teacher_context_mode", "reference_context"),
                school_p0=config.get("school_p0"), target_nodes=config["target_nodes"],
                temperature=config["temperature"], seed=config["seed"]))
            # Replay committed updates after a crash between checkpoint commit and logging.
            for receipt in ledger.state['updates']:
                publish(telemetry,receipt,(out/receipt['checkpoint']).parent,config)
        assert not config["external_api"]
        remote_cfg = config.get("remote_teacher")
        assert torch.cuda.device_count() == (1 if remote_cfg else 2)
        if remote_cfg:
            from budgetsi.remote_teacher import RemoteTeacher, contract
            teacher = RemoteTeacher(remote_cfg["endpoint"], contract(config, gate["verified_bindings"]["git_commit"]))
        # Fixed approved local assets. Fail before loading if any changed.
        for name, expected in config["source_provenance"].items():
            assert file_hash(ROOT / "social_protocol" / name) == expected, name
        event("verifying_assets")
        verified = {}
        for name, expected in config["files_sha256"].items():
            remote_asset = remote_cfg and Path(name).parent == Path(config["teacher"])
            actual = teacher.info["assets"][name] if remote_asset else file_hash(name)
            assert actual == expected, name
            if remote_asset and not name.endswith(".safetensors"):
                assert file_hash(name) == expected, "Local teacher tokenizer/config changed: " + name
            verified[name] = actual
        atomic(out / "verified_assets.json", verified)
        atomic(out / "config.json", config)
        torch.cuda.set_device(0)
        torch.manual_seed(config["seed"])
        torch.distributed.init_process_group(
            "nccl", init_method=f"file://{out}/dist_init_{os.getpid()}", rank=0, world_size=1
        )
        from budgetsi.model_runtime import upstream_actor_class
        DataParallelPPOActor = upstream_actor_class()

        st = AutoTokenizer.from_pretrained(config["student"], local_files_only=True)
        tt = AutoTokenizer.from_pretrained(config["teacher"], local_files_only=True)
        # Load the shipped backend verbatim: Transformers 5 reconstructs a
        # different Qwen2 Unicode pre-tokenizer than the rollout environment.
        from tokenizers import Tokenizer
        st._tokenizer = Tokenizer.from_file(str(Path(config["student"]) / "tokenizer.json"))
        tt._tokenizer = Tokenizer.from_file(str(Path(config["teacher"]) / "tokenizer.json"))
        assert st.get_vocab() == tt.get_vocab()
        from budgetsi.model_runtime import load_model, lora_targets, text_config, eos_contract
        student = load_model(config['student'], {'': 0})
        targets = lora_targets(student)
        student = get_peft_model(
            student,
            LoraConfig(
                r=config["optimizer"]["r"],
                lora_alpha=config["optimizer"]["alpha"],
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
        event("student_loaded")
        if not remote_cfg:
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
        eos = eos_contract(st, config['student'])
        if eos != eos_contract(tt, config['teacher']):
            raise ValueError('Teacher/student EOS contract mismatch')
        optimizer = torch.optim.AdamW(
            [p for p in student.parameters() if p.requires_grad],
            lr=config["optimizer"]["lr"],
            weight_decay=config["optimizer"]["weight_decay"],
        )
        frozen = {
            k: p._version for k, p in student.named_parameters() if not p.requires_grad
        }
        teacher_versions = {} if remote_cfg else {k: p._version for k, p in teacher.named_parameters()}
        parallel = config.get("collection_parallel")
        engine_factory = Engine
        engine_options = {"temperature": config["temperature"]}
        if parallel:
            from budgetsi.parallel_collect import make_parallel_engine
            engine_factory = make_parallel_engine
            engine_options.update(batch_size=parallel["batch_size"], wait_ms=parallel["wait_ms"])
        if config.get('student_replica'):
            from budgetsi.student_replica import StudentReplica
            student_replica = StudentReplica(config, config['student_replica']['cuda_visible_devices'])
            engine_options['student_replica'] = student_replica
        if config.get('student_score_replica'):
            from budgetsi.score_replica import StudentScoreReplica
            student_score_replica = StudentScoreReplica(config)
            engine_options['student_score_replica'] = student_score_replica
        engine = engine_factory(
            student, teacher, {"student": st, "teacher": tt}, config["context"], out, **engine_options
        )
        server = start_server(engine)
        if config.get("school_p0"):
            from budgetsi.school_data import load_initializations, schedule_batch
            pool = load_initializations(ROOT.parent / config["training_data"]["path"])
        else:
            pool = json.loads((ROOT / "social_protocol/inputs.json").read_text())["scenes"]
        used = ledger.used
        previous_snapshot = None
        if ledger.state["updates"]:
            receipt = ledger.state["updates"][-1]
            update_dir = out / receipt["checkpoint"]
            set_peft_model_state_dict(student, load_file(str(update_dir / "adapter/adapter_model.safetensors")))
            optimizer.load_state_dict(torch.load(update_dir / "optimizer.pt", map_location="cpu", weights_only=True))
            assert state_hash(adapter_state(student)) == receipt["snapshot_after"]
            assert {int(s["step"].item()) for s in optimizer.state.values()} == {len(ledger.state["updates"])}
            previous_snapshot = receipt["snapshot_after"]
            event("resumed", used_nodes=len(used), restored_optimizer_step=len(ledger.state["updates"]))
        if phased and len(used) == config['target_nodes']:
            teacher.call("phase", {"action": "collected"})
            teacher.call("phase", {"action": "updated", "done": True})
        for batch_index in range(ledger.state["next_batch"], config["max_scene_batches"]):
            if len(used) == config["target_nodes"]:
                break
            if __import__("shutil").disk_usage(out).free < 3 * 2**30:
                raise RuntimeError("Less than 3 GiB available; stop before collection")
            # Every retry has its own evidence directory. Only state.json commits progress.
            batch = out / f"batch_{batch_index:04d}_attempt_{time.time_ns()}"
            batch.mkdir()
            current = state_hash(adapter_state(student))
            if previous_snapshot is not None:
                assert current == previous_snapshot
            if student_replica is not None:
                student_replica.sync_adapter(adapter_state(student), current)
            if student_score_replica is not None:
                student_score_replica.sync_adapter(adapter_state(student), current)
            engine.snapshot = current
            if config.get("school_p0"):
                inputs = {"schema_version": "school_online_batch_v1", "batch_index": batch_index,
                          "training_data": config["training_data"],
                          "scenes": schedule_batch(pool, batch_index, config["dialogues_per_batch"], config["seed"])}
            else:
                inputs = batch_scenes(pool, batch_index, config["dialogues_per_batch"], config["seed"])
            atomic(batch / "inputs.json", inputs)
            event("collecting", batch=batch_index, snapshot=current)
            engine.accepting = True
            request_start = len(engine.requests)
            teacher_timing_start = len(teacher.timings) if remote_cfg else 0
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
            command.extend(["--teacher-context-mode", config.get("teacher_context_mode", "reference_context"),
                            "--temperature", str(config["temperature"])])
            if parallel:
                command.extend(["--concurrency", str(parallel["dialogues"])])
            if config["formal_training"]:
                command.append("--formal-training")
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
                child.wait(timeout=config["collection_timeout_seconds"])
            if child.returncode:
                raise RuntimeError(f"Collector failed; see {batch}/collector.log")
            child = None
            if parallel:
                engine.close_collection()
            else:
                with engine.lock:
                    engine.accepting = False
            assert state_hash(adapter_state(student)) == current
            summary = json.loads((batch / "rollout/summary.json").read_text())
            if summary.get("teacher_context_mode") != config.get("teacher_context_mode", "reference_context"):
                raise ValueError("Collector teacher context differs from configuration")
            if (summary.get("prompt_version") != PROMPT_VERSION
                    or summary.get("prompt_binding") != prompt_binding()
                    or summary.get("teacher_context_binding") != teacher_context_binding(config.get("teacher_context_mode", "reference_context"))
                    or summary.get("settings", {}).get("temperature") != config["temperature"]):
                raise ValueError("Collector protocol/temperature binding differs from configuration")
            records = json.loads((batch / "rollout/selected_records.json").read_text())
            from budgetsi.batch_validation import validate_batch
            validation = validate_batch(summary,
                json.loads((batch / "rollout/dialogues.json").read_text()),
                [scene["id"] for scene in inputs["scenes"]], records)
            atomic(batch / "batch_validation.json", validation)
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
            if phased:
                event("waiting_for_shared_teacher_scoring", batch=batch_index)
                barrier_start = time.monotonic()
                teacher.call("phase", {"action": "collected"})
                report["batches"][-1]["teacher_scoring_barrier_seconds"] = time.monotonic()-barrier_start
            records, excluded = choose_quota(records, used, config["target_nodes"])
            atomic(batch / "quota.json", {"used_node_ids": [r["id"] for r in records], "excluded_node_ids": excluded,
                                         "remaining_before": config["target_nodes"] - len(used)})
            if not records:
                ledger.commit(batch_index, report["batches"][-1], None)
                if phased:
                    teacher.call("phase", {"action": "updated", "done": False})
                continue
            actions = [
                from_collector_record(
                    r,
                    student_tokenizer=st,
                    teacher_tokenizer=tt,
                    validate_record=lambda record: validate_reference_record(record, expected_mode=config.get("teacher_context_mode")),
                    snapshot_id=current,
                    eos_ids=eos,
                )
                for r in records
            ]
            if phased:
                differences=[]
                for action,record in list(zip(actions,records))[:4]:
                    q=teacher.call('validate_raw',dict(prompt=list(action.teacher_prompt_ids),target=list(action.target_ids)))['teacher_sampled']
                    old=record['teacher_logprobs']
                    if len(q)!=len(old):raise ValueError('vLLM/HF target alignment mismatch')
                    differences.extend(abs(x-y) for x,y in zip(q,old))
                import math
                numeric=dict(tokens=len(differences),mean_abs=sum(differences)/len(differences),max_abs=max(differences),mean_abs_tolerance=.1)
                atomic(batch/'teacher_cross_engine.json',numeric)
                if not all(math.isfinite(x) for x in differences) or numeric['mean_abs']>=.1:
                    raise ValueError('vLLM/HF raw score discrepancy: see teacher_cross_engine.json')
            assert not used.intersection(a.node_id for a in actions)
            if config.get("school_p0"):
                from budgetsi.top16 import school_actor_config
                cfg = school_actor_config(len(actions), max_context=config["context"])
            else:
                cfg = actor_config(len(actions))
            service = LocalActorService(
                DataParallelPPOActor(cfg, student, optimizer), config["temperature"]
            )
            if config.get("school_p0"):
                from budgetsi.identity_division import enable_t1_identity_repair
                enable_t1_identity_repair(service.actor)
            event("scoring_opd", batch=batch_index, selected_nodes=len(actions))
            profile_mark(out, "exact_scoring_start", batch_index)
            update_started = time.monotonic()
            # Score one action at a time; retain only T x 16 CPU results.
            # Updating still uses the unchanged upstream full minibatch accumulation.
            scores = []
            for action in actions:
                scores.extend(score_with_actor(service, teacher, [action], variant=variant, snapshot_id=current,
                                               eos_ids=eos, pad_id=st.pad_token_id))
            profile_mark(out, "exact_scoring_end", batch_index)
            settings = config.get('diagnostics')
            if settings:
                step = len(report['updates'])
                if step < settings['early_steps'] or step % settings['later_interval'] == 0:
                    from budgetsi.diagnostic_run import diagnose
                    atomic(batch/'diagnostics.json',diagnose(student,teacher,actions,settings,step))
            profile_mark(out, "diagnostics_end", batch_index)
            event("updating", batch=batch_index)
            # Release inactive generation/scoring cache before the unchanged backward.
            torch.cuda.empty_cache()
            profile_mark(out, "actor_start", batch_index)
            before = adapter_state(student)
            # Execution-only saved-tensor placement; same upstream loss/microbatches.
            with torch.autograd.graph.save_on_cpu(pin_memory=True) if student_replica is not None else nullcontext():
                metrics = update_actor(
                    service,
                    actions,
                    scores,
                    variant=variant,
                    actor_config=cfg,
                    snapshot_id=current,
                    eos_ids=eos,
                    pad_id=st.pad_token_id,
                )
            profile_mark(out, "actor_end", batch_index)
            after = adapter_state(student)
            changed = sum(not torch.equal(before[k], after[k]) for k in before)
            assert changed and all(torch.isfinite(v).all() for v in after.values())
            assert all(
                p._version == frozen[k] and p.grad is None
                for k, p in student.named_parameters()
                if not p.requires_grad
            )
            if remote_cfg:
                atomic(batch / "teacher_status.json", teacher.assert_frozen())
            else:
                assert all(
                    p._version == teacher_versions[k] and p.grad is None
                    for k, p in teacher.named_parameters()
                )
            update_dir = batch / "update"
            update_dir.mkdir()
            student.save_pretrained(update_dir / "adapter", safe_serialization=True)
            torch.save(optimizer.state_dict(), update_dir / "optimizer.pt")
            optimizer.zero_grad(set_to_none=True)
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
                "opd_variant": variant.contract(),
                "checkpoint": str(update_dir.relative_to(out)),
                "batch": batch_index,
                "snapshot_before": current,
                "snapshot_after": previous_snapshot,
                "nodes": [a.node_id for a in actions],
                "changed_adapter_tensors": changed,
                "optimizer_step": next(iter(steps)),
                "adapter_restore_exact": True,
                "optimizer_restore_exact": True,
                "update_and_restore_seconds": time.monotonic() - update_started,
                "collection_seconds": report["batches"][-1]["collection_seconds"],
                "student_prompt_lengths": [len(a.student_prompt_ids) for a in actions],
                "teacher_prompt_lengths": [len(a.teacher_prompt_ids) for a in actions],
                "target_lengths": [len(a.target_ids) for a in actions],
                "metrics": metrics,
                "teacher_timing": ({
                    "requests": len(teacher.timings[teacher_timing_start:]),
                    "rpc_seconds_sum": sum(r['rpc_seconds'] for r in teacher.timings[teacher_timing_start:]),
                    "admission_queue_seconds_sum": sum(r.get('admission_queue_seconds',0) for r in teacher.timings[teacher_timing_start:]),
                    "backend_roundtrip_seconds_sum": sum(r.get('backend_roundtrip_seconds',0) for r in teacher.timings[teacher_timing_start:]),
                    "scoring_barrier_seconds": report['batches'][-1].get('teacher_scoring_barrier_seconds',0),
                } if phased else {}),
                "adapter_file_sha256": file_hash(
                    update_dir / "adapter/adapter_model.safetensors"
                ),
                "optimizer_file_sha256": file_hash(update_dir / "optimizer.pt"),
                "peak_allocated_gib": [
                    torch.cuda.max_memory_allocated(i) / 2**30 for i in range(torch.cuda.device_count())
                ],
                "peak_reserved_gib": [
                    torch.cuda.max_memory_reserved(i) / 2**30 for i in range(torch.cuda.device_count())
                ],
            }
            atomic(update_dir / "result.json", receipt)
            ledger.commit(batch_index, report["batches"][-1], receipt)
            if telemetry:
                publish(telemetry,receipt,batch,config)
            report["updates"].append(receipt)
            used = ledger.used
            report["used_nodes"] = len(used)
            del scores, before, after, expected_optimizer
            torch.cuda.empty_cache()
            profile_mark(out, "checkpoint_end", batch_index)
            event("batch_passed", batch=batch_index)
            if phased:
                event("waiting_for_shared_teacher_generation", batch=batch_index)
                teacher.call("phase", {"action": "updated", "done": len(used)==config["target_nodes"]})
        assert len(used) == config["target_nodes"], "Node quota not reached within declared batch bound"
        event("completed", status="completed", used_nodes=len(used),
              optimizer_updates=len(report["updates"]),
              peak_allocated_gib=[torch.cuda.max_memory_allocated(i) / 2**30 for i in range(torch.cuda.device_count())],
              peak_reserved_gib=[torch.cuda.max_memory_reserved(i) / 2**30 for i in range(torch.cuda.device_count())])
    except BaseException as error:
        if phased and teacher is not None:
            try:
                teacher.call("abort", {"reason": "student_failed"})
            except Exception:
                pass
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
            if getattr(engine, "parallel", False):
                engine.close()
        if student_replica is not None:
            student_replica.close()
        if student_score_replica is not None:
            student_score_replica.close()
        if telemetry:
            telemetry.finish(exit_code=0 if report.get("status") == "completed" else 1)
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
