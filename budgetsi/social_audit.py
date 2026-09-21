"""Independent environment replay of finished acceptance records; no model calls."""

import argparse
import asyncio
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ["SOTOPIA_STORAGE_BACKEND"] = "local"
sys.path.insert(0, str(Path(__file__).parent / "social_protocol"))
from adapter import create_session
from runner import (
    prompt_binding,
    seed_for,
    select,
    validate_reference_record,
)
from transformers import AutoTokenizer


def read(path):
    return json.loads(path.read_text())


async def audit(path):
    config = read(path / "config.json")
    report = read(path / "report.json")
    assert report["status"] == "passed" and len(report["updates"]) == 2
    st = AutoTokenizer.from_pretrained(config["student"], local_files_only=True)
    tt = AutoTokenizer.from_pretrained(config["teacher"], local_files_only=True)
    assert st.get_vocab() == tt.get_vocab()

    def render(tok, messages):
        return tok.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_dict=False,
        )

    batches = []
    for batch in sorted(path.glob("batch_*")):
        inputs = read(batch / "inputs.json")["scenes"]
        rows = {r["id"]: r for r in read(batch / "rollout/nodes.json")}
        records = {r["id"]: r for r in read(batch / "rollout/selected_records.json")}
        traces = {r["id"]: r for r in read(batch / "rollout/dialogues.json")}
        observed = set()
        events = 0
        for scene in inputs:
            session = create_session(scene)
            trace = traces[scene["id"]]
            assert trace["complete"]
            for index, event in enumerate(trace["events"]):
                role = session.active_role
                assert role == event["role"]
                messages = session.messages(role)
                generation = event["generation"]
                assert render(st, messages) == generation["prompt_token_ids"]
                assert generation["behavior_temperature"] == 0.7
                key = f"{scene['id']}:{index}:{role}"
                if role == 0:
                    node = rows[key]
                    assert node["original"] == generation
                    assert node["target_text"] == session.goal(0)
                    if key in records:
                        record = records[key]
                        validate_reference_record(record)
                        chosen, _ = select(
                            node["candidates"], seed_for(config["seed"], key, "select")
                        )
                        assert chosen == node["selected"] == record["specialty"]
                        assert record["visible_messages"] == messages
                        assert record["student_prefix"] == render(st, messages)
                        assert record["target_ids"] == generation["generated_token_ids"]
                        assert record["teacher_score"]["prompt_token_ids"] == render(
                            tt, record["teacher_messages"]
                        )
                        reference = next(
                            c["generation"]["action"]
                            for c in node["candidates"]
                            if c["specialty"] == chosen
                        )
                        assert reference == record["reference_action"]
                        observed.add(key)
                await session.step(generation["action"])
                events += 1
            assert session.done
        assert observed == records.keys()
        update_path = batch / "update/result.json"
        if update_path.exists():
            update = read(update_path)
            assert set(update["nodes"]) == records.keys()
            assert all(
                r["snapshot_id"] == update["snapshot_before"] for r in records.values()
            )
        batches.append(
            {
                "batch": batch.name,
                "replayed_events": events,
                "selected_records": len(records),
                "causal_prompt_match": True,
            }
        )
    first, second = report["updates"]
    assert first["snapshot_after"] == second["snapshot_before"]
    assert first["optimizer_step"] == 1 and second["optimizer_step"] == 2
    requests = [
        json.loads(line)
        for line in (path / "engine_requests.jsonl").read_text().splitlines()
    ]
    assert {r["snapshot"] for r in requests if r["model"] == "student"} == {
        first["snapshot_before"],
        first["snapshot_after"],
    }
    result = {
        "status": "passed",
        "scope": "offline independent environment replay and receipt audit",
        "prompt_binding": prompt_binding(),
        "batches": batches,
        "fresh_second_snapshot_and_optimizer_step": True,
        "report_sha256": hashlib.sha256(
            (path / "report.json").read_bytes()
        ).hexdigest(),
    }
    (path / "audit.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    asyncio.run(audit(Path(parser.parse_args().run)))
