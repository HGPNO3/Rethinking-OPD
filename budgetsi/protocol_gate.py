"""Validate the human-approved BudgetSI formal experiment protocol.

Formal launchers must call this module before starting model servers or paid API
work.  A proposal, discussion note, or partially completed review is never an
approval.  The approval packet binds the human decisions to the exact manifest,
model config, and Git commit used by the run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "budgetsi_protocol_gate_v1"
APPROVED_STATUS = "approved_by_hg"
REQUIRED_DECISIONS = (
    "student_backbone",
    "training_partner",
    "evaluation_partner",
    "environment_model",
    "judge_policy",
    "scene_profile_pairing",
    "seeds_repeats_decoding",
    "metrics_statistics_stopping",
    "baseline_delta",
    "manifest_config_hash",
)


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_hash(document: dict[str, Any]) -> str:
    payload = dict(document)
    payload.pop("manifest_sha256", None)
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def approval_hash(document: dict[str, Any]) -> str:
    payload = dict(document)
    payload.pop("approval_sha256", None)
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def model_identity(value: str) -> str:
    identity = value.strip()
    if identity.startswith("custom/"):
        identity = identity[len("custom/") :]
    if "@" in identity:
        identity = identity.split("@", 1)[0]
    return identity


def nonempty(value: Any) -> bool:
    if isinstance(value, str):
        return bool(value.strip()) and value.strip().lower() not in {
            "pending",
            "tbd",
            "todo",
            "default",
            "unspecified",
        }
    if isinstance(value, list):
        return bool(value) and all(nonempty(item) for item in value)
    if isinstance(value, dict):
        return bool(value) and all(nonempty(item) for item in value.values())
    return value is not None


def require_fields(
    decision_id: str, selection: dict[str, Any], fields: tuple[str, ...], errors: list[str]
) -> None:
    for field in fields:
        if not nonempty(selection.get(field)):
            errors.append(f"decisions.{decision_id}.selection.{field} is required")


def validate_decision_shapes(document: dict[str, Any], errors: list[str]) -> None:
    decisions = document.get("decisions")
    if not isinstance(decisions, dict):
        errors.append("decisions must be an object")
        return

    unknown = sorted(set(decisions) - set(REQUIRED_DECISIONS))
    missing = sorted(set(REQUIRED_DECISIONS) - set(decisions))
    if missing:
        errors.append(f"missing decisions: {missing}")
    if unknown:
        errors.append(f"unknown decisions: {unknown}")

    required_fields = {
        "student_backbone": (
            "primary_model_id",
            "transfer_model_id",
            "external_anchor_ids",
        ),
        "training_partner": (
            "model_id",
            "checkpoint_policy",
            "update_policy",
        ),
        "evaluation_partner": (
            "primary_model_id",
            "robustness_partner_ids",
        ),
        "environment_model": ("model_id", "role_boundary"),
        "judge_policy": (
            "manifest_judge_model_id",
            "backbone_assignment",
            "samples",
            "aggregation",
            "second_judge_policy",
        ),
        "scene_profile_pairing": (
            "training_scene_source",
            "evaluation_scene_source",
            "manifest_pairing_method",
            "hard14_role",
        ),
        "seeds_repeats_decoding": (
            "training_seeds",
            "evaluation_repeats",
            "decoding_seed_policy",
            "unsupported_seed_policy",
            "manifest_rollout_seed_support",
        ),
        "metrics_statistics_stopping": (
            "primary_metric",
            "secondary_metric",
            "statistical_unit",
            "analysis",
            "stop_conditions",
        ),
        "baseline_delta": (
            "closest_baseline",
            "adopted_protocol",
            "deliberate_changes",
        ),
        "manifest_config_hash": (
            "manifest_path",
            "manifest_file_sha256",
            "manifest_canonical_sha256",
            "model_config_path",
            "model_config_file_sha256",
            "git_commit",
        ),
    }

    for decision_id in REQUIRED_DECISIONS:
        decision = decisions.get(decision_id)
        if not isinstance(decision, dict):
            continue
        if decision.get("status") != APPROVED_STATUS:
            errors.append(f"decisions.{decision_id}.status must be {APPROVED_STATUS}")
        selection = decision.get("selection")
        if not isinstance(selection, dict):
            errors.append(f"decisions.{decision_id}.selection must be an object")
            continue
        require_fields(decision_id, selection, required_fields[decision_id], errors)
        if not nonempty(decision.get("confirmed_at")):
            errors.append(f"decisions.{decision_id}.confirmed_at is required")

    seeds = ((decisions.get("seeds_repeats_decoding") or {}).get("selection") or {})
    training_seeds = seeds.get("training_seeds")
    if isinstance(training_seeds, list) and len(set(training_seeds)) < 1:
        errors.append("at least one distinct training seed is required")
    repeats = seeds.get("evaluation_repeats")
    if isinstance(repeats, int) and repeats < 1:
        errors.append("evaluation_repeats must be at least 1")

    training_partner = ((decisions.get("training_partner") or {}).get("selection") or {})
    if training_partner and training_partner.get("update_policy") not in {"frozen", "never_update", "frozen_within_batch_refresh_between_batches"}:
        errors.append("training partner must remain frozen during a formal comparison")

    judge = ((decisions.get("judge_policy") or {}).get("selection") or {})
    samples = judge.get("samples")
    if isinstance(samples, int) and samples < 1:
        errors.append("judge samples must be at least 1")
    if judge:
        aggregation = judge.get("aggregation")
        if samples == 1 and aggregation != "single":
            errors.append("a one-sample formal judge must use single aggregation")
        elif samples != 1 and aggregation != "median":
            errors.append("multi-sample formal judge aggregation must be median")


def git_commit(repo_root: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True
    ).strip()


def git_status(repo_root: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo_root), "status", "--porcelain", "--untracked-files=all"],
        text=True,
    ).strip()


def validate_bindings(
    document: dict[str, Any],
    manifest_path: Path,
    model_config_path: Path,
    repo_root: Path,
    errors: list[str],
) -> dict[str, Any]:
    decisions = document.get("decisions") or {}
    selected_hashes = ((decisions.get("manifest_config_hash") or {}).get("selection") or {})
    manifest = load_json(manifest_path)
    model_config = load_json(model_config_path)
    # Version-specific method validation is additional to artifact hashing.
    # Historical packets retain their original identity; they cannot approve v3.
    two_teacher_version = "direct_two_teacher_no_thinking_v3"
    requested_method = document.get("method_schema_version")
    actual_method = model_config.get("schema_version")
    if requested_method is not None and requested_method != actual_method:
        errors.append("method_schema_version differs between approval and model config")
    if actual_method == two_teacher_version:
        # The v3 CLI is an engineering harness, not an accepted formal runtime.
        # Nonempty placeholder sections must never stand in for typed model,
        # optimizer, tokenizer and official-environment acceptance contracts.
        errors.append("two-teacher formal runtime adapter has not been accepted; "
                      "engineering-only configuration cannot authorize a formal run")
        if requested_method != two_teacher_version:
            errors.append("two-teacher run requires explicit method_schema_version approval binding")
        from pipeline.v2.direct_multi_teacher.current_run import validate_config
        try:
            pending = validate_config(model_config)
            if pending:
                errors.append("two-teacher method configuration pending: " + ", ".join(pending))
        except (ValueError, TypeError, AttributeError) as exc:
            errors.append("two-teacher method configuration invalid: " + str(exc))

    if actual_method in {"two_teacher_non_thinking_online_v1", "two_teacher_reference_online_v2", "two_teacher_prefix_only_online_v3"}:
        if requested_method != actual_method:
            errors.append("Online runtime requires explicit method approval")
        expected = {"thinking": False, "specialties": ["expression", "turn"],
                    "target_nodes": 1000, "teacher_candidates_per_specialty": 1,
                    "partner_samples_per_unique_action": 1,
                    "selection": "positive_action_ig_per_token_v1_20260916",
                    "training_partner_policy": "frozen_within_batch_refresh_between_batches",
                    "loss": "sampled_reverse_kl_original_student_tokens",
                    "train_role": "A", "temperature": 0.7}
        for key, value in expected.items():
            if model_config.get(key) != value:
                errors.append("Online runtime config mismatch: " + key)
        allowed=[{'student':'Qwen/Qwen3-4B','teacher':'Qwen/Qwen3-14B','partner':'Qwen/Qwen3-4B','IG':'Qwen/Qwen3-4B'}, {'student':'Qwen/Qwen3.5-4B','teacher':'Qwen/Qwen3.5-27B','partner':'Qwen/Qwen3.5-4B','IG':'Qwen/Qwen3.5-4B'}]
        if model_config.get('models') not in allowed or model_config.get('models') != document.get('approved_model_roles'):
            errors.append("Online runtime model identities differ from explicitly approved combination")
        if not document.get('delegated_engineering_binding'):
            errors.append("Missing explicit delegated engineering binding")
        if ((decisions.get('training_partner') or {}).get('selection') or {}).get('update_policy') != expected['training_partner_policy']:
            errors.append("Online training partner refresh approval differs")

    if actual_method in {"two_teacher_reference_online_v2", "two_teacher_prefix_only_online_v3"}:
        from runner import PROMPT_VERSION, prompt_binding
        if model_config.get("opd_teacher_context") != "selected_reference_without_full_original_draft":
            errors.append("Reference OPD context config mismatch")
        if model_config.get("reference_prompt_version") != PROMPT_VERSION or model_config.get("reference_prompt_binding") != prompt_binding():
            errors.append("Reference OPD prompt binding mismatch")
        if document.get("reference_opd_authorized") is not True:
            errors.append("Reference OPD requires explicit new user authorization")

    if actual_method == "two_teacher_prefix_only_online_v3":
        if model_config.get("candidate_teacher_context") != "target_time_visible_messages_only":
            errors.append("Prefix-only candidate context config mismatch")
        if document.get("prefix_only_candidates_authorized") is not True:
            errors.append("Prefix-only candidates require explicit user authorization")

    if model_config.get("ig_target_mode") != "verbatim_goal":
        errors.append("New run requires exact verbatim goal target")
    if model_config.get("candidate_teacher_context") != "visible_history_and_original_student_action":
        errors.append("New run requires candidate teacher to see original action")
    if document.get("verbatim_goal_positive_ig_authorized") is not True:
        errors.append("New goal and selection rules require explicit approval")

    actual_manifest_file_hash = sha256_file(manifest_path)
    actual_model_config_hash = sha256_file(model_config_path)
    actual_git_commit = git_commit(repo_root)
    actual_git_status = git_status(repo_root)
    actual_manifest_canonical_hash = manifest_hash(manifest)
    recorded_manifest_canonical_hash = str(manifest.get("manifest_sha256") or "")

    if recorded_manifest_canonical_hash != actual_manifest_canonical_hash:
        errors.append("manifest_sha256 does not match independently recomputed canonical content")
    if actual_git_status:
        errors.append("Git worktree must be clean, including untracked files, before a formal launch")

    approved_manifest_path = Path(str(selected_hashes.get("manifest_path") or "")).expanduser()
    approved_model_config_path = Path(str(selected_hashes.get("model_config_path") or "")).expanduser()
    if not approved_manifest_path.is_absolute():
        approved_manifest_path = repo_root / approved_manifest_path
    if not approved_model_config_path.is_absolute():
        approved_model_config_path = repo_root / approved_model_config_path
    if approved_manifest_path.resolve() != manifest_path.expanduser().resolve():
        errors.append("manifest path differs between approval and launcher")
    if approved_model_config_path.resolve() != model_config_path.expanduser().resolve():
        errors.append("model config path differs between approval and launcher")

    expected_pairs = (
        ("manifest_file_sha256", selected_hashes.get("manifest_file_sha256"), actual_manifest_file_hash),
        (
            "manifest_canonical_sha256",
            selected_hashes.get("manifest_canonical_sha256"),
            actual_manifest_canonical_hash,
        ),
        (
            "model_config_file_sha256",
            selected_hashes.get("model_config_file_sha256"),
            actual_model_config_hash,
        ),
        ("git_commit", selected_hashes.get("git_commit"), actual_git_commit),
    )
    for label, expected, actual in expected_pairs:
        if expected != actual:
            errors.append(f"{label} mismatch: approved={expected!r} actual={actual!r}")

    protocol = manifest.get("protocol") or {}
    evaluation_partner = ((decisions.get("evaluation_partner") or {}).get("selection") or {})
    environment = ((decisions.get("environment_model") or {}).get("selection") or {})
    judge = ((decisions.get("judge_policy") or {}).get("selection") or {})
    pairing = ((decisions.get("scene_profile_pairing") or {}).get("selection") or {})
    seeds = ((decisions.get("seeds_repeats_decoding") or {}).get("selection") or {})

    identity_pairs = (
        (
            "evaluation partner",
            evaluation_partner.get("primary_model_id"),
            protocol.get("partner_model_id"),
        ),
        ("environment model", environment.get("model_id"), protocol.get("environment_model_id")),
        (
            "manifest judge",
            judge.get("manifest_judge_model_id"),
            protocol.get("judge_model_id"),
        ),
    )
    for label, approved, actual in identity_pairs:
        if model_identity(str(approved or "")) != model_identity(str(actual or "")):
            errors.append(f"{label} differs between approval and manifest")

    if pairing.get("manifest_pairing_method") != (manifest.get("profile_split") or {}).get("method"):
        errors.append("scene/profile pairing method differs between approval and manifest")
    if seeds.get("evaluation_repeats") != manifest.get("repeats_per_scene"):
        errors.append("evaluation repeat count differs between approval and manifest")
    if judge.get("samples") != protocol.get("judge_samples"):
        errors.append("judge sample count differs between approval and manifest")
    manifest_seed_support = (protocol.get("decoding_config") or {}).get("rollout_seed_support")
    if seeds.get("manifest_rollout_seed_support") != manifest_seed_support:
        errors.append("rollout decoding seed support differs between approval and manifest")

    return {
        "manifest_file_sha256": actual_manifest_file_hash,
        "manifest_canonical_sha256": actual_manifest_canonical_hash,
        "model_config_file_sha256": actual_model_config_hash,
        "git_commit": actual_git_commit,
        "git_clean": not actual_git_status,
    }


def verify_approval(
    document: dict[str, Any],
    manifest_path: Path | None = None,
    model_config_path: Path | None = None,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    errors: list[str] = []
    if document.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version must be {SCHEMA_VERSION}")
    if document.get("status") != APPROVED_STATUS:
        errors.append(f"status must be {APPROVED_STATUS}")
    if document.get("approved_by") != "HG":
        errors.append("approved_by must be exactly HG")
    if not nonempty(document.get("approved_at")):
        errors.append("approved_at is required")

    validate_decision_shapes(document, errors)

    expected_hash = approval_hash(document)
    if document.get("approval_sha256") != expected_hash:
        errors.append("approval_sha256 does not match canonical approval content")

    bindings: dict[str, Any] = {}
    supplied = (manifest_path, model_config_path, repo_root)
    if any(item is not None for item in supplied) and not all(item is not None for item in supplied):
        errors.append("manifest, model config, and repo root must be supplied together")
    elif all(item is not None for item in supplied):
        assert manifest_path is not None
        assert model_config_path is not None
        assert repo_root is not None
        try:
            bindings = validate_bindings(
                document, manifest_path, model_config_path, repo_root, errors
            )
        except (OSError, ValueError, json.JSONDecodeError, subprocess.CalledProcessError) as exc:
            errors.append(f"binding verification failed: {exc}")

    return {
        "ok": not errors,
        "schema_version": document.get("schema_version"),
        "protocol_id": document.get("protocol_id"),
        "approval_sha256": document.get("approval_sha256"),
        "verified_bindings": bindings,
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    verify_parser = subparsers.add_parser("verify", help="verify approval and bound artifacts")
    verify_parser.add_argument("--approval", required=True)
    verify_parser.add_argument("--manifest")
    verify_parser.add_argument("--model-config")
    verify_parser.add_argument("--repo-root")
    verify_parser.add_argument("--output")

    hash_parser = subparsers.add_parser("hash", help="print the canonical approval hash")
    hash_parser.add_argument("--approval", required=True)

    args = parser.parse_args()
    document = load_json(Path(args.approval))
    if args.command == "hash":
        print(approval_hash(document))
        return

    optional_paths = (args.manifest, args.model_config, args.repo_root)
    if any(optional_paths) and not all(optional_paths):
        parser.error("--manifest, --model-config, and --repo-root must be used together")
    report = verify_approval(
        document,
        Path(args.manifest) if args.manifest else None,
        Path(args.model_config) if args.model_config else None,
        Path(args.repo_root) if args.repo_root else None,
    )
    if args.output:
        write_json(Path(args.output), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
