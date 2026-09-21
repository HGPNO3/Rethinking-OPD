"""Reuse project protocol gate plus typed checks for the new upstream OPD path."""
import json
import subprocess
from pathlib import Path
from budgetsi.protocol_gate import verify_approval
from budgetsi.run_state import digest
from budgetsi.variant_spec import SCHEMA, from_config

PIN = 'ac26e38d6f1572eb027597b48a9f4e01f6915ef8'


def check_launch(config_path, approval_path, repo):
    config_path, approval_path, repo = map(lambda p: Path(p).resolve(), (config_path, approval_path, repo))
    if approval_path.is_relative_to(repo):
        raise ValueError('Approval must be stored outside tracked repository')
    config = json.loads(config_path.read_text())
    approval = json.loads(approval_path.read_text())
    school = config.get("school_p0")
    if school:
        validate_school(config, approval, repo)
    variant = from_config(config)
    if config["schema_version"] == SCHEMA and approval.get("opd_variant") != variant.contract():
        raise ValueError("Approval must explicitly bind this OPD variant")
    if config.get("remote_teacher"):
        remote = config["remote_teacher"]
        if variant.name not in {"student_top16", "union_student", "sampled_token"}:
            raise ValueError("Remote deployment is scoped to the two requested variants")
        if approval.get("remote_teacher") != remote or remote.get("transport") != ("local_loopback" if school else "ssh_loopback"):
            raise ValueError("Remote teacher deployment must be explicitly bound")
        if remote.get("endpoint") != "http://127.0.0.1:18740/":
            raise ValueError("Remote teacher requires fixed loopback SSH tunnel")
    manifest = 'training_manifest_qwen35.json' if Path(config['student']).name == 'Qwen3.5-4B' else 'training_manifest.json'
    if school:
        manifest = 'training_manifest_school.json'
    report = verify_approval(approval, repo / 'budgetsi' / manifest, config_path, repo)
    if not report['ok']:
        raise ValueError(report)
    expected = dict(schema_version=config['schema_version'], external_api=False,
                    temperature=1.0 if school else 0.7, teacher_temperature=1.0, context=40960,
                    ig_target_mode='verbatim_goal', candidate_teacher_context='visible_history_and_original_student_action',
                    upstream_commit=PIN, optimizer=dict(lr=approval.get("learning_rate_control", {}).get("lr", 1e-6), weight_decay=0.01, r=32, alpha=64, dropout=0))
    if expected["optimizer"]["lr"] not in {1e-6, 1e-5}:
        raise ValueError("Unapproved learning rate")
    if expected["optimizer"]["lr"] == 1e-5 and approval.get("learning_rate_control") != {"lr": 1e-5, "initialization": "fresh_base", "target_nodes": 3000 if school else 1000}:
        raise ValueError("Learning-rate control requires exact fresh-base authorization")
    for k, v in expected.items():
        if config.get(k) != v:
            raise ValueError(f'Unapproved setting: {k}')
    if config['formal_training']:
        if (config['target_nodes'], config['dialogues_per_batch'], config['seed']) != (3000 if school else 1000, 16, 20360915):
            raise ValueError('Formal run requires exactly1000 nodes,16 dialogues, historical formal seed')
    elif (config['target_nodes'], config['dialogues_per_batch'], config['seed']) != (3, 1, 20260919):
        raise ValueError('Engineering validation limited to3 nodes and single-dialogue batches')
    names = tuple(Path(config[k]).name for k in ('student', 'teacher'))
    if names not in {('Qwen3-4B', 'Qwen3-14B'), ('Qwen3.5-4B', 'Qwen3.5-27B')}:
        raise ValueError('Unapproved model identity')
    if names[0] == 'Qwen3.5-4B':
        if not config.get('remote_teacher') or approval.get('model_pair') != list(names):
            raise ValueError('Qwen3.5 pair requires explicit bound approval and remote two-GPU teacher')
    if names[0] == 'Qwen3.5-4B':
        if config.get('diagnostics') != dict(k=16,max_actions=4,early_steps=30,later_interval=5):
            raise ValueError('Qwen3.5 runs require the bound early diagnostic schedule')
        logging = config.get('logging', {})
        if logging.get('mode') not in {'offline', 'online'} or not logging.get('project'):
            raise ValueError('Qwen3.5 runs require active W&B logging')
    parallel = config.get('collection_parallel')
    if parallel:
        if approval.get('collection_parallel') != parallel:
            raise ValueError('Parallel backend/settings must be explicitly bound')
        if parallel.get('backend') != 'hf_seeded_batch_v1' or set(parallel) != {'backend', 'dialogues', 'batch_size', 'wait_ms'}:
            raise ValueError('Invalid parallel configuration')
        if any(type(parallel[k]) is not int for k in ('dialogues', 'batch_size', 'wait_ms')):
            raise ValueError('Parallel limits must be integers')
        if not (1 <= parallel['dialogues'] <= 16 and 1 <= parallel['batch_size'] <= 8 and 0 <= parallel['wait_ms'] <= 100):
            raise ValueError('Parallel limits outside validated bounds')
    inference = config.get('teacher_inference')
    if inference:
        if approval.get('teacher_inference') != inference:
            raise ValueError('Teacher inference must be explicitly bound')
        if (inference.get('backend') != ('vllm_persistent_hf_v1' if school else 'vllm_phased_hf_v1') or
            (inference.get('tp'), inference.get('pp')) not in {(2,1),(1,2)} or
            inference.get('max_num_seqs') not in {4,8,16} or
            inference.get('max_num_batched_tokens') not in {512,4096} or
            inference.get('gpu_memory_utilization') not in {.80,.90} or
            type(inference.get('language_model_only', False)) is not bool):
            raise ValueError('Unvalidated teacher inference settings')
    benchmark = config.get('teacher_benchmark')
    if benchmark:
        if (approval.get('teacher_benchmark') != benchmark or config['formal_training'] or
            benchmark.get('backend') not in {'hf','vllm'} or benchmark.get('count') != 64 or
            benchmark.get('concurrency') != 24 or len(benchmark.get('requests_sha256','')) != 64):
            raise ValueError('Unbound teacher benchmark')
    subprocess.run(['git', '-C', str(repo), 'diff', '--exit-code', PIN, '--', 'verl', 'on_policy_distillation.sh'], check=True, capture_output=True)
    for name, expected_hash in config['source_provenance'].items():
        if digest(repo / 'budgetsi/social_protocol' / name) != expected_hash:
            raise ValueError('Social protocol changed: ' + name)
    return report


def validate_school(config, approval, repo):
    """Narrow, hash-bound P0 extension; old approvals cannot start this study."""
    expected = dict(version="school_p0_v1", training_scenarios=200,
                    development_scenarios=0, test_scenarios=90,
                    initialization="fresh_base", formal_target_nodes=3000,
                    actor_recipe="upstream_dynamic_token_mean_v1")
    if config["school_p0"] != expected or approval.get("school_p0") != expected:
        raise ValueError("Unbound school P0 protocol")
    mode = config.get("teacher_context_mode")
    if mode not in {"same_context", "reference_context"} or approval.get("teacher_context_mode") != mode:
        raise ValueError("Unbound teacher context ablation")
    if from_config(config).name != "sampled_token":
        raise ValueError("School P0 requires sampled-token K0")
    data = config.get("training_data", {})
    if approval.get("training_data") != data or set(data) != {"path", "sha256"}:
        raise ValueError("Training data must be explicitly hash-bound")
    path = (repo / data["path"]).resolve()
    if not path.is_relative_to(repo) or digest(path) != data["sha256"]:
        raise ValueError("School training data changed")
    replica = config.get('student_replica')
    if replica is not None:
        expected_replica = dict(backend='hf_seeded_replica_v1', cuda_visible_devices='6' if mode == 'same_context' else '7')
        if replica != expected_replica or approval.get('student_replica') != replica or not config.get('collection_parallel'):
            raise ValueError('Unbound student generation replica')
    if config.get("optimizer", {}).get("lr") != 1e-5:
        raise ValueError("HG confirmed P0 learning rate 1e-5")
    score = config.get('student_score_replica')
    if score is not None:
        expected_score = dict(backend='hf_frozen_score_replica_v1',
                              cuda_visible_devices='2' if mode == 'same_context' else '3',
                              lease_path='/hpc2ssd/JH_DATA/spooler/lgong265/school_opd_20260920/launch/p0_20260920/scoring_gpu.lease')
        if score != expected_score or approval.get('student_score_replica') != score or replica is None:
            raise ValueError('Unbound student scoring replica')
    logging = config.get("logging", {})
    if logging.get("entity") != "2364713056-hkust" or logging.get("project") != "budgetsi-qwen35-opd":
        raise ValueError("School logging destination differs from authorization")
