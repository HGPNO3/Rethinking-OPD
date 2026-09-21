# P0 actor, context and output-boundary implementation

2026-09-20. Project-side adaptation; vendored verl is not modified.

## Authoritative upstream recipe

Pinned upstream commit: `ac26e38d6f1572eb027597b48a9f4e01f6915ef8`.

`on_policy_distillation.sh` sets actor `use_dynamic_bsz=True`, `loss_agg_mode=token-mean`, microbatch fallback1, mini batch64, actor LR1e-6, rollout temperature1, teacher temperature1, no extra reference-KL loss. PPO token budget is max(32768, prompt+response); its default prompt1024+response7168 gives32768. The launch script does not explicitly set a reproducibility seed. Rollout vLLM worker defaults model seed0 (`verl/verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py:261`); dataloader seed defaultsnull and only seeds the generator if nonnull (`main_ppo.py:427-429`). These are not a single verified universal training seed.

Actor config default WD=.01 (`verl/verl/trainer/config/actor/actor.yaml:126`). P0 LR1e-5 is the explicit user override; P0 generation temperature1 and teacher score temperature1 match the upstream default. Project scenario/node schedule and explicit seeds remain required for repeatability. The root shell script's eight-rank Ray/FSDP launch, math datasets, full FP32 model training and minibatch64 are not being reproduced by the LoRA/Sotopia single-rank project loop.

## Actual dynamic normalization

The unchanged upstream `DataParallelPPOActor.update_policy` uses `prepare_dynamic_batch` to split the optimizer minibatch; it takes the token-mean vanilla loss within each microbatch, then multiplies by the microbatch's answer count divided by the full minibatch's answer count. It finally does one optimizer step after all microbatches. Formula:

`sum_g (answers_in_g / total_answers) * mean_valid_tokens_in_g(policy_loss)`.

It is generally neither the single token mean of the whole minibatch nor an invariant answer mean across arbitrary packing. Therefore both P0 arms use the same configured dynamic token budget and upstream implementation, and the documentation must not label it whole-batch token weighting. Old fixed microbatch1 behavior stays available; its equal-answer weighting is not silently retained under the new dynamic recipe.

API: `budgetsi.top16.school_actor_config(count, max_context=40960, max_token_len_per_gpu=None)`. The default token budget equals40960, which can accommodate one valid full context; both arms must use the same value. `validate_actor` recognizes the explicit school recipe, insists on one student rank, dynamic batching, nonempty minibatch, and sufficient configured token capacity. Core reward, direct advantage, vanilla loss and optimizer invocation remain upstream.

The configured token budget is not a proof of peak-memory safety: upstream balancing chooses the number of partitions from token totals and balances compute, while padded tensors can retain long sequence widths. Full40k variable-length mixed batches need a real-model memory test. Do not claim dynamic batching guarantees no OOM. The Qwen3.5 project adapter still uses no remove-padding/fused kernels/torch.compile in this first version for compatibility; these are disclosed engineering deviations from the full upstream launcher.

## Teacher-context protocol

`runner.PROMPT_VERSION=school_p0_teacher_context_v1_20260920`.

- `same_context`: original student messages deep-copied; teacher receives exactly original student prompt token IDs and generated target IDs.
- `reference_context`: copy the same messages, append only an explicitly unexecuted teacher-reference JSON data block to the final user content. Original system, previous turns, candidate-generation prompts and selection code are unchanged. Selected specialty does not inject extra OPD instructions.
- `teacher_context_mode` and `teacher_context_binding` persist in nodes/records/summary; `validate_reference_record(record, expected_mode=...)` rejects cross-arm evidence. Reference candidate metadata remains for selection audit even in the same arm, but does not enter scored prompts.
- `social_collect.py` requires `--teacher-context-mode` and accepts `--temperature`; model request and behavior probability metadata use the same supplied rollout temperature. Raw teacher probability route remains temperature1.

## Output handling

Only reviewed JSON/schema/duplicate/nonfinite-JSON and known truncation/context/thinking/EOS-boundary output errors can terminate one dialogue while retaining already executed valid nodes. A valid source action that no longer fits the teacher scoring context is executed but not counted as an OPD node. Malformed `action_type` list/dict becomes a schema error rather than a PythonTypeError. No bad answer is repaired or selectively resampled. OOM, transport/probability contract failures and unrecognized errors remain visible failures.

## Validation performed

CPU tests execute exact pinned upstream function bodies through the existing AST probe harness; decorators/type annotations and distributed/model containers are replaced for CPU fixtures, not the numerical bodies. Actual pinned dynamic partitioner, update_policy, direct advantage and vanilla loss produce a variable-length two-microbatch/one-update result. Gradients and one AdamW parameter update match an independent accumulation using the actual partition assignment, tolerance1e-12. A separate test asserts that this is not generally equal to whole-batch token averaging.

Commands:

```
python -m unittest budgetsi.tests.test_school_actor budgetsi.tests.test_school_context budgetsi.tests.test_school_output_edges budgetsi.tests.test_batch_validation
PYTHONPATH=budgetsi/social_protocol python -m unittest test_runner test_reference_opd
```

These are meaningful CPU wiring/numerical tests, not evidence of school GPU acceptance, throughput, imported runtime version compatibility or completed training. Formal launcher must still verify real variable-length model updates, fresh student snapshot serving, maximum context and W&B round-trip.
