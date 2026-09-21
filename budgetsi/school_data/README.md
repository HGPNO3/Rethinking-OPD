# School P0 train200 data, version 1

User-approved decision (2026-09-20): all 200 non-test public scenarios train; no heldout dev20; each P0 arm has a budget of **3000 effective selected OPD nodes**, not 3000 optimizer updates. Existing test90 / 450 initializations are fixed.

## What was built

The pinned public `cmu-lti/sotopia-pi` episode index has 290 environment IDs. Excluding exactly the current benchmark's 90 IDs leaves **200 environments and 1129 distinct ordered (environment, Agent1, Agent2) configurations**. The previous20 dev are now included by explicit user authorization. Counts per environment: 20 with4 pairs,45 with5,121 with6,14 with7.

`inputs_200.json` contains only authoritative environment/character initializations and evidence IDs. It does not contain old dialogues, model answers, rewards or OPD targets. Both P0 arms must consume this same file and scheduling seed. Student role is fixed to0. Teacher reference remains allowed only in the reference-context arm's teacher scoring input, never the student input.

Every environment and character comes directly from the pinned public Redis dump. We wrote a narrow read-only RedisJSON module payload decoder (`extract_relationships.py`), verify the complete dump SHA256 before parsing, and validate key type/length/module ID/payload terminator/JSON pk/decompressed length. Each output preserves RDB offsets and JSON hashes for audit. No Redis database server was created or modified.

## How relationships are determined

SOTOPIA-π describes tasks using scenarios, character profiles and private goals (§2.1); its experiments generate100 tasks and collect10 character-pair interactions per task (§4). This supports varying characters within a scenario, not freely inventing relationships. [Paper](https://aclanthology.org/2024.acl-long.698.pdf)

The official `ConstraintBasedSampler._get_fit_agents_for_one_env` reads the environment relationship category, selects matching `RelationshipProfile` records, filters each role's age where non-default constraints apply, and yields the relationship record's agent1/agent2 order. It **does not filter occupation_constraint**. [Official sampler](https://github.com/sotopia-lab/sotopia/blob/main/sotopia/samplers/constraint_based_sampler.py)

Our dump contains120 native relationship records. **All1129 candidate configurations match an existing ordered (agent_1_id, agent_2_id, relationship) tuple directly**: missing0, reverse-only0, age mismatch0. We preserve existing pairs; no random role reversal, fabricated family/friend link, generated relationship story, or changed goal. Relationship record IDs are in each initialization's source evidence. Native relationship background stories are retained only in the separate audit export; we do not inject them into prompts, because the existing official adapter uses environment relationship to control profile visibility and does not inject that record's story.

188 of1129 pairs trigger our additional literal occupation-field review. These are informational metadata, **not filtered**: the official sampler did not apply this criterion; tightening it silently would change the experiment. Case/wording/occupation synonyms also mean a literal mismatch is not a proven semantic contradiction.

The source relationship enumeration is0=stranger,1=know_by_name,2=acquaintance,3=friend,4=romantic_relationship,5=family_member. Keep the exact supplied value. Official role-visible rendering handles the resulting access to profile information; do not serialize full environment/agents dictionaries into a student prompt. [Official profile schema](https://github.com/sotopia-lab/sotopia/blob/main/sotopia/database/persistent_profile.py)

## Split and scheduling contract

- Environment IDs and normalized scenario-text hashes have zero train/test overlap. Unicode normalization, whitespace folding and case folding are used. Character-similarity screening at0.85 also found zero cross-split candidates; this is a screening result, not proof of semantic independence.
- Test inputs SHA256: `e561d40a821ec4538224ced8c3912648adec5a77b0c633f31fbce1b623d10774`.
- `load_initializations()` verifies file SHA256, exact200 scenario coverage, no test IDs, no dev, role0 and relationship receipts.
- `schedule_batch(scenes,batch_index,batch_size=16,seed=20260920)` chooses16 distinct scenarios per collection, balanced round-robin across200; within each scenario, it cycles its published ordered pairs. Batch index starts0.
- Each revisit gets a distinct `scene.id = initialization_id + '.visitN'` and derived seed, preventing reused node IDs. Both arms receive identical initialization assignments; their actual on-policy dialogues may diverge after learning.
- Persist **collection batch cursor**, not optimizer step, on successful durable collection. Empty-node batches may advance collection cursor without an optimizer update. Do not change batch size or schedule seed on resume.
- Update on the collected current-student nodes, then use the updated student for the next fresh rollout batch. Never refill from historical public answers.
- 3000 nodes need not traverse all1129 configurations. Report scenario coverage, configuration coverage, conversation attempts/successes, effective nodes/tokens and optimizer steps separately. There is no dev set under this user choice; fixed probe metrics are diagnostics, not heldout model-selection scores. Final test should stay final, rather than repeatedly tuning to it.

## Rebuild and verify

From repository root, with the pinned public dump and episode index, reconstruct:

```sh
python budgetsi/school_data/extract_relationships.py /path/to/dump.rdb budgetsi/school_data/relationship_profiles.json
python budgetsi/school_data/extract_relationships.py /path/to/dump.rdb budgetsi/school_data/AgentProfile_public.json --profile-type AgentProfile
python budgetsi/school_data/extract_relationships.py /path/to/dump.rdb budgetsi/school_data/EnvironmentProfile_public.json --profile-type EnvironmentProfile
```

All three profile exports use the same verified parser. Then:

```sh
python budgetsi/school_data/build_inputs.py --episodes /path/to/sotopia_pi_episodes.jsonl --test-inputs /path/to/fixed450/inputs.json
python -m unittest budgetsi.tests.test_school_data -v
```

Run official runtime acceptance in the configured Sotopia dependency environment:

```sh
PYTHONPATH=. python budgetsi/school_data/validate_runtime.py
```

Runtime acceptance uses official AgentProfile/EnvironmentProfile constructor, reset and per-role prompt rendering for all1129 initializations; it makes no model calls. It requires the configured Sotopia environment. Its result is separate `runtime_validation.json`; absence means that acceptance has not yet run on that host.

`manifest.json` stores counts, split IDs, source hashes and relationship audit. `SHA256SUMS.json` covers the immutable data/code source files in this directory (runtime_validation.json excluded because it is a host-specific receipt).
