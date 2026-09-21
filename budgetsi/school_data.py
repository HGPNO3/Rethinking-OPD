"""Verified train200 initializations and resumable scene-balanced scheduling.

A schedule cursor is a collection batch index, not an optimizer step: failed
or empty collections must still have their completed cursor checkpointed.
Generation remains fresh after every committed student update. This module
never loads public historical answers, generates text or calls a model API.
"""
from __future__ import annotations
import copy,hashlib,json
from collections import defaultdict
from pathlib import Path

DATA_DIR=Path(__file__).with_name('school_data')
DEFAULT_INPUTS=DATA_DIR/'inputs_200.json'
SCHEDULE_VERSION='scene_balanced_ordered_pair_round_robin_v1'

def _hash(*parts):
    return hashlib.sha256(json.dumps(parts,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()

def load_initializations(path=None):
    path=Path(path) if path is not None else DEFAULT_INPUTS
    manifest_path=path.with_name('manifest.json')
    if not manifest_path.is_file():raise ValueError('School input manifest/checksum missing')
    manifest=json.loads(manifest_path.read_text());raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=manifest['inputs_200_sha256']:raise ValueError('School input SHA256 mismatch')
    payload=json.loads(raw)
    if payload.get('schema_version')!='budgetsi_school_pi200_v1':raise ValueError('Unsupported school input schema')
    scenes=payload['scenes'];ids=set();env_ids=set()
    for scene in scenes:
        if scene['id'] in ids:raise ValueError('Duplicate initialization ID')
        ids.add(scene['id']);env=scene['environment'];env_ids.add(env['pk'])
        if scene.get('student_agent_index')!=0:raise ValueError('School P0 fixes student role 0')
        if len(scene['agents'])!=2 or len(env['agent_goals'])!=2:raise ValueError('Expected two ordered roles')
        if scene['agents'][0]['pk']==scene['agents'][1]['pk']:raise ValueError('Two distinct agents required')
        if not scene['source'].get('relationship_profile_ids'):raise ValueError('Missing native relationship verification')
        if any(key in scene for key in ['raw_messages','messages','social_interactions','rewards','target','target_ids','reference_action']):raise ValueError('Historical training target forbidden in initialization')
    if len(env_ids)!=200 or len(scenes)!=manifest['train_ordered_configurations']:raise ValueError('School P0 requires verified full train200 coverage')
    if env_ids!=set(manifest['train_environment_ids']):raise ValueError('Train environment manifest mismatch')
    if env_ids&set(manifest['test_environment_ids']):raise ValueError('Train/test environment overlap')
    if manifest.get('dev_scenarios')!=0:raise ValueError('User-approved school split does not reserve dev scenarios')
    return scenes

def schedule_batch(scenes,batch_index,batch_size=16,seed=20260920):
    """Return independent scene copies; same settings reproduce both P0 arms.

    Across all batches each scene's occurrence counts differ by at most one.
    Pair selection rotates on each visit to that scene. Every visit receives a
    distinct scene ID and seed, preventing node-ID collisions on later passes.
    Persist batch_index+1 only when collection is durably recorded; keep the
    seed and batch_size immutable on resume.
    """
    if type(batch_index)is not int or batch_index<0:raise ValueError('batch_index must be a nonnegative integer')
    if type(batch_size)is not int or batch_size<1:raise ValueError('batch_size must be a positive integer')
    groups=defaultdict(list);seen=set()
    for scene in scenes:
        if scene['id'] in seen:raise ValueError('Duplicate initialization ID')
        seen.add(scene['id']);groups[scene['environment']['pk']].append(scene)
    if not groups or batch_size>len(groups):raise ValueError('A batch requires distinct available scenarios')
    env_order=sorted(groups,key=lambda eid:_hash(SCHEDULE_VERSION,seed,'environment',eid))
    for eid in groups:groups[eid].sort(key=lambda s:_hash(SCHEDULE_VERSION,seed,'pair',s['id']))
    result=[];n_env=len(env_order)
    for global_index in range(batch_index*batch_size,(batch_index+1)*batch_size):
        eid=env_order[global_index%n_env];visit=global_index//n_env
        original=groups[eid][visit%len(groups[eid])];scene=copy.deepcopy(original)
        scene['id']=original['id']+'.visit'+str(visit)
        scene['seed']=int(_hash(SCHEDULE_VERSION,seed,original['id'],visit)[:8],16)%2147483647
        scene['sampling']={'schedule_version':SCHEDULE_VERSION,'initialization_id':original['id'],'batch_index':batch_index,'global_rollout_index':global_index,'environment_visit':visit,'pair_index':visit%len(groups[eid]),'schedule_seed':seed,'batch_size':batch_size}
        result.append(scene)
    return result

def dataset_identity(path=None):
    path=Path(path) if path is not None else DEFAULT_INPUTS
    manifest=json.loads(path.with_name('manifest.json').read_text())
    return {'input_sha256':manifest['inputs_200_sha256'],'schedule_version':SCHEDULE_VERSION,'train_scenarios':manifest['train_scenarios'],'train_ordered_configurations':manifest['train_ordered_configurations'],'test_inputs_sha256':manifest['test_inputs_sha256']}
