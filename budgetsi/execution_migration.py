"""Explicit, audited execution-only ledger rebinding at a stopped batch boundary.

No remote calls, checkpoint mutation, scientific configuration edit or implicit
restart. Planning is read-only. Applying requires an exact authorization packet
and the training run's exclusive lock. Existing committed history is retained.
"""
from __future__ import annotations
import argparse,ast,copy,fcntl,hashlib,json,math,subprocess,time
from pathlib import Path
from budgetsi.run_state import Ledger,digest,durable_json

SCHEMA='school_student_replica_execution_migration_v1'
IMMUTABLE=(
 'on_policy_distillation.sh','budgetsi/top16.py','budgetsi/variant_bridge.py',
 'budgetsi/variant_spec.py','budgetsi/model_runtime.py','budgetsi/gpu_smoke.py',
 'budgetsi/run_state.py','budgetsi/social_protocol/runner.py','budgetsi/social_protocol/token_contract.py',
 'budgetsi/batch_validation.py','budgetsi/social_collect.py','budgetsi/school_data.py',
 'budgetsi/diagnostics.py','budgetsi/diagnostic_run.py',
)
ALLOWED_EXECUTION=(
 'budgetsi/formal_run.py','budgetsi/formal_gate.py','budgetsi/parallel_collect.py',
 'budgetsi/student_replica.py','budgetsi/execution_migration.py',
 'budgetsi/http_runtime.py','budgetsi/remote_teacher.py','budgetsi/social_loop.py','budgetsi/social_protocol/runner.py',
 'budgetsi/gpu_lease.py','budgetsi/score_replica.py',
 'budgetsi/identity_division.py',
)
SCIENTIFIC_CALLS={
 'AdamW','load_model','LoraConfig','get_peft_model','gradient_checkpointing_enable',
 'school_actor_config','actor_config','score_with_actor','update_actor','from_collector_record',
 'select','seed_for','diagnose','schedule_batch','choose_quota','eos_contract',
}

def object_hash(obj):
 return hashlib.sha256(json.dumps(obj,sort_keys=True).encode()).hexdigest()

def git(repo,*args):
 return subprocess.check_output(['git','-C',str(repo),*args],text=True)

def file_at(repo,commit,path):return git(repo,'show',commit+':'+path)

def named_node(source,name):
 tree=ast.parse(source)
 return ast.dump(next(n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name),include_attributes=False)

def scientific_calls(source):
 tree=ast.parse(source);main=next(n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name=='main')
 calls=[]
 for node in ast.walk(main):
  if isinstance(node,ast.Call):
   name=node.func.id if isinstance(node.func,ast.Name) else node.func.attr if isinstance(node.func,ast.Attribute) else None
   if name in SCIENTIFIC_CALLS:calls.append(ast.dump(node,include_attributes=False))
 return sorted(calls)

PROFILE_HELPER = 'def profile_mark(out, stage, batch):\n    """Synchronized stage boundary; numerical inputs and outputs are untouched."""\n    if torch.cuda.is_initialized():\n        torch.cuda.synchronize()\n    with (out / \'phase_profile.jsonl\').open(\'a\') as handle:\n        handle.write(json.dumps(dict(stage=stage, batch=batch, monotonic=time.monotonic(), epoch=time.time()))+\'\\n\')'

def strip_profile(source):
 class Normalize(ast.NodeTransformer):
  def visit_FunctionDef(self,node):
   if node.name=='profile_mark':
    if ast.dump(node,include_attributes=False)!=ast.dump(ast.parse(PROFILE_HELPER).body[0],include_attributes=False):raise ValueError('Unreviewed profiling helper')
    return None
   return self.generic_visit(node)
  def visit_Expr(self,node):
   if ast.dump(node,include_attributes=False)==ast.dump(ast.parse('torch.cuda.empty_cache()').body[0],include_attributes=False):return None
   if isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Name) and node.value.func.id=='profile_mark':
    allowed={ast.dump(ast.parse('profile_mark(out, '+repr(stage)+', batch_index)').body[0],include_attributes=False) for stage in ('exact_scoring_start','exact_scoring_end','diagnostics_end','actor_start','actor_end','checkpoint_end')}
    if ast.dump(node,include_attributes=False) not in allowed:raise ValueError('Unreviewed profiling call')
    return None
   return self.generic_visit(node)
 return ast.unparse(Normalize().visit(ast.parse(source)))

def normalize_profile_metadata(source):
 class Normalize(ast.NodeTransformer):
  def visit_Dict(self,node):
   pairs=[(k,v) for k,v in zip(node.keys,node.values) if not (isinstance(k,ast.Constant) and k.value=='profile_kind' and isinstance(v,ast.Name) and v.id=='kind')]
   node.keys=[k for k,v in pairs];node.values=[v for k,v in pairs]
   return self.generic_visit(node)
 return ast.dump(Normalize().visit(ast.parse(source)),include_attributes=False)


def normalized_formal(source):
 source=strip_profile(source)
 # These are the four exact, reviewed execution-only insertion sites. Everything
 # else in the original driver AST must remain identical (including masks/loops).
 snippets=[
  "from tokenizers import Tokenizer",
  'st._tokenizer = Tokenizer.from_file(str(Path(config["student"]) / "tokenizer.json"))',
  'tt._tokenizer = Tokenizer.from_file(str(Path(config["teacher"]) / "tokenizer.json"))',
  "student_replica = None",
  "student_score_replica = None",
  "if config.get('school_p0'):\n from budgetsi.identity_division import enable_t1_identity_repair\n enable_t1_identity_repair(service.actor)",
  "from contextlib import nullcontext",
  "if config.get('student_replica'):\n from budgetsi.student_replica import StudentReplica\n student_replica = StudentReplica(config, config['student_replica']['cuda_visible_devices'])\n engine_options['student_replica'] = student_replica",
  "if student_replica is not None:\n student_replica.sync_adapter(adapter_state(student), current)",
  "if student_replica is not None:\n student_replica.close()",
  "if config.get('student_score_replica'):\n from budgetsi.score_replica import StudentScoreReplica\n student_score_replica = StudentScoreReplica(config)\n engine_options['student_score_replica'] = student_score_replica",
  "if student_score_replica is not None:\n student_score_replica.sync_adapter(adapter_state(student), current)",
  "if student_score_replica is not None:\n student_score_replica.close()",
 ]
 allowed={ast.dump(ast.parse(text).body[0],include_attributes=False) for text in snippets}
 offload_contexts={ast.dump(ast.parse(text,mode='eval').body,include_attributes=False) for text in (
  'torch.autograd.graph.save_on_cpu(pin_memory=True)',
  'torch.autograd.graph.save_on_cpu(pin_memory=True) if student_replica is not None else nullcontext()')} 
 class StripReviewedAdditions(ast.NodeTransformer):
  def visit_With(self,node):
   if (len(node.items)==1 and node.items[0].optional_vars is None
       and ast.dump(node.items[0].context_expr,include_attributes=False) in offload_contexts
       and len(node.body)==1 and isinstance(node.body[0],ast.Assign)
       and any(isinstance(t,ast.Name) and t.id=='metrics' for t in node.body[0].targets)
       and isinstance(node.body[0].value,ast.Call) and isinstance(node.body[0].value.func,ast.Name)
       and node.body[0].value.func.id=='update_actor'):
    return self.visit(node.body[0])
   return self.generic_visit(node)
  def visit(self,node):
   if isinstance(node,ast.stmt) and ast.dump(node,include_attributes=False) in allowed:return None
   return super().visit(node)
 return ast.dump(StripReviewedAdditions().visit(ast.parse(source)),include_attributes=False)

def normalized_http_transport(source):
 # Only the exact local class import and the parallel server class are removed.
 # Handler validation, dispatch, exception propagation and all other AST remain.
 local_import=ast.dump(ast.parse('from budgetsi.http_runtime import BurstHTTPServer').body[0],include_attributes=False)
 class Normalize(ast.NodeTransformer):
  def visit_ImportFrom(self,node):
   return None if ast.dump(node,include_attributes=False)==local_import else node
  def visit_Assign(self,node):
   if (len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id=='server_class'
       and isinstance(node.value,ast.IfExp) and isinstance(node.value.body,ast.Name)
       and node.value.body.id=='BurstHTTPServer' and isinstance(node.value.orelse,ast.Name)
       and node.value.orelse.id=='HTTPServer'):
    node.value.body.id='ThreadingHTTPServer'
   return self.generic_visit(node)
 return ast.dump(Normalize().visit(ast.parse(source)),include_attributes=False)

def normalized_parallel_transport(source):
 assignment=ast.dump(ast.parse('self.teacher_rpc_slots = threading.BoundedSemaphore(8)').body[0],include_attributes=False)
 guarded=ast.dump(ast.parse("result = self.models[name].call('collect', data)").body[0],include_attributes=False)
 class Normalize(ast.NodeTransformer):
  def visit_Assign(self,node):
   if ast.dump(node,include_attributes=False)==assignment:return None
   return self.generic_visit(node)
  def visit_With(self,node):
   if (len(node.items)==1 and node.items[0].optional_vars is None
       and ast.dump(node.items[0].context_expr,include_attributes=False)==ast.dump(ast.parse('self.teacher_rpc_slots',mode='eval').body,include_attributes=False)
       and len(node.body)==1 and ast.dump(node.body[0],include_attributes=False)==guarded):
    return node.body[0]
   return self.generic_visit(node)
 return ast.dump(Normalize().visit(ast.parse(source)),include_attributes=False)

def normalized_teacher_lease(source):
 class Normalize(ast.NodeTransformer):
  def visit_ImportFrom(self,node):
   if ast.dump(node,include_attributes=False)==ast.dump(ast.parse('from budgetsi.gpu_lease import teacher_scoring_lease').body[0],include_attributes=False):return None
   return self.generic_visit(node)
  def visit_FunctionDef(self,node):
   if node.name=='dispatch':
    node.decorator_list=[d for d in node.decorator_list if not (isinstance(d,ast.Name) and d.id=='teacher_scoring_lease')]
   return self.generic_visit(node)
 return ast.unparse(Normalize().visit(ast.parse(source)))

def verify_code(repo,old,new,review,reason='add_student_replica'):
 for commit in (old,new):
  if len(commit)!=40 or git(repo,'rev-parse',commit+'^{commit}').strip()!=commit:raise ValueError('Exact existing commits required')
 if git(repo,'rev-parse','HEAD').strip()!=new or git(repo,'status','--porcelain','--untracked-files=no').strip():raise ValueError('Migration requires a clean checkout at the reviewed new commit')
 changed=git(repo,'diff','--name-only',old,new).splitlines()
 for path in changed:
  permitted=(path in ALLOWED_EXECUTION or path.startswith('budgetsi/tests/test_') or
             path=='budgetsi/test_parallel_collect.py' or path.startswith('budgetsi/deployments/school_p0/') or
             path.startswith('budgetsi/evidence/school_p0/') or path.endswith('.md'))
  if not permitted:raise ValueError('Unreviewed code change: '+path)
  if path.startswith('budgetsi/deployments/school_p0/'):
   config=json.loads(file_at(repo,new,path))
   verify_config(json.loads(file_at(repo,old,path)),config,reason)
   if reason=='separate_score_queue_profile' and config['source_provenance']['runner.py']!=hashlib.sha256(file_at(repo,new,'budgetsi/social_protocol/runner.py').encode()).hexdigest():
    raise ValueError('Profiling source provenance differs from executed collector')
 for path in ('budgetsi/remote_teacher.py','budgetsi/social_loop.py'):
  before=file_at(repo,old,path) if path in changed else ''
  after=file_at(repo,new,path) if path in changed else ''
  if reason=='add_student_score_replica' and path=='budgetsi/remote_teacher.py':
   before=normalized_teacher_lease(before);after=normalized_teacher_lease(after)
  if path in changed and normalized_http_transport(before)!=normalized_http_transport(after):
   raise ValueError('HTTP transport changed beyond exact parallel server substitution: '+path)
 if reason=='bound_teacher_rpc_connections':
  for path in ('budgetsi/formal_run.py','budgetsi/formal_gate.py','budgetsi/student_replica.py'):
   if path in changed:raise ValueError('Transport repair cannot change execution driver/replica: '+path)
  path='budgetsi/parallel_collect.py'
  if normalized_parallel_transport(file_at(repo,old,path))!=normalized_parallel_transport(file_at(repo,new,path)):
   raise ValueError('Parallel transport changed beyond exact semaphore insertion')
 immutable={}
 for path in IMMUTABLE:
  before=file_at(repo,old,path);after=file_at(repo,new,path)
  if before!=after:
   if not (reason=='separate_score_queue_profile' and path=='budgetsi/social_protocol/runner.py' and normalize_profile_metadata(before)==normalize_profile_metadata(after)):
    raise ValueError('Scientific source changed: '+path)
  immutable[path]=hashlib.sha256(after.encode()).hexdigest()
 if git(repo,'diff','--name-only',old,new,'--','verl').strip():raise ValueError('Upstream source changed')
 for name in ('RowSampler','generate_batch'):
  if named_node(file_at(repo,old,'budgetsi/parallel_collect.py'),name)!=named_node(file_at(repo,new,'budgetsi/parallel_collect.py'),name):raise ValueError('Generation probability/sampling implementation changed: '+name)
 before=file_at(repo,old,'budgetsi/formal_run.py');after=file_at(repo,new,'budgetsi/formal_run.py')
 if scientific_calls(before)!=scientific_calls(after):raise ValueError('Formal runner scientific calls changed')
 if normalized_formal(before)!=normalized_formal(after):raise ValueError('Formal runner changed beyond the exact replica boundary insertions')
 patch=git(repo,'diff','--no-ext-diff',old,new,'--',*ALLOWED_EXECUTION)
 patch_hash=hashlib.sha256(patch.encode()).hexdigest()
 if review!={'old_commit':old,'new_commit':new,'execution_diff_sha256':patch_hash,'scientific_calls_reviewed':True}:
  raise ValueError('Exact independent execution patch review required')
 return {'changed_paths':changed,'immutable_source_sha256':immutable,'execution_diff_sha256':patch_hash,'scientific_call_ast_unchanged':True,'formal_ast_unchanged_after_exact_execution_insertions':True}

def verify_config(old,new,reason='add_student_replica'):
 if reason not in ('add_student_replica','bound_teacher_rpc_connections','separate_score_queue_profile','add_student_score_replica','skip_t1_identity_division','restore_shipped_tokenizer_backend'):
  raise ValueError('Unknown execution migration reason')
 if 'student_replica' not in new:raise ValueError('Explicit student_replica configuration required')
 replica=new['student_replica']
 expected_gpu={'same_context':'6','reference_context':'7'}.get(old.get('teacher_context_mode'))
 if expected_gpu is None or replica!={'backend':'hf_seeded_replica_v1','cuda_visible_devices':expected_gpu}:raise ValueError('Replica must match the exact approved backend/GPU for its P0 arm')
 if reason=='add_student_replica':
  if 'student_replica' in old:raise ValueError('Only first explicit student_replica addition is supported')
  stripped=copy.deepcopy(new);del stripped['student_replica']
  if object_hash(stripped)!=object_hash(old):raise ValueError('Scientific or other existing configuration changed')
 elif reason=='add_student_score_replica':
  expected_score={'backend':'hf_frozen_score_replica_v1','cuda_visible_devices':'2' if old['teacher_context_mode']=='same_context' else '3',
                  'lease_path':'/hpc2ssd/JH_DATA/spooler/lgong265/school_opd_20260920/launch/p0_20260920/scoring_gpu.lease'}
  if 'student_score_replica' in old or new.get('student_score_replica')!=expected_score:raise ValueError('Only exact first scoring-replica addition is supported')
  stripped=copy.deepcopy(new);del stripped['student_score_replica']
  if object_hash(old)!=object_hash(stripped):raise ValueError('Scoring replica changed scientific/existing configuration')
 else:
  compared=copy.deepcopy(new)
  if reason=='separate_score_queue_profile' and old.get('source_provenance'):
   compared['source_provenance']['runner.py']=old['source_provenance']['runner.py']
  if object_hash(old)!=object_hash(compared):raise ValueError('Execution repair changed configuration beyond profiling source provenance')
 return replica

def verify_checkpoint(out,state,config):
 # Reuse all existing sequential-chain, path, file hash and receipt checks.
 ledger=Ledger.__new__(Ledger);ledger.out=out;ledger.target=config['target_nodes'];ledger.state=state;ledger.validate(state)
 if not state['updates']:raise ValueError('No committed update; use a fresh run, not ledger migration')
 from safetensors.torch import load_file
 import torch
 last=state['updates'][-1];folder=out/last['checkpoint']
 adapter=load_file(str(folder/'adapter/adapter_model.safetensors'),device='cpu')
 h=hashlib.sha256()
 for name,value in sorted(adapter.items()):
  if not torch.isfinite(value).all():raise ValueError('Nonfinite checkpoint adapter')
  h.update(name.encode());h.update(value.contiguous().view(torch.uint8).numpy().tobytes())
 if h.hexdigest()!=last['snapshot_after']:raise ValueError('Adapter tensor snapshot mismatch')
 optimizer=torch.load(folder/'optimizer.pt',map_location='cpu',weights_only=True)
 if not optimizer.get('state') or not optimizer.get('param_groups'):raise ValueError('Empty optimizer checkpoint')
 steps=[]
 for slot in optimizer['state'].values():
  if 'step' not in slot:raise ValueError('Optimizer state missing step')
  step=slot['step'].item() if isinstance(slot['step'],torch.Tensor) else slot['step']
  if not isinstance(step,(int,float)) or not math.isfinite(step) or step!=len(state['updates']):raise ValueError('Optimizer step mismatch')
  steps.append(step)
  for value in slot.values():
   if isinstance(value,torch.Tensor) and not torch.isfinite(value).all():raise ValueError('Nonfinite optimizer tensor')
 parameter_ids=[pid for group in optimizer['param_groups'] for pid in group.get('params',[])]
 if not parameter_ids or len(set(parameter_ids))!=len(parameter_ids) or not set(optimizer['state']).issubset(parameter_ids):raise ValueError('Optimizer parameter mapping mismatch')
 for group in optimizer['param_groups']:
  if group.get('lr')!=config['optimizer']['lr'] or group.get('weight_decay')!=config['optimizer']['weight_decay']:
   raise ValueError('Optimizer hyperparameters differ from bound configuration')
 return {'snapshot':h.hexdigest(),'optimizer_step':len(state['updates']),'optimizer_states':len(steps),
         'used_nodes':len(ledger.used),'next_batch':state['next_batch'],'last_checkpoint':last['checkpoint']}

def plan_execution_migration(run_dir,new_config,repo,authorization):
 out=Path(run_dir).resolve();new_config=copy.deepcopy(new_config)
 old=json.loads((out/'config.json').read_text());state=json.loads((out/'state.json').read_text())
 old_binding=state['binding'];new_commit=authorization.get('new_binding',{}).get('git_commit')
 if old_binding!={'config_sha256':object_hash(old),'git_commit':old_binding.get('git_commit')}:raise ValueError('Stored old configuration binding mismatch')
 new_binding={'config_sha256':object_hash(new_config),'git_commit':new_commit}
 reason=authorization.get('reason')
 required={'schema':SCHEMA,'authorized':True,'reason':reason,'run_dir':str(out),'old_binding':old_binding,'new_binding':new_binding}
 if authorization.get('authorized') is not True or any(authorization.get(k)!=v for k,v in required.items()):raise ValueError('Explicit exact execution migration authorization required')
 replica=verify_config(old,new_config,reason)
 code=verify_code(repo,old_binding['git_commit'],new_commit,authorization.get('code_review'),reason)
 checkpoint=verify_checkpoint(out,state,old)
 new_state=copy.deepcopy(state);new_state['binding']=new_binding
 # Batch indices, used nodes and old receipts never get rewritten.
 identifier=object_hash(required)[:20]
 return {'schema':SCHEMA,'migration_id':identifier,'run_dir':str(out),'repo':str(Path(repo).resolve()),'authorization':authorization,
         'old_config':old,'new_config':new_config,'old_state':state,'new_state':new_state,
         'old_state_file_sha256':digest(out/'state.json'),'old_config_file_sha256':digest(out/'config.json'),
         'old_state_object_sha256':object_hash(state),'new_state_object_sha256':object_hash(new_state),
         'replica':replica,'code_verification':code,'checkpoint_verification':checkpoint}

def apply_execution_migration(plan):
 out=Path(plan['run_dir']).resolve()
 reason=plan['authorization'].get('reason')
 verify_config(plan['old_config'],plan['new_config'],reason)
 expected=copy.deepcopy(plan['old_state']);expected['binding']={'config_sha256':object_hash(plan['new_config']),'git_commit':plan['authorization']['new_binding']['git_commit']}
 if expected!=plan['new_state'] or object_hash(expected)!=plan['new_state_object_sha256']:raise ValueError('Plan attempts to change committed scientific history')
 required={'schema':SCHEMA,'authorized':True,'reason':reason,'run_dir':str(out),'old_binding':plan['old_state']['binding'],'new_binding':expected['binding']}
 if plan['authorization'].get('authorized') is not True or any(plan['authorization'].get(k)!=v for k,v in required.items()) or object_hash(required)[:20]!=plan['migration_id']:raise ValueError('Plan authorization mismatch')
 verify_code(plan['repo'],required['old_binding']['git_commit'],required['new_binding']['git_commit'],plan['authorization'].get('code_review'),reason)
 with (out/'run.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  archive=out/'execution_migrations'/plan['migration_id']
  if not archive.resolve().is_relative_to(out):raise ValueError('Migration journal outside run')
  current=json.loads((out/'state.json').read_text())
  if object_hash(current)==plan['new_state_object_sha256'] and (archive/'plan.json').exists():
   stored=json.loads((archive/'plan.json').read_text())
   if stored!=plan:raise ValueError('Existing migration journal differs')
   verify_checkpoint(out,current,plan['old_config'])
   # Recover only an already-published exact same migration, never resample/rewrite history.
   receipt={'status':'committed','migration_id':plan['migration_id'],'old_binding':plan['old_state']['binding'],'new_binding':plan['new_state']['binding'],'checkpoint_verification':plan['checkpoint_verification']}
   durable_json(archive/'receipt.json',receipt);return receipt
  if digest(out/'state.json')!=plan['old_state_file_sha256'] or digest(out/'config.json')!=plan['old_config_file_sha256']:
   raise ValueError('Run changed since migration planning')
  verify_checkpoint(out,current,plan['old_config'])
  if archive.exists():
   if json.loads((archive/'plan.json').read_text())!=plan:raise ValueError('Existing migration journal differs')
  else:
   archive.mkdir(parents=True,exist_ok=False)
   durable_json(archive/'old_state.json',plan['old_state']);durable_json(archive/'old_config.json',plan['old_config'])
   durable_json(archive/'new_config.json',plan['new_config']);durable_json(archive/'authorization.json',plan['authorization'])
   durable_json(archive/'plan.json',plan)
  durable_json(out/'state.json',plan['new_state'])
  receipt={'status':'committed','migration_id':plan['migration_id'],'old_binding':plan['old_state']['binding'],'new_binding':plan['new_state']['binding'],'checkpoint_verification':plan['checkpoint_verification']}
  durable_json(archive/'receipt.json',receipt)
  # formal_run --resume writes the new config only after its own launch gate passes.
  return receipt

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--run-dir',required=True);parser.add_argument('--new-config',required=True);parser.add_argument('--repo',required=True);parser.add_argument('--authorization',required=True);parser.add_argument('--plan-output',required=True);parser.add_argument('--apply',action='store_true');args=parser.parse_args()
 plan=plan_execution_migration(args.run_dir,json.loads(Path(args.new_config).read_text()),args.repo,json.loads(Path(args.authorization).read_text()))
 durable_json(Path(args.plan_output),plan)
 print(json.dumps(apply_execution_migration(plan) if args.apply else {'status':'planned_only','migration_id':plan['migration_id'],'checkpoint_verification':plan['checkpoint_verification']}))
if __name__=='__main__':main()
