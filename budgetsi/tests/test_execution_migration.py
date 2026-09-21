import copy,fcntl,hashlib,json,subprocess,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import torch
from safetensors.torch import save_file
from budgetsi import execution_migration as m
from budgetsi.run_state import digest,durable_json

class MigrationTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
  self.root=Path(self.tmp.name).resolve();self.repo=self.root/'repo';self.repo.mkdir();self.run=self.root/'run';self.run.mkdir()
  self.command('init','-q');self.command('config','user.email','test@example.invalid');self.command('config','user.name','Test')
  for name in m.IMMUTABLE:
   p=self.repo/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('pass\n')
  (self.repo/'budgetsi/formal_run.py').write_text('def main():\n update_actor(service, actions, scores, actor_config=cfg)\n')
  (self.repo/'budgetsi/parallel_collect.py').write_text('class RowSampler:\n pass\ndef generate_batch(model,tokenizer,requests,max_context):\n return requests\n')
  self.old=self.commit()
  (self.repo/'budgetsi/student_replica.py').write_text('class StudentReplica:\n pass\n')
  with (self.repo/'budgetsi/formal_run.py').open('a') as f:f.write(' if student_replica is not None:\n  student_replica.sync_adapter(adapter_state(student),current)\n')
  self.new=self.commit()
  self.config={'target_nodes':3,'teacher_context_mode':'same_context','optimizer':{'lr':1e-5,'weight_decay':.01}}
  self.newconfig={**self.config,'student_replica':{'backend':'hf_seeded_replica_v1','cuda_visible_devices':'6'}}
  checkpoint=self.run/'batch_0000/update';(checkpoint/'adapter').mkdir(parents=True)
  tensor=torch.tensor([1.,2.]);save_file({'lora.weight':tensor},str(checkpoint/'adapter/adapter_model.safetensors'))
  snapshot=hashlib.sha256(b'lora.weight'+tensor.view(torch.uint8).numpy().tobytes()).hexdigest()
  self.optimizer={'state':{0:{'step':torch.tensor(1.),'exp_avg':torch.zeros(2)}},'param_groups':[{'lr':1e-5,'weight_decay':.01,'params':[0]}]}
  torch.save(self.optimizer,checkpoint/'optimizer.pt')
  self.record={'status':'passed','optimizer_step':1,'batch':0,'snapshot_before':'initial','snapshot_after':snapshot,'nodes':['n0'],'checkpoint':'batch_0000/update','adapter_file_sha256':digest(checkpoint/'adapter/adapter_model.safetensors'),'optimizer_file_sha256':digest(checkpoint/'optimizer.pt')}
  durable_json(checkpoint/'result.json',self.record)
  self.state={'binding':{'config_sha256':m.object_hash(self.config),'git_commit':self.old},'next_batch':1,'updates':[self.record],'batches':[{'batch':0}]}
  durable_json(self.run/'state.json',self.state);durable_json(self.run/'config.json',self.config)

 def command(self,*args):return subprocess.check_output(['git','-C',str(self.repo),*args],text=True)
 def commit(self):
  self.command('add','.');self.command('commit','-qm','test');return self.command('rev-parse','HEAD').strip()
 def auth(self):
  diff=self.command('diff','--no-ext-diff',self.old,self.new,'--',*m.ALLOWED_EXECUTION)
  return {'schema':m.SCHEMA,'authorized':True,'reason':'add_student_replica','run_dir':str(self.run),'old_binding':self.state['binding'],'new_binding':{'config_sha256':m.object_hash(self.newconfig),'git_commit':self.new},'code_review':{'old_commit':self.old,'new_commit':self.new,'execution_diff_sha256':hashlib.sha256(diff.encode()).hexdigest(),'scientific_calls_reviewed':True}}
 def plan(self):return m.plan_execution_migration(self.run,self.newconfig,self.repo,self.auth())
 def rewrite_optimizer(self,value):
  folder=self.run/self.record['checkpoint'];torch.save(value,folder/'optimizer.pt');self.record['optimizer_file_sha256']=digest(folder/'optimizer.pt');durable_json(folder/'result.json',self.record);durable_json(self.run/'state.json',self.state)

 def test_plan_read_only_and_apply_preserves_history(self):
  before=(self.run/'state.json').read_bytes();plan=self.plan()
  self.assertEqual(before,(self.run/'state.json').read_bytes());self.assertFalse((self.run/'execution_migrations').exists())
  result=m.apply_execution_migration(plan);self.assertEqual(result['status'],'committed')
  state=json.loads((self.run/'state.json').read_text());self.assertEqual(state['binding'],plan['new_state']['binding'])
  for key in ('updates','batches','next_batch'):self.assertEqual(state[key],self.state[key])
  self.assertEqual(json.loads((self.run/'config.json').read_text()),self.config)
  journal=self.run/'execution_migrations'/plan['migration_id'];self.assertEqual(json.loads((journal/'old_state.json').read_text()),self.state)
  self.assertEqual(m.apply_execution_migration(plan),result)

 def test_scientific_config_and_gpu_changes_rejected(self):
  for mutation in ({'optimizer':{'lr':1e-4,'weight_decay':.01}},{'seed':42},{'student_replica':{'backend':'hf_seeded_replica_v1','cuda_visible_devices':'7'}}):
   with self.subTest(mutation=mutation):
    original=self.newconfig;self.newconfig={**original,**mutation}
    with self.assertRaises(ValueError):self.plan()
    self.newconfig=original

 def test_transport_repair_same_config_plan_apply(self):
  self.old=self.new
  self.state['binding']['git_commit']=self.old
  (self.repo/'budgetsi/http_runtime.py').write_text('from http.server import ThreadingHTTPServer\nclass BurstHTTPServer(ThreadingHTTPServer):\n request_queue_size=128\n')
  self.new=self.commit()
  self.config=copy.deepcopy(self.newconfig)
  self.state['binding']['config_sha256']=m.object_hash(self.config)
  durable_json(self.run/'config.json',self.config);durable_json(self.run/'state.json',self.state)
  auth=self.auth();auth['reason']='bound_teacher_rpc_connections'
  plan=m.plan_execution_migration(self.run,self.newconfig,self.repo,auth)
  self.assertEqual(plan['old_config'],plan['new_config'])
  self.assertEqual(m.apply_execution_migration(plan)['status'],'committed')
  self.assertEqual(json.loads((self.run/'state.json').read_text())['updates'],self.state['updates'])

 def test_transport_repair_rejects_any_config_or_reason_change(self):
  for new in (self.config,{**self.newconfig,'seed':123},{**self.newconfig,'optimizer':{'lr':1e-4,'weight_decay':.01}}):
   with self.assertRaises(ValueError):m.verify_config(self.newconfig,new,'bound_teacher_rpc_connections')
  with self.assertRaises(ValueError):m.verify_config(self.newconfig,self.newconfig,'unknown')
  with self.assertRaises(ValueError):m.verify_config(self.config,self.newconfig,'bound_teacher_rpc_connections')

 def test_score_replica_only_exact_gpu_and_lease_config_addition(self):
  score={'backend':'hf_frozen_score_replica_v1','cuda_visible_devices':'2','lease_path':'/hpc2ssd/JH_DATA/spooler/lgong265/school_opd_20260920/launch/p0_20260920/scoring_gpu.lease'}
  new={**self.newconfig,'student_score_replica':score}
  m.verify_config(self.newconfig,new,'add_student_score_replica')
  for bad in ({**new,'seed':5},{**new,'student_score_replica':{**score,'cuda_visible_devices':'4'}},{**new,'student_score_replica':{**score,'lease_path':'/tmp/other'}}):
   with self.assertRaises(ValueError):m.verify_config(self.newconfig,bad,'add_student_score_replica')

 def test_teacher_lease_normalization_cannot_hide_changed_scoring(self):
  before='class TeacherService:\n def dispatch(self,request):\n  return compact_scores(model,prompt,target,ids,k)\n'
  after='from budgetsi.gpu_lease import teacher_scoring_lease\n'+before.replace(' def dispatch',' @teacher_scoring_lease\n def dispatch')
  self.assertEqual(m.normalized_teacher_lease(before),m.normalized_teacher_lease(after))
  self.assertNotEqual(m.normalized_teacher_lease(before),m.normalized_teacher_lease(after.replace('target,ids','target[:2],ids')))

 def test_parallel_transport_allows_only_exact_semaphore(self):
  old="def call(self, name, data):\n result=self.models[name].call('collect', data)\n return result\n"
  new="def call(self, name, data):\n self.teacher_rpc_slots=threading.BoundedSemaphore(8)\n with self.teacher_rpc_slots:\n  result=self.models[name].call('collect', data)\n return result\n"
  self.assertEqual(m.normalized_parallel_transport(old),m.normalized_parallel_transport(new))
  for bad in (new.replace('Semaphore(8)','Semaphore(16)'),new.replace("'collect', data","'collect', changed"),new.replace('return result','return None')):
   self.assertNotEqual(m.normalized_parallel_transport(old),m.normalized_parallel_transport(bad))

 def test_transport_only_ast_preserves_entire_handler(self):
  old="def serve(engine):\n result=engine.call(data)\n server_class=ThreadingHTTPServer if getattr(engine, 'parallel', False) else HTTPServer\n return server_class(addr, Handler)\n"
  new=old.replace(' server_class=', ' from budgetsi.http_runtime import BurstHTTPServer\n server_class=').replace('=ThreadingHTTPServer if','=BurstHTTPServer if')
  self.assertEqual(m.normalized_http_transport(old),m.normalized_http_transport(new))
  for bad in (new.replace('engine.call(data)','engine.call(other)'),new.replace("'parallel', False", "'parallel', True"),new.replace('else HTTPServer','else BurstHTTPServer')):
   self.assertNotEqual(m.normalized_http_transport(old),m.normalized_http_transport(bad))

 def test_exact_authorization_required(self):
  auth=self.auth();auth['authorized']=False
  with self.assertRaisesRegex(ValueError,'authorization'):m.plan_execution_migration(self.run,self.newconfig,self.repo,auth)

 def test_optimizer_semantics_checked_after_file_hashes(self):
  bad=copy.deepcopy(self.optimizer);bad['state'][0]['step']=torch.tensor(2.);self.rewrite_optimizer(bad)
  with self.assertRaisesRegex(ValueError,'Optimizer step'):self.plan()

 def test_snapshot_tensor_hash_checked(self):
  folder=self.run/self.record['checkpoint'];save_file({'lora.weight':torch.tensor([9.,9.])},str(folder/'adapter/adapter_model.safetensors'))
  self.record['adapter_file_sha256']=digest(folder/'adapter/adapter_model.safetensors');durable_json(folder/'result.json',self.record);durable_json(self.run/'state.json',self.state)
  with self.assertRaisesRegex(ValueError,'snapshot'):self.plan()

 def test_scientific_source_and_sampler_changes_rejected(self):
  (self.repo/'budgetsi/top16.py').write_text('changed=True\n');self.new=self.commit()
  with self.assertRaisesRegex(ValueError,'Scientific source|Unreviewed'):self.plan()

 def test_formal_update_call_edit_rejected(self):
  p=self.repo/'budgetsi/formal_run.py';p.write_text(p.read_text().replace('actor_config=cfg','actor_config=other'))
  self.new=self.commit()
  with self.assertRaisesRegex(ValueError,'scientific calls'):self.plan()

 def test_sampler_edit_rejected(self):
  p=self.repo/'budgetsi/parallel_collect.py';p.write_text(p.read_text().replace('return requests','return requests[:1]'))
  self.new=self.commit()
  with self.assertRaisesRegex(ValueError,'sampling'):self.plan()

 def test_only_exact_saved_tensor_offload_wrapper_allowed(self):
  old="def main():\n metrics=update_actor(service, actions, scores)\n"
  allowed="def main():\n with torch.autograd.graph.save_on_cpu(pin_memory=True):\n  metrics=update_actor(service, actions, scores)\n"
  self.assertEqual(m.normalized_formal(old),m.normalized_formal(allowed))
  conditional='from contextlib import nullcontext\n'+allowed.replace('pin_memory=True):','pin_memory=True) if student_replica is not None else nullcontext():')
  self.assertEqual(m.normalized_formal(old),m.normalized_formal(conditional))
  self.assertNotEqual(m.normalized_formal(old),m.normalized_formal(allowed.replace('scores)','scores[:1])')))
  self.assertNotEqual(m.normalized_formal(old),m.normalized_formal(allowed.replace('pin_memory=True','pin_memory=False')))

 def test_identity_division_repair_only_exact_driver_insertion(self):
  old='def main():\n service=LocalActorService(actor, temperature)\n metrics=update_actor(service, actions, scores)\n'
  inserted=" if config.get('school_p0'):\n  from budgetsi.identity_division import enable_t1_identity_repair\n  enable_t1_identity_repair(service.actor)\n"
  new=old.replace(' metrics=',inserted+' metrics=')
  self.assertEqual(m.normalized_formal(old),m.normalized_formal(new))
  self.assertEqual(m.scientific_calls(old),m.scientific_calls(new))
  self.assertNotEqual(m.normalized_formal(old),m.normalized_formal(new.replace('service.actor)','other.actor)')))
  m.verify_config(self.newconfig,self.newconfig,'skip_t1_identity_division')
  with self.assertRaises(ValueError):m.verify_config(self.newconfig,{**self.newconfig,'seed':7},'skip_t1_identity_division')

 def test_active_run_lock_and_stale_plan_rejected(self):
  plan=self.plan()
  with (self.run/'run.lock').open('a') as lock:
   fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
   with self.assertRaises(BlockingIOError):m.apply_execution_migration(plan)
  with (self.run/'state.json').open('a') as f:f.write(' ')
  with self.assertRaisesRegex(ValueError,'changed since'):m.apply_execution_migration(plan)

 def test_plan_tamper_rejected(self):
  plan=self.plan();plan['new_state']['next_batch']=99
  with self.assertRaisesRegex(ValueError,'history'):m.apply_execution_migration(plan)

 def test_crash_after_atomic_state_commit_recovers_exact_plan(self):
  plan=self.plan();real=m.durable_json
  def fail_receipt(path,value):
   if Path(path).name=='receipt.json':raise OSError('simulated crash after authoritative state publication')
   return real(path,value)
  with patch.object(m,'durable_json',side_effect=fail_receipt):
   with self.assertRaises(OSError):m.apply_execution_migration(plan)
  self.assertEqual(m.apply_execution_migration(plan)['status'],'committed')

if __name__=='__main__':unittest.main()
