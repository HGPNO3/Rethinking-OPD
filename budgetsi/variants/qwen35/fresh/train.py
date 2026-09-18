"""Resumable gated two-teacher training. Receipts, never counters, own progress."""
import argparse,copy,fcntl,hashlib,json,os,random,signal,subprocess,time,urllib.request,shutil
from pathlib import Path
from protocol_gate import verify_approval,load_json
from runner import PROMPT_VERSION,prompt_binding,validate_reference_record
from batch_validation import validate_batch
ROOT=Path(__file__).resolve().parent
COL='/home/ecs-user/budgetsi-collector-venv/bin/python'
HF='/home/ecs-user/budgetsi-venv/bin/python'
VLLM='/home/ecs-user/budgetsi-vllm-speed-venv/bin/vllm'
BASE='/home/ecs-user/models/'

def digest(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for x in iter(lambda:f.read(8*1024*1024),b''):h.update(x)
 return h.hexdigest()
def atomic(p,obj):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2));tmp.replace(p)
def receipts(out):
 used=[];prior=None;count=0;updates=[]
 for p in sorted(Path(out).glob('batch_*/update/result.json')):
  r=load_json(p)
  if r['status']!='passed':continue
  if r.get('prompt_version')!=PROMPT_VERSION or r.get('prompt_binding')!=prompt_binding():raise ValueError('Cannot resume a checkpoint from another OPD protocol')
  assert r['optimizer_step']==count+1
  if prior:assert r['adapter_before']==load_json(prior/'result.json')['adapter_after']
  assert digest(p.parent/'adapter'/'adapter_model.safetensors')==r['checkpoint_file_sha256']
  assert digest(p.parent/'optimizer.pt')==r['optimizer_file_sha256']
  ids=r['used_node_ids'];assert len(ids)==r['selected_records'] and not set(ids).intersection(used)
  used.extend(ids);prior=p.parent;count+=1;updates.append(str(p))
 return used,prior,updates

def batch_scenes(pool,mode,index,n,seed):
 # First 80 hash-ordered scenes train; last 20 reserved, not inspected for selection.
 order=sorted(pool,key=lambda s:hashlib.sha256(('split20260915'+s['id']).encode()).hexdigest())[:80]
 rng=random.Random(seed);rng.shuffle(order);ans=[]
 for j in range(n):
  absolute=index*n+j;s=copy.deepcopy(order[absolute%len(order)])
  s['id']=f'{mode}:b{index:04d}:j{j:02d}:'+s['id'];s['seed']=seed+absolute;ans.append(s)
 return {'schema_version':'fresh_online_batch_v1','scenes':ans}

class Controller:
 def __init__(self,args,cfg):
  self.a=args;self.cfg=cfg;
  global COL,HF,VLLM,BASE
  runtime=cfg['runtime'];COL=runtime['collector_python'];HF=runtime['train_python'];VLLM=runtime['vllm'];BASE=runtime['model_root'];
  self.out=Path(args.output);self.children=[];self.phase='starting';self.phase_at=time.time();self.started=time.time();self.detail={};self.out.mkdir(parents=True,exist_ok=True)
 def beat(self,phase=None,**detail):
  if phase is not None:self.phase=phase;self.phase_at=time.time()
  self.detail.update(detail)
  atomic(self.out/'state.json',dict(status=self.phase,pid=os.getpid(),heartbeat=time.time(),phase_started=self.phase_at,started=self.started,**self.detail))
 def spawn(self,cmd,log,env=None):
  p=subprocess.Popen(cmd,cwd=ROOT,env={**os.environ,**(env or {})},stdout=open(log,'a'),stderr=subprocess.STDOUT,start_new_session=True);self.children.append(p);self.beat(child_pids=[c.pid for c in self.children]);return p
 def clean(self):
  for p in self.children:
   if p.poll() is None:
    try:os.killpg(p.pid,signal.SIGTERM)
    except ProcessLookupError:pass
  deadline=time.time()+30
  for p in self.children:
   try:p.wait(timeout=max(1,deadline-time.time()))
   except subprocess.TimeoutExpired:
    try:os.killpg(p.pid,signal.SIGKILL)
    except ProcessLookupError:pass
    p.wait()
  self.children=[];self.beat(child_pids=[])
 def command(self,cmd,log,timeout,env=None):
  p=self.spawn(cmd,log,env);end=time.time()+timeout
  while p.poll() is None:
   if time.time()>end:raise TimeoutError('phase timeout: '+self.phase)
   self.beat();time.sleep(5)
  self.children.remove(p);self.beat(child_pids=[c.pid for c in self.children])
  if p.returncode:raise RuntimeError(f'child exited {p.returncode}; see {log}')
 def servers(self,d,previous):
  self.beat('starting_student',checkpoint=str(previous) if previous else 'initial')
  endpoint=self.cfg['runtime']['teacher_endpoint']
  try: urllib.request.urlopen('http://127.0.0.1:18004/health',timeout=2)
  except Exception: pass
  else: raise RuntimeError('Student port occupied; refusing duplicate launch')
  urllib.request.urlopen(endpoint+'/health',timeout=10)
  name='base_student' if previous else 'student'
  cmd=[VLLM,'serve',BASE+self.cfg['runtime']['student_folder'],'--served-model-name',name,'--host','127.0.0.1','--port','18004','--dtype','bfloat16','--max-model-len',str(self.cfg['context']),'--max-num-seqs','4','--max-num-batched-tokens',str(self.cfg['context']),'--gpu-memory-utilization','0.80','--generation-config','vllm','--logprobs-mode','processed_logprobs','--enforce-eager','--no-enable-prefix-caching','--no-enable-chunked-prefill','--no-async-scheduling','--seed',str(self.cfg['seed'])]
  if previous:cmd+=['--enable-lora','--max-lora-rank','32','--lora-modules','student='+str(previous/'adapter')]
  env={'CUDA_VISIBLE_DEVICES':'0','VLLM_USE_V2_MODEL_RUNNER':'0'}
  atomic(d/'student_launch.json',{'command':cmd,'environment_overrides':env,'teacher_endpoint':endpoint});self.spawn(cmd,d/'student_server.log',env)
  end=time.time()+1200
  while True:
   if any(p.poll() is not None for p in self.children):raise RuntimeError('Student server exited during startup')
   try:urllib.request.urlopen('http://127.0.0.1:18004/health',timeout=2);break
   except Exception:
    if time.time()>end:raise TimeoutError('Student model startup')
    self.beat();time.sleep(3)
  models=json.load(urllib.request.urlopen('http://127.0.0.1:18004/v1/models'))['data']
  assert 'student' in [m['id'] for m in models];atomic(d/'served_models.json',{'student':models})
 def run_mode(self,mode,batches=None):
  out=self.out/mode;out.mkdir(exist_ok=True);pool=load_json(ROOT/'inputs.json')['scenes'];limit=1000 if mode=='formal' else 10000
  n=self.cfg['batch_dialogues'] if mode=='formal' else self.cfg['acceptance_dialogues']
  seed=self.cfg['seed']+(100000 if mode=='formal' else 0)
  index=0
  # Find the continuation point cheaply, then verify every committed hash once.
  # This preserves fail-closed receipt validation without an O(batches^2) resume scan.
  while (out/f'batch_{index:04d}'/'update/result.json').exists() and load_json(out/f'batch_{index:04d}'/'update/result.json')['status']=='passed':index+=1
  while True:
   used,previous,updates=receipts(out)
   if len(used)>=limit or (batches is not None and index>=batches):break
   if index>=self.cfg['max_batches']:raise RuntimeError('Max batch safety stop before quota')
   d=out/f'batch_{index:04d}';d.mkdir(exist_ok=True)
   if (d/'update/result.json').exists() and load_json(d/'update/result.json')['status']=='passed':index+=1;continue
   if shutil.disk_usage(out).free<15*1024**3:raise RuntimeError('Disk below 15 GB reserve')
   snapshot=digest(previous/'adapter/adapter_model.safetensors') if previous else self.cfg['initial_snapshot_id']
   self.beat('preparing_batch',mode=mode,batch=index,used_nodes=len(used),updates=len(updates),snapshot_id=snapshot)
   # Save once: a restart must not change the scene or seed.
   inputs=batch_scenes(pool,mode,index,n,seed)
   if (d/'inputs.json').exists():assert load_json(d/'inputs.json')==inputs
   else:atomic(d/'inputs.json',inputs)
   summary_path=d/'rollout/summary.json'
   summary=load_json(summary_path) if summary_path.exists() else {}
   if summary.get('status') not in {'completed','partial'}:
    if (d/'rollout').exists():
     # Preserve failed attempt evidence. Same seed/snapshot only; no candidate-quality retries.
     (d/'rollout').rename(d/('failed_rollout_'+str(time.time_ns())))
    atomic(d/('runtime_'+str(time.time_ns())+'.json'),{'git_commit':load_json(self.out/'gate.json')['verified_bindings']['git_commit'],'config_sha256':digest(ROOT/'config.json'),'snapshot_id':snapshot,'reason':'new_or_infrastructure_retry_same_scene_seed'})
    self.servers(d,previous)
    self.beat('api_acceptance');self.command([COL,'smoke_api.py','--output',str(d/'api_smoke.json')],d/'smoke.log',300)
    self.beat('collecting')
    self.command([COL,'runner.py','--inputs',str(d/'inputs.json'),'--output',str(d/'rollout'),'--student-tokenizer',BASE+self.cfg['runtime']['student_folder'],'--teacher-tokenizer',BASE+self.cfg['runtime']['teacher_folder'],'--teacher-endpoint',self.cfg['runtime']['teacher_endpoint'],'--max-context',str(self.cfg['context']),'--concurrency',str(self.cfg['concurrency']),'--seed',str(seed),'--snapshot-id',snapshot,'--goal-instruction',self.cfg['goal_instruction']],d/'runner.log',7200)
    self.clean();summary=load_json(summary_path)
   if summary.get('prompt_version')!=PROMPT_VERSION or summary.get('prompt_binding')!=prompt_binding():raise RuntimeError('Old rollout cache: collect a fresh reference-OPD batch')
   rows=json.loads((d/'rollout/selected_records.json').read_text())
   validation=validate_batch(summary,json.loads((d/'rollout/dialogues.json').read_text()),[x['id'] for x in inputs['scenes']],rows)
   atomic(d/'batch_validation.json',validation)
   for record in rows:validate_reference_record(record)
   if not rows:
    atomic(d/'empty.json',summary);index+=1
    if index>=3 and not used:raise RuntimeError('No selected nodes across three batches')
    continue
   if (d/'update').exists():(d/'update').rename(d/('failed_update_'+str(time.time_ns())))
   self.beat('updating',selected_nodes=summary['selected_nodes'],A_actions=summary['A_actions'])
   cmd=[HF,self.cfg['runtime']['update_script'],'--model',BASE+self.cfg['runtime']['student_folder'],'--records',str(d/'rollout/selected_records.json'),'--output',str(d/'update/result.json'),'--snapshot-id',snapshot,'--limit',str(limit-len(used))]
   if previous:cmd+=['--previous',str(previous)]
   self.command(cmd,d/'update.log',3600,{'CUDA_VISIBLE_DEVICES':'0'})
   result=load_json(d/'update/result.json');assert result['status']=='passed'
   # Empirical initial-run tolerance; gross wrong-model/LoRA routing fails, numeric BF16 differences do not.
   assert result['cross_engine_raw_difference']['mean_abs']<0.1,'Serving/HF raw scores disagree materially'
   index+=1
  used,previous,updates=receipts(out)
  result={'status':'passed','used_nodes':len(used),'updates':len(updates),'checkpoint':str(previous),'update_receipts':updates}
  atomic(out/'result.json',result);return result
 def run(self):
  try:
   if (self.out/'gate.json').exists():
    history=self.out/'restarts'/str(time.time_ns());history.mkdir(parents=True)
    for name in ['gate.json','approval.json','config.json','manifest.json','state.json','monitor.json','error.json']:
     if (self.out/name).exists():shutil.copy2(self.out/name,history/name)
    if (self.out/'error.json').exists():(self.out/'error.json').unlink()
   self.beat('gate')
   approval=load_json(Path(self.a.approval));report=verify_approval(approval,ROOT/'manifest.json',ROOT/'config.json',ROOT);atomic(self.out/'gate.json',report)
   assert report['ok'],report['errors']
   for name in ['config.json','manifest.json']:shutil.copy2(ROOT/name,self.out/name)
   shutil.copy2(self.a.approval,self.out/'approval.json')
   self.beat('verifying_artifacts')
   for p,h in self.cfg['files_sha256'].items():assert digest(p)==h,p;self.beat()
   assert digest(ROOT/'inputs.json')==load_json(ROOT/'manifest.json')['inputs_sha256']
   self.command([COL,'-m','unittest','test_runner','test_train','test_reference_opd','test_batch_validation'],self.out/'tests.log',120)
   acceptance=self.run_mode('acceptance',batches=2)
   assert acceptance['updates']==2, 'Acceptance must exercise optimizer and served checkpoint continuation'
   self.beat('acceptance_passed')
   result=self.run_mode('formal')
   self.beat('completed',used_nodes=result['used_nodes'],updates=result['updates'],checkpoint=result['checkpoint'],elapsed_seconds=time.time()-self.started)
  except BaseException as e:
   import traceback
   atomic(self.out/'error.json',{'time':time.time(),'phase':self.phase,'error':repr(e),'traceback':traceback.format_exc()})
   self.beat('failed',error=repr(e));raise
  finally:self.clean()

def main():
 p=argparse.ArgumentParser();p.add_argument('--approval',required=True);p.add_argument('--output',required=True);a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
 with open(out/'controller.lock','w') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  c=Controller(a,load_json(ROOT/'config.json'))
  def term(*_):raise KeyboardInterrupt('signal stop')
  signal.signal(signal.SIGTERM,term);c.run()
if __name__=='__main__':main()
