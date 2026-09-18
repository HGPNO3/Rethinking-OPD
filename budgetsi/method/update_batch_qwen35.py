"""One transactional online batch update; retain adapter and optimizer between batches."""
import argparse,json,time,hashlib,gc
from pathlib import Path
from runner import validate_reference_record, PROMPT_VERSION, prompt_binding

def sha(ts):
 h=hashlib.sha256()
 for n,t in ts:
  h.update(n.encode());h.update(t.detach().cpu().contiguous().reshape(-1).view(__import__('torch').uint8).numpy().tobytes())
 return h.hexdigest()

def main():
 p=argparse.ArgumentParser();p.add_argument('--records',required=True);p.add_argument('--output',required=True);p.add_argument('--previous');p.add_argument('--limit',type=int,default=1000);p.add_argument('--snapshot-id',required=True);p.add_argument('--model',default='/home/ecs-user/models/Qwen3.5-4B');a=p.parse_args()
 rows=json.loads(Path(a.records).read_text());rows=rows.get('records',[]) if isinstance(rows,dict) else rows
 for record in rows:validate_reference_record(record)
 import torch
 from transformers import AutoModelForImageTextToText,AutoTokenizer
 from token_contract import build_contract,validate_action_ids
 from peft import LoraConfig,get_peft_model,PeftModel
 start=time.perf_counter();out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
 assert not out.exists(), 'Never overwrite an update receipt; resume via controller'
 rows=sorted(rows,key=lambda r:r['id'])[:a.limit]
 assert len({r['id'] for r in rows})==len(rows)
 assert all(r['snapshot_id']==a.snapshot_id for r in rows), 'stale records'
 report={'prompt_version':PROMPT_VERSION,'prompt_binding':prompt_binding(),'engineering_only':False,'snapshot_id':a.snapshot_id,'used_node_ids':[r['id'] for r in rows],'selected_records':len(rows),'records_sha256':hashlib.sha256(Path(a.records).read_bytes()).hexdigest(),'status':'starting','timings':{}}
 def save():
  tmp=out.with_suffix('.tmp');tmp.write_text(json.dumps(report,indent=2));tmp.replace(out)
 save()
 if not rows:report['status']='no_selected_records';save();return
 try:
  torch.manual_seed(20260915)
  t=time.perf_counter();base=AutoModelForImageTextToText.from_pretrained(a.model,local_files_only=True,trust_remote_code=False,dtype=torch.bfloat16,device_map={'':0},attn_implementation='sdpa')
  tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True);vocab=set(tok.get_vocab().values())
  config=base.config.to_dict();gpath=Path(a.model)/'generation_config.json'
  contract=build_contract(tok,config,json.loads(gpath.read_text()) if gpath.exists() else None)
  targets=[n for n,m in base.named_modules() if n.startswith('model.language_model.') and isinstance(m,torch.nn.Linear)]
  assert targets, 'No verified text-backbone linear modules'
  report['lora_target_modules']=targets;report['token_contract']=contract
  model=PeftModel.from_pretrained(base,Path(a.previous)/'adapter',is_trainable=True) if a.previous else get_peft_model(base,LoraConfig(r=32,lora_alpha=64,lora_dropout=0,bias='none',target_modules=targets,task_type='CAUSAL_LM'));model.eval();model.config.use_cache=False;model.config.text_config.use_cache=False;model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.enable_input_require_grads()
  report['timings']['load']=time.perf_counter()-t
  def scores(r):
   prefix=r['student_prefix'];target=r['target_ids'];x=torch.tensor([prefix+target],device='cuda:0')
   positions=torch.arange(len(prefix)-1,len(prefix)+len(target)-1,device='cuda:0')
   logits=model(input_ids=x,attention_mask=torch.ones_like(x),use_cache=False,logits_to_keep=positions).logits[0].float()
   ids=torch.tensor(target,device='cuda:0');return logits.gather(-1,ids[:,None]).squeeze(-1)-logits.logsumexp(-1)
  t=time.perf_counter();cache=[];cross=[]
  for r in rows:
   validate_action_ids(r['target_ids'],tok,contract)
   assert all(type(x)is int and x in vocab for x in r['target_ids'])
   assert len(r['target_ids'])==len(r['behavior_logprobs'])==len(r['teacher_logprobs'])
   with torch.no_grad():old=scores(r).detach()
   b=torch.tensor(r['behavior_logprobs'],device='cuda:0');q=torch.tensor(r['teacher_logprobs'],device='cuda:0');assert torch.isfinite(old).all() and torch.isfinite(b).all() and torch.isfinite(q).all()
   cache.append((r,old,b,q))
   if r.get('old_raw_logprobs'):
    cross.extend((old.cpu()-torch.tensor(r['old_raw_logprobs'])).abs().tolist())
  report['timings']['cache_old_raw_hf']=time.perf_counter()-t
  params=[(n,v) for n,v in model.named_parameters() if v.requires_grad];before=sha(params);optimizer=torch.optim.AdamW([v for n,v in params],lr=1e-5,betas=(.9,.999),eps=1e-8,weight_decay=0)
  previous_step=0
  if a.previous:
   prior=json.loads((Path(a.previous)/'result.json').read_text());assert prior['status']=='passed'
   if prior.get('prompt_version')!=PROMPT_VERSION or prior.get('prompt_binding')!=prompt_binding():raise ValueError('Prior checkpoint uses another OPD protocol')
   assert before==prior['adapter_after'], 'adapter continuation mismatch'
   optimizer.load_state_dict(torch.load(Path(a.previous)/'optimizer.pt',weights_only=True,map_location='cpu'))
   previous_step=max(int(state['step']) for state in optimizer.state.values())
   assert previous_step==prior['optimizer_step']
  report['optimizer_step_before']=previous_step
  total_tokens=sum(len(r['target_ids']) for r in rows);optimizer.zero_grad(set_to_none=True);loss_total=0.;t=time.perf_counter();ratio_min=float('inf');ratio_max=0.
  model.train()  # Qwen3 attention dropout and LoRA dropout are both zero.
  assert model.config.text_config.attention_dropout==0
  for r,old,b,q in cache:
   current=scores(r);ratio=(current-b.detach()).exp();advantage=(q-old).detach()
   assert torch.isfinite(ratio).all();ratio_min=min(ratio_min,float(ratio.detach().min()));ratio_max=max(ratio_max,float(ratio.detach().max()))
   loss=-(ratio*advantage).sum()/total_tokens;assert torch.isfinite(loss);loss.backward();loss_total+=float(loss.detach());del current,ratio,loss
  norm=torch.nn.utils.clip_grad_norm_([v for n,v in params],1,error_if_nonfinite=True);optimizer.step();after=sha(params);assert before!=after
  report['optimizer_step']=max(int(state['step']) for state in optimizer.state.values());assert report['optimizer_step']==previous_step+1
  report['timings']['gradient_accumulation_one_update']=time.perf_counter()-t
  report.update(supervised_tokens=total_tokens,loss=loss_total,gradient_norm=float(norm),adapter_before=before,adapter_after=after,behavior_ratio_min=ratio_min,behavior_ratio_max=ratio_max,loss_definition='raw-policy per-prefix reverse-KL first-step surrogate: ratio exp(current_raw-behavior_tempered), detached teacher_raw-old_hf_raw; selected-sample diagnostic, not exact full trajectory KL')
  if cross:report['cross_engine_raw_difference']={'mean_abs':sum(cross)/len(cross),'max_abs':max(cross),'tokens':len(cross)}
  assert report['cross_engine_raw_difference']['mean_abs']<0.1, 'Serving/HF raw scores disagree materially'
  t=time.perf_counter();adapter=out.parent/'adapter';model.save_pretrained(adapter,safe_serialization=True);torch.save(optimizer.state_dict(),out.parent/'optimizer.pt');opt_before=sha((f'{k}/{n}',v) for k,s in optimizer.state_dict()['state'].items() for n,v in s.items() if torch.is_tensor(v))
  model.eval()
  with torch.no_grad():reference=scores(rows[0]).cpu()
  del optimizer,params,cache;base=model.unload();del model;gc.collect();torch.cuda.empty_cache();model=PeftModel.from_pretrained(base,adapter,is_trainable=True);model.eval()
  opt=torch.optim.AdamW([v for v in model.parameters() if v.requires_grad],lr=1e-5,betas=(.9,.999),eps=1e-8,weight_decay=0);opt.load_state_dict(torch.load(out.parent/'optimizer.pt',weights_only=True,map_location='cpu'))
  assert sha((n,v) for n,v in model.named_parameters() if v.requires_grad)==after
  assert sha((f'{k}/{n}',v) for k,s in opt.state_dict()['state'].items() for n,v in s.items() if torch.is_tensor(v))==opt_before
  with torch.no_grad():err=float((scores(rows[0]).cpu()-reference).abs().max())
  assert err<1e-5;report['restore_max_error']=err;report['timings']['save_reload']=time.perf_counter()-t;report['checkpoint_file_sha256']=hashlib.sha256((adapter/'adapter_model.safetensors').read_bytes()).hexdigest();report['optimizer_file_sha256']=hashlib.sha256((out.parent/'optimizer.pt').read_bytes()).hexdigest();report['status']='passed'
 except Exception as e:
  import traceback
  report['status']='failed';report['error']=str(e);report['traceback']=traceback.format_exc()
 finally:
  report['timings']['total']=time.perf_counter()-start;save();print(json.dumps({'status':report['status'],'output':str(out),'error':report.get('error')}),flush=True)
 if report['status']=='failed':raise SystemExit(1)
if __name__=='__main__':main()
