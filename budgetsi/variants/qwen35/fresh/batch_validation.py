"""Allow only fully accounted dialogue attempts; never repair model output."""
import json
def reviewed_format_failure(f):
 if not f.get('receipt'):return False
 if f.get('reason')=='invalid action schema':return True
 r=f['receipt'];text=r.get('raw_text','')
 # Diagnose only: remove terminal serialization token, never repair/execute action.
 endings=('<|im_end|>','<|endoftext|>')
 eos=next((e for e in endings if text.endswith(e)),None)
 if r.get('finish_reason')!='stop' or not eos:return False
 body=text[:-len(eos)]
 if not body or any(m in body for m in ['<think>','</think>','<|']):return False
 try:json.loads(body)
 except json.JSONDecodeError as e:return str(e)==f.get('reason')
 return False

def validate_batch(summary, dialogues, scene_ids, records):
 if summary.get('fatal_error') or summary.get('status') not in {'completed','partial'}:
  raise ValueError('Fatal or unrecognized rollout status')
 if len(scene_ids)!=len(set(scene_ids)) or sorted(d['id'] for d in dialogues)!=sorted(scene_ids):
  raise ValueError('Missing or duplicate dialogue attempt')
 allowed=set();failures=[]
 for d in dialogues:
  if d.get('fatal_error'):raise ValueError('Fatal dialogue error')
  if not d.get('complete'):
   f=d.get('failure',{})
   if not reviewed_format_failure(f):
    raise ValueError('Unreviewed incomplete dialogue reason')
   failures.append(d['id'])
  for i,event in enumerate(d['events']):
   if event['role']==0:allowed.add(f"{d['id']}:{i}:0")
 if any(r['id'] not in allowed for r in records):
  raise ValueError('Selected node not a successfully executed original A action')
 return {'policy':'retain_valid_executed_prefix_nodes_log_reproduced_json_syntax_or_schema_v3','normal_dialogues':sum(bool(d.get('complete')) for d in dialogues),'invalid_schema_dialogues':failures,'invalid_outputs_repaired':False,'resampled_for_quality':False,'validated_selected_nodes':len(records)}
