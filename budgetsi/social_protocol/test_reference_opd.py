"""Offline wiring tests: no model/API calls or quality claims."""
import asyncio,copy,json,unittest
from types import SimpleNamespace
from runner import Pilot,validate_reference_record,teacher_messages

class ReferenceFlow(unittest.TestCase):
 def collect(self):
  calls=[]
  history=[{'role':'user','content':'VISIBLE: online only'}]
  original={'action':{'action_type':'speak','argument':'ORIGINAL_DRAFT'},'generated_token_ids':[81,82,151645],'prompt_token_ids':[21,22],'behavior_logprobs':[-1.,-2.,-.1]}
  class Session:
   done=False;active_role=0
   def goal(self,r):return 'PRIVATE_A_GOAL'
   def clone(self):return Session()
   async def step(self,a):self.active_role=1-self.active_role;self.action=a
   def messages(self,r):return [{'role':'user','content':self.action['argument']}]
  class Model:
   class tok:
    @staticmethod
    def encode(text,**kwargs):return list(text.encode())
   def render(self,messages):return list(json.dumps(messages).encode())
   async def generate(self,messages,seed,kind):
    calls.append((kind,copy.deepcopy(messages)))
    argument={'teacher_expression':'EXPRESSION_REF','teacher_turn':'SELECTED_REF','partner_branch':'PARTNER_REPLY'}[kind]
    return {'action':{'action_type':'speak','argument':argument},'generated_token_ids':[71,151645]}
   async def score(self,prompt,ids,kind):
    calls.append((kind,prompt.copy()))
    return {'prompt_token_ids':prompt.copy(),'target_ids':ids.copy(),'raw_logprobs':[-.5]*len(ids)}
  model=Model();pilot=Pilot(SimpleNamespace(seed=1,snapshot_id='snapshot'),model,model)
  # Preserve branch candidate action when partner is stepped; only test branch scoring, not environment.
  async def goal(messages,target):return {'mean':-.1 if 'SELECTED_REF' in str(messages) else -1.}
  original_step=Session.step
  async def step(self,a):
   if self.active_role==0:self.first=a
   await original_step(self,a)
  Session.step=step
  Session.messages=lambda self,r:[{'role':'user','content':self.first['argument']}]
  async def goal(messages,target):return {'mean':-.1 if 'SELECTED_REF' in str(messages) else -1.}
  pilot.goal=goal
  asyncio.run(pilot.inspect(Session(),history,original,'node'))
  self.assertEqual(pilot.rows[0]['selected'],'turn')
  self.assertEqual(history,[{'role':'user','content':'VISIBLE: online only'}])
  self.assertEqual(pilot.rows[0]['target_text'],'PRIVATE_A_GOAL')
  return pilot.selected[0],calls
 def test_selected_reference_and_causal_target(self):
  record,calls=self.collect();validate_reference_record(record)
  context=str(record['teacher_messages'])
  self.assertIn('SELECTED_REF',context)
  for forbidden in ['ORIGINAL_DRAFT','EXPRESSION_REF','PARTNER_REPLY','PRIVATE_A_GOAL']:
   self.assertNotIn(forbidden,context)
  self.assertEqual(record['target_ids'],[81,82,151645])
  self.assertEqual(record['student_prefix'],[21,22])
  self.assertEqual(record['teacher_score']['prompt_token_ids'],list(json.dumps(record['teacher_messages']).encode()))
  for kind,messages in calls:
   if kind=='partner_branch':self.assertNotIn('reference_action',str(messages))
   if kind.startswith('teacher_'):
    self.assertEqual(json.loads(messages[1]['content'])['original_target_generation'], {'action_type':'speak','argument':'ORIGINAL_DRAFT'})
    for forbidden in ['SELECTED_REF','PARTNER_REPLY']:
     self.assertNotIn(forbidden,str(messages))
 def test_cache_tampering_rejected(self):
  base,_=self.collect()
  for change in [lambda r:r.pop('prompt_version'),lambda r:r.update(target_ids=[71,151645]),lambda r:r['reference_action'].update(argument='OTHER'),lambda r:r['teacher_messages'][0].update(content='OTHER'),lambda r:r.update(loss_mask=[False]*3),lambda r:r.update(teacher_logprobs=[0]*3)]:
   record=copy.deepcopy(base);change(record)
   with self.assertRaises((ValueError,KeyError)):validate_reference_record(record)
 def test_candidate_prompt_no_fallback(self):
  for specialty in ('expression','turn'):
   text=str(teacher_messages([],{'action_type':'speak','argument':'original'},specialty))
   self.assertIn('original_target_generation',text)
   for old in ['If already concise','reproduce the original','No change is required']:
    self.assertNotIn(old,text)
if __name__=='__main__':unittest.main()
