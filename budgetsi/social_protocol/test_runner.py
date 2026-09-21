import unittest
from runner import strict_action,select,teacher_messages,opd_messages,seed_for,Client,gather_cancel
import asyncio
class Check(unittest.TestCase):
 def test_parse_strict(self):
  for x in ['{"action_type":"speak","action_type":"leave","argument":""}', '```json\n{"action_type":"speak","argument":"x"}\n```','{"action_type":"speak","argument":NaN}']:
   with self.assertRaises(ValueError):strict_action(x)
  self.assertEqual(strict_action('{"action_type":"action","argument":"wave"}')['action_type'],'action')
 def candidate(self,specialty,ig,g,tokens,ratio):
  return dict(status='ok',specialty=specialty,branch={'ig':ig},goal_gain=g,token_cost={'action_tokens':tokens},cost_ratio=ratio)
 def test_pareto(self):
  rows=[self.candidate('expression',.09,-.01,60,.6),self.candidate('turn',.15,.05,90,.9)]
  self.assertEqual(select(rows,0)[0],'turn');self.assertEqual(len(select(rows,0)[1]),2)
 def test_nonpositive_own_ig_rejected_even_with_positive_gain(self):
  for ig in [-.1,0]:
   self.assertEqual(select([self.candidate('turn',ig,.2,20,.2)],0),(None,[]))
 def test_expression_may_lose_ig_but_must_save_tokens(self):
  self.assertEqual(select([self.candidate('expression',.09,-.01,60,.6)],0)[0],'expression')
  self.assertEqual(select([self.candidate('expression',.2,.1,100,1)],0),(None,[]))
 def test_strategy_requires_improvement(self):
  self.assertEqual(select([self.candidate('turn',.1,0,20,.2)],0),(None,[]))
 def test_efficiency_can_beat_larger_ig(self):
  rows=[self.candidate('expression',.09,-.01,20,.2),self.candidate('turn',.15,.05,90,.9)]
  self.assertEqual(select(rows,1)[0],'expression')
 def test_bad_score(self):
  for bad in [float('nan'),float('inf')]:
   with self.assertRaises(ValueError):select([self.candidate('turn',bad,.1,20,.2)],0)
 def test_unchanged_is_excluded(self):
  r=self.candidate('expression',.1,0,20,.2);r['unchanged']=True
  self.assertEqual(select([r],0),(None,[]))
 def test_context_boundary(self):
  m=[{'role':'user','content':'visible history'}];original='UNIQUE_CURRENT_ACTION'
  self.assertIn(original,str(teacher_messages(m,{'action_type':'speak','argument':original},'turn')))
  self.assertNotIn(original,str(opd_messages(m,'turn',{'action_type':'speak','argument':'REFERENCE'})))
  self.assertEqual(m,[{'role':'user','content':'visible history'}])
 def test_score_span(self):
  c=Client.__new__(Client);c.max_context=100;c.model='student'
  async def request(payload,kind):
   self.assertEqual(payload['prompt'],[11,12,13,14]);self.assertEqual(payload['max_tokens'],1)
   return {'choices':[{'prompt_logprobs':[None,{'12':{'logprob':-.1}}, {'13':{'logprob':-.2}}, {'14':-.3}],'token_ids':[15]}]}
  c.request=request
  out=asyncio.run(c.score([11,12],[13,14],'test'))
  self.assertEqual(out['raw_logprobs'],[-.2,-.3]);self.assertEqual(out['discarded_score_completion_ids'],[15])
 def test_transport_retry_identical(self):
  import aiohttp
  class Response:
   status=200
   async def json(self):return {'usage':{'completion_tokens':2}}
  class Attempt:
   def __init__(self,fail):self.fail=fail
   async def __aenter__(self):
    if self.fail:raise aiohttp.ServerDisconnectedError('test transient')
    return Response()
   async def __aexit__(self,*args):return False
  class HTTP:
   def __init__(self):self.calls=[]
   def post(self,url,json):
    self.calls.append(json.copy());return Attempt(len(self.calls)<3)
  c=Client.__new__(Client);c.http=HTTP();c.url='local';c.model='student';c.metrics=[]
  asyncio.run(c.request({'seed':42,'prompt':[1,2]},'test'))
  self.assertEqual(len(c.http.calls),3);self.assertTrue(all(x==c.http.calls[0] for x in c.http.calls));self.assertEqual(c.metrics[0]['attempts'],3)
 def test_gather_cancel_awaits_cleanup(self):
  state=[]
  async def slow():
   try:await asyncio.sleep(30)
   finally:state.append('cleanup')
  async def fail():
   await asyncio.sleep(0);raise ValueError('hard failure')
  with self.assertRaises(ValueError):asyncio.run(gather_cancel(slow(),fail()))
  self.assertEqual(state,['cleanup'])
 def test_identical_parsed_actions_share_one_partner_branch(self):
  from runner import Pilot
  from types import SimpleNamespace
  action={'action_type':'speak','argument':'Hi'}
  gen={'action':action,'raw_text':'different whitespace is harmless','generated_token_ids':[1,2],'prompt_token_ids':[3]}
  calls=[]
  class Session:
   done=False;active_role=0
   def goal(self,r):return 'goal'
   def clone(self):return Session()
   async def step(self,a):self.active_role=1-self.active_role
   def messages(self,r):return [{'role':'user','content':'history'}]
  class Model:
   class tok:
    @staticmethod
    def encode(text,**kwargs):return list(text.encode())
   async def generate(self,*a):
    if a[-1].startswith('teacher_'):
     self_outer.assertNotIn('different whitespace is harmless',str(a[0]))
    calls.append(a[-1]);return gen.copy()
  self_outer=self;model=Model();pilot=Pilot(SimpleNamespace(seed=1),model,model)
  async def goal(*a):return {'mean':-.5}
  pilot.goal=goal
  asyncio.run(pilot.inspect(Session(),[],gen,'node'))
  self.assertEqual(calls.count('partner_branch'),1)
  self.assertIsNone(pilot.rows[0]['selected'])
  self.assertEqual(pilot.rows[0]['unique_branch_count'],1)
  self.assertTrue(all(c['unchanged'] and c['goal_gain']==0 for c in pilot.rows[0]['candidates']))
 def test_seed(self):
  self.assertEqual(seed_for(1,'a'),seed_for(1,'a'));self.assertNotEqual(seed_for(1,'a'),seed_for(1,'b'))
if __name__=='__main__':unittest.main()
