import json,tempfile,threading,unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch
from budgetsi.parallel_collect import make_parallel_engine
from budgetsi.execution_migration import normalize_profile_metadata, normalized_formal

class ScoreQueueTests(unittest.TestCase):
 def test_scores_do_not_occupy_replica_generation_worker(self):
  entered=threading.Event();release=threading.Event();replica_entered=threading.Event()
  def scoring(self,data):
   entered.set();release.wait(5)
   return {'usage':{'prompt_tokens':1,'completion_tokens':1},'id':data['id']}
  def generation(*args):return [{'usage':{'prompt_tokens':1,'completion_tokens':1}} for _ in args[2]]
  class Replica:
   snapshot='fixed'
   def generate_batch(self,rs):
    replica_entered.set();return [{'usage':{'prompt_tokens':1,'completion_tokens':1}} for _ in rs]
  with tempfile.TemporaryDirectory() as d,patch('budgetsi.social_loop.Engine.call',scoring),patch('budgetsi.parallel_collect.generate_batch',generation):
   e=make_parallel_engine(object(),None,{'student':None},9,Path(d),batch_size=1,wait_ms=0,temperature=1.,student_replica=Replica());e.accepting=True;e.snapshot='fixed'
   try:
    with ThreadPoolExecutor(4) as pool:
     s1=pool.submit(e.call,{'model':'student','operation':'score','id':1,'profile_kind':'ig_score'});self.assertTrue(entered.wait(2))
     s2=pool.submit(e.call,{'model':'student','operation':'score','id':2,'profile_kind':'ig_score'})
     gs=[pool.submit(e.call,{'model':'student','temperature':1.,'profile_kind':'source_A'}) for _ in range(2)]
     try:self.assertTrue(replica_entered.wait(2))
     finally:release.set()
     self.assertEqual([s1.result()['id'],s2.result()['id']],[1,2]);[x.result() for x in gs]
    rows=[json.loads(x) for x in (Path(d)/'request_profile.jsonl').read_text().splitlines()]
    timed=[x for x in rows if 'queue_seconds' in x];self.assertEqual(len(timed),4)
    self.assertTrue(all(x['queue_seconds']>=0 and x['service_seconds']>=0 for x in timed))
   finally:release.set();e.close()
   self.assertFalse(e.score_workers['student'].thread.is_alive())
 def test_metadata_normalization_does_not_hide_payload_changes(self):
  old="x={'model':self.model,**payload}"
  new="x={'model':self.model,**payload,'profile_kind':kind}"
  self.assertEqual(normalize_profile_metadata(old),normalize_profile_metadata(new))
  self.assertNotEqual(normalize_profile_metadata(old),normalize_profile_metadata(new.replace('**payload','**other')))
 def test_profile_does_not_hide_updated_inputs(self):
  old='def main():\n metrics=update_actor(service,actions,scores)\n'
  new='def main():\n profile_mark(out,"actor_start",batch_index)\n metrics=update_actor(service,actions,scores)\n profile_mark(out,"actor_end",batch_index)\n'
  self.assertEqual(normalized_formal(old),normalized_formal(new))
  self.assertNotEqual(normalized_formal(old),normalized_formal(new.replace('scores)','scores[:1])')))
if __name__=='__main__':unittest.main()


class ScorePriorityTests(unittest.TestCase):
 def test_pending_scores_finish_before_new_primary_generation(self):
  import time
  entered=threading.Event();release=threading.Event();order=[]
  def score(self,data):
   if data['id']==1:entered.set();release.wait(5)
   order.append(data['id']);return {'usage':{'prompt_tokens':1,'completion_tokens':1}}
  def generate(*args):order.append('generate');return [{'usage':{'prompt_tokens':1,'completion_tokens':1}}]
  with tempfile.TemporaryDirectory() as d,patch('budgetsi.social_loop.Engine.call',score),patch('budgetsi.parallel_collect.generate_batch',generate):
   e=make_parallel_engine(object(),None,{'student':None},9,Path(d),batch_size=1,wait_ms=0,temperature=1.);e.accepting=True;e.snapshot='fixed'
   try:
    with ThreadPoolExecutor(3) as pool:
     one=pool.submit(e.call,dict(model='student',operation='score',id=1));self.assertTrue(entered.wait(2))
     two=pool.submit(e.call,dict(model='student',operation='score',id=2))
     deadline=time.monotonic()+2
     while e.pending_scores!=2 and time.monotonic()<deadline:time.sleep(.001)
     self.assertEqual(e.pending_scores,2)
     gen=pool.submit(e.call,dict(model='student',temperature=1.))
     release.set();one.result();two.result();gen.result()
    self.assertEqual(order,[1,2,'generate']);self.assertEqual(e.pending_scores,0)
   finally:release.set();e.close()
