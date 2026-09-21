import threading,time,unittest
from concurrent.futures import ThreadPoolExecutor
from budgetsi.parallel_collect import WorkerPool

class PoolTests(unittest.TestCase):
 def test_two_executors_overlap_and_preserve_request_identity(self):
  barrier=threading.Barrier(2);seen=[];lock=threading.Lock()
  def execute(worker,requests):
   with lock:seen.append(worker)
   barrier.wait(timeout=5)
   return [{'id':r['id']} for r in requests]
  pool=WorkerPool([lambda rs:execute(0,rs),lambda rs:execute(1,rs)],1,0)
  try:
   with ThreadPoolExecutor(2) as threads:
    results=list(threads.map(pool.submit,[{'id':10},{'id':20}]))
   self.assertEqual(results,[{'id':10},{'id':20}]);self.assertEqual(set(seen),{0,1})
  finally:pool.close()
 def test_error_propagates_and_workers_close(self):
  def fail(rs):raise ValueError('explicit backend failure')
  pool=WorkerPool([fail,fail],1,0)
  try:
   with self.assertRaisesRegex(ValueError,'explicit backend'):pool.submit({'id':0})
  finally:pool.close()
  self.assertTrue(all(not w.thread.is_alive() for w in pool.workers))
if __name__=='__main__':unittest.main()
