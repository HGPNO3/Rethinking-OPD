import concurrent.futures
import tempfile
import threading
import time
import unittest
from pathlib import Path
from budgetsi.teacher_schedule import FairScheduler,Cohort
from budgetsi.teacher_inference import VLLMTeacher


class SchedulingTests(unittest.TestCase):
    def test_bounded_parallel_execution_and_trace(self):
        lock=threading.Lock();active=0;peak=0
        def execute(r):
            nonlocal active,peak
            with lock:active+=1;peak=max(peak,active)
            time.sleep(.02)
            with lock:active-=1
            return r['payload']
        with tempfile.TemporaryDirectory() as d:
            s=FairScheduler(['a','b','c'],execute,Path(d)/'trace',max_active=3,per_run=1)
            with concurrent.futures.ThreadPoolExecutor(max_workers=12) as p:
                fs=[p.submit(s.submit,'abc'[i%3],dict(operation='collect',payload=i)) for i in range(12)]
                self.assertEqual([f.result() for f in fs],list(range(12)))
            self.assertEqual(peak,3);self.assertTrue(s.idle());s.close()
            self.assertEqual(len((Path(d)/'trace').read_text().splitlines()),12)

    def test_short_score_priority_cannot_starve_generation(self):
        # Directly exercise the policy on an already queued mixed workload.
        import collections
        s=object.__new__(FairScheduler);s.score_streak=collections.Counter()
        def item(kind,i):return dict(request=dict(payload=dict(operation=kind)),number=i)
        s.queues={'a':collections.deque([item('generate',0)]+[item('score',i) for i in range(1,5)])}
        self.assertEqual([s.pop_fair('a')['number'] for _ in range(5)],[1,2,0,3,4])

    def test_phase_barrier_drain_and_retired_member(self):
        switches=[];c=Cohort(['a','b'],switches.append,timeout=2)
        with concurrent.futures.ThreadPoolExecutor(2) as p:
            f=p.submit(c.arrive,'a','collected');time.sleep(.02)
            self.assertFalse(f.done());self.assertEqual(switches,[])
            with self.assertRaises(RuntimeError):c.check('a','collect')
            c.check('b','collect');g=p.submit(c.arrive,'b','collected')
            f.result();g.result();self.assertEqual(switches,['hf'])
            f=p.submit(c.arrive,'a','updated',True);g=p.submit(c.arrive,'b','updated',False)
            f.result();g.result();self.assertEqual(c.states,{'a':'finished','b':'collecting'})
            c.arrive('b','collected');c.arrive('b','updated',True)
            self.assertEqual(switches,['hf','vllm','hf','closed'])

    def test_abort_unblocks_other_run(self):
        c=Cohort(['a','b'],lambda _:None,timeout=2)
        with concurrent.futures.ThreadPoolExecutor(1) as p:
            f=p.submit(c.arrive,'a','collected');time.sleep(.02);c.abort('failed peer')
            with self.assertRaises(RuntimeError):f.result()

    def test_abort_during_backend_load_does_not_lock_barrier(self):
        entered=threading.Event();release=threading.Event()
        def switch(_):entered.set();release.wait(2)
        c=Cohort(['a','b'],switch,timeout=3)
        with concurrent.futures.ThreadPoolExecutor(2) as p:
            f=p.submit(c.arrive,'a','collected');g=p.submit(c.arrive,'b','collected')
            self.assertTrue(entered.wait(1));c.abort('operator stop')
            with self.assertRaises(RuntimeError):f.result(timeout=.5)
            release.set()
            with self.assertRaises(RuntimeError):g.result()

    def test_transport_rejects_changed_sampling_before_network(self):
        t=VLLMTeacher({'context':10},Path('/tmp'))
        with self.assertRaises(ValueError):
            t.call(dict(model='teacher',prompt=[1],max_tokens=9,temperature=1.,top_p=1,top_k=-1,stop_token_ids=[2]))

if __name__=='__main__':unittest.main()
