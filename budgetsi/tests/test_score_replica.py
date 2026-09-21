import tempfile, threading, time, unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from budgetsi.gpu_lease import device_lease, teacher_scoring_lease


class LeaseTests(unittest.TestCase):
    def test_readers_overlap_but_teacher_excludes_readers(self):
        with tempfile.TemporaryDirectory() as folder:
            path=str(Path(folder)/'gpu.lease');entered=threading.Event();release=threading.Event()
            def reader():
                with device_lease(path):entered.set();release.wait(3)
            with ThreadPoolExecutor(2) as pool:
                first=pool.submit(reader);self.assertTrue(entered.wait(1))
                with device_lease(path):pass
                teacher=pool.submit(lambda: self.acquire_writer(path))
                time.sleep(.05);self.assertFalse(teacher.done());release.set()
                first.result(2);self.assertTrue(teacher.result(2))
            entered.clear();release.clear()
            with ThreadPoolExecutor(1) as pool:
                with device_lease(path,exclusive=True):
                    future=pool.submit(reader);time.sleep(.05);self.assertFalse(entered.is_set())
                self.assertTrue(entered.wait(1));release.set();future.result(1)

    def acquire_writer(self,path):
        with device_lease(path,exclusive=True):return True

    def test_lease_releases_after_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            path=str(Path(folder)/'gpu.lease')
            with self.assertRaises(ValueError):
                with device_lease(path,exclusive=True):raise ValueError('failure')
            self.assertTrue(self.acquire_writer(path))

    def test_teacher_wrapper_keeps_numerical_dispatch_and_checks_binding(self):
        with tempfile.TemporaryDirectory() as folder:
            class Service:
                def resolve_config(self,b):
                    if b!='valid':raise ValueError('binding')
                    return {'student_score_replica':{'lease_path':str(Path(folder)/'gpu.lease')}}
                @teacher_scoring_lease
                def dispatch(self,r):return {'unchanged':r['payload']}
            with patch('budgetsi.gpu_lease.release_transient_cuda') as cleanup:
                self.assertEqual(Service().dispatch({'binding':'valid','operation':'opd','payload':[1,2]}),{'unchanged':[1,2]})
                cleanup.assert_called_once()
                with self.assertRaisesRegex(ValueError,'binding'):
                    Service().dispatch({'binding':'bad','operation':'opd','payload':[]})


class ScoreQueueTests(unittest.TestCase):
    def test_generation_can_progress_while_external_score_waits_and_close_drains(self):
        from budgetsi.parallel_collect import make_parallel_engine
        score_entered=threading.Event();score_release=threading.Event();generation_done=threading.Event()
        class Scorer:
            device='2'
            def score(self,request,*,snapshot):
                if snapshot!='current':raise ValueError('stale')
                score_entered.set();score_release.wait(3)
                return {'raw_logprobs':[-1.], 'usage':{'prompt_tokens':1,'completion_tokens':1}}
        def generate(*args):
            generation_done.set()
            return [{'usage':{'prompt_tokens':1,'completion_tokens':1},'choices':[]}]
        with tempfile.TemporaryDirectory() as folder, patch('budgetsi.parallel_collect.generate_batch',side_effect=generate):
            engine=make_parallel_engine(object(),None,{'student':object()},9,Path(folder),temperature=1.,student_score_replica=Scorer())
            engine.snapshot='current';engine.accepting=True
            try:
                with ThreadPoolExecutor(3) as pool:
                    score=pool.submit(engine.call,{'model':'student','operation':'score','prompt_ids':[1],'target_ids':[2]})
                    self.assertTrue(score_entered.wait(1))
                    gen=pool.submit(engine.call,{'model':'student','temperature':1.})
                    self.assertTrue(generation_done.wait(1));gen.result(1)
                    closing=pool.submit(engine.close_collection);time.sleep(.05);self.assertFalse(closing.done())
                    score_release.set();self.assertEqual(score.result(1)['raw_logprobs'],[-1.]);closing.result(1)
                self.assertEqual(engine.active,0)
            finally:score_release.set();engine.close()


if __name__=='__main__':unittest.main()
