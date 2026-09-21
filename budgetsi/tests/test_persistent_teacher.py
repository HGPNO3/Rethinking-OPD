import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch, Mock
from budgetsi.persistent_teacher import IndependentPhases, PersistentTeacher
from budgetsi.remote_teacher import digest
from budgetsi.teacher_schedule import FairScheduler
from budgetsi.teacher_inference import VLLMTeacher


class PhasesTests(unittest.TestCase):
    def test_generation_uses_bound_temperature_and_rejects_mismatch(self):
        t = VLLMTeacher({'context':8,'temperature':1.0},Path('/tmp'))
        data=dict(model='teacher',prompt=[1],max_tokens=7,temperature=1.0,
                  top_p=1,top_k=-1,stop_token_ids=[2],seed=1)
        response={'choices':[dict(token_ids=[2],logprobs={'token_logprobs':[-.5]},finish_reason='stop')]}
        with patch('budgetsi.teacher_inference.post',return_value=response) as call:
            t.call(data)
            self.assertEqual(call.call_args.args[1]['temperature'],1.0)
        with self.assertRaises(ValueError):t.call({**data,'temperature':.7})

    def test_one_run_updates_without_waiting_for_other_collection(self):
        p = IndependentPhases(['a','b'])
        self.assertEqual(p.arrive('a','collected')['state'],'updating')
        p.enter('a','opd'); p.enter('b','collect')
        p.leave('a'); p.leave('b')
        self.assertEqual(p.arrive('a','updated',True)['state'],'finished')
        self.assertEqual(p.states['b'],'collecting')
        p.arrive('b','collected')
        self.assertTrue(p.arrive('b','updated',True)['all_finished'])

    def test_outstanding_request_and_invalid_transitions_fail(self):
        p = IndependentPhases(['a','b']); p.enter('a','collect')
        with self.assertRaises(RuntimeError): p.arrive('a','collected')
        p.leave('a')
        with self.assertRaises(ValueError): p.arrive('a','collected',True)
        with self.assertRaises(RuntimeError): p.enter('a','opd')
        p.abort('infrastructure failure')
        with self.assertRaisesRegex(RuntimeError,'infrastructure failure'): p.arrive('b','collected')

    def test_runtime_requires_disjoint_devices_and_explicit_python(self):
        r=dict(backend='vllm_persistent_hf_v1',python='/venv/vllm/bin/python',hf_python='/venv/train/bin/python',cuda_visible_devices='0,1',hf_cuda_visible_devices='2,3',tp=2,pp=1)
        PersistentTeacher.validate_runtime(r)
        for updates in [dict(hf_cuda_visible_devices='1,2'),dict(cuda_visible_devices='0,0'),dict(hf_python='python'),dict(hf_port=18741)]:
            with self.assertRaises(ValueError): PersistentTeacher.validate_runtime({**r,**updates})

    def test_independent_queues_do_not_block_generation_behind_exact_score(self):
        blocked=threading.Event(); release=threading.Event(); generated=threading.Event()
        def execute(r):
            if r['operation']=='opd':blocked.set();release.wait(3)
            else:generated.set()
            return r['operation']
        with tempfile.TemporaryDirectory() as d:
            gen=FairScheduler(['a','b'],execute,Path(d)/'gen.jsonl',max_active=2,per_run=1)
            score=FairScheduler(['a','b'],execute,Path(d)/'score.jsonl',max_active=1,per_run=1)
            thread=threading.Thread(target=lambda:score.submit('a',{'operation':'opd'}));thread.start()
            try:
                self.assertTrue(blocked.wait(1))
                self.assertEqual(gen.submit('b',{'operation':'collect'}),'collect')
                self.assertTrue(generated.is_set())
            finally:
                release.set();thread.join();gen.close();score.close()

    def test_bound_response_and_last_finish_lifecycle(self):
        a,b={'run':'a'},{'run':'b'};keys=[digest(a),digest(b)]
        t=PersistentTeacher.__new__(PersistentTeacher)
        t.bindings={digest(a):a,digest(b):b};t.phases=IndependentPhases(keys)
        t.check_alive=Mock();t.stop_backends=Mock();t.assets={};t.runtime={}
        def request(binding,action,done=False):return dict(binding=binding,operation='phase',payload=dict(action=action,done=done))
        for binding in [a,b]:
            t.dispatch(request(binding,'collected'))
            r=t.dispatch(request(binding,'updated',True))
            self.assertEqual(r['binding'],binding)
            if binding==a:t.stop_backends.assert_not_called()
        t.stop_backends.assert_called_once()
        with self.assertRaises(ValueError):t.dispatch(request({'run':'outsider'},'collected'))
        q=dict(binding=a,operation='opd',payload={})
        with patch('budgetsi.persistent_teacher.post',return_value={'binding':b,'request_sha256':digest(q),'result':{}}):
            with self.assertRaises(ValueError):t.hf_call(q)

if __name__=='__main__': unittest.main()
