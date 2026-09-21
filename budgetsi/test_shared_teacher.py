import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor
import torch
from transformers import GPT2Config, GPT2LMHeadModel
from budgetsi.remote_teacher import SharedTeacherService, digest
from budgetsi.variant_spec import get_variant

class SharedTeacherTests(unittest.TestCase):
    def test_binding_specific_k_and_shared_generation_worker(self):
        torch.set_num_threads(1)
        model=GPT2LMHeadModel(GPT2Config(vocab_size=23,n_positions=40,n_embd=16,n_layer=1,n_head=2,
                                      bos_token_id=1,eos_token_id=22,pad_token_id=0)).eval().requires_grad_(False)
        configs=[];bindings=[]
        for name in ('student_top16','union_student','sampled_token'):
            v=get_variant(name)
            configs.append(dict(schema_version='budgetsi_upstream_variants_online_v1',opd_variant=name,
                opd=v.contract(),context=9,teacher='same',temperature=.7,teacher_temperature=1.,diagnostics=dict(k=16),
                collection_parallel=dict(batch_size=4,wait_ms=100)))
            bindings.append({'config_sha256':digest(configs[-1]),'git_commit':'test'})
        with tempfile.TemporaryDirectory() as folder:
            service=SharedTeacherService(model,SimpleNamespace(pad_token_id=0),configs,bindings,{},Path(folder))
            def call(i,operation,payload):return service.dispatch(dict(binding=bindings[i],operation=operation,payload=payload))
            try:
                def generate(i):return call(i,'collect',dict(model='teacher',prompt=[1,2],seed=i,max_tokens=7,
                                temperature=.7,top_p=1,top_k=-1,stop_token_ids=[22]))
                with ThreadPoolExecutor(3) as pool:results=list(pool.map(generate,range(3)))
                self.assertEqual([r['binding'] for r in results],bindings)
                self.assertEqual(len(service.engine.workers),1)
                self.assertTrue(any(r['batch_size']==3 for r in service.engine.requests))
                self.assertEqual({r['experiment_binding'] for r in service.engine.requests},{digest(b) for b in bindings})
                score=dict(prompt=[1,2],target=[3,22],ids=[list(range(16))]*2,k=16)
                self.assertIn('teacher',call(0,'opd',score)['result'])
                with self.assertRaises(ValueError):call(2,'opd',score)
                self.assertIn('teacher_sampled',call(2,'opd',{**score,'ids':[],'k':0})['result'])
                from budgetsi.diagnostics import student_summary,compare
                from budgetsi.top16 import response_logits
                logits=response_logits(model,[1,2],[3,22])
                summary=student_summary(logits,.7,16)
                payload=dict(prompt=[1,2],target=[3,22],summary={k:v.tolist() if isinstance(v,torch.Tensor) else v for k,v in summary.items()})
                diagnostic=call(2,'diagnostics',payload)['result']
                expected=compare(summary,logits,1.)
                for key in expected:torch.testing.assert_close(torch.tensor(diagnostic[key]),expected[key],rtol=0,atol=0)

                with self.assertRaises(ValueError):service.dispatch(dict(binding={'unknown':1},operation='status',payload={}))
            finally:service.engine.close()

if __name__=='__main__':unittest.main()
