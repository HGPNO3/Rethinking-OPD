import json
import tempfile
import threading
import time
import unittest
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
import torch
from transformers import GPT2Config, GPT2LMHeadModel
from budgetsi.parallel_collect import generate_batch, BatchWorker, make_parallel_engine
from budgetsi.social_loop import start_server

class ParallelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)
        self.model = GPT2LMHeadModel(GPT2Config(vocab_size=23,n_positions=40,n_embd=16,n_layer=1,n_head=2,
            bos_token_id=1,eos_token_id=22,pad_token_id=0,resid_pdrop=0,embd_pdrop=0,attn_pdrop=0)).eval()
        self.tok = SimpleNamespace(pad_token_id=0)
    def request(self, prompt, seed=123):
        return dict(model='student',prompt=prompt,seed=seed,max_tokens=9-len(prompt),
                    temperature=.7,top_p=1,top_k=-1,stop_token_ids=[22])
    def test_batch_matches_single_and_original_probability(self):
        rs=[self.request([1,2,3]),self.request([1,4],17)]
        before=torch.random.get_rng_state().clone()
        batch=generate_batch(self.model,self.tok,rs,9)
        self.assertTrue(torch.equal(before,torch.random.get_rng_state()))
        for r,result in zip(rs,batch):
            single=generate_batch(self.model,self.tok,[r],9)[0]
            self.assertEqual(result['choices'][0]['token_ids'],single['choices'][0]['token_ids'])
            ids=result['choices'][0]['token_ids']
            self.assertLessEqual(len(ids),r['max_tokens'])
            for j,(tid,lp) in enumerate(zip(ids,result['choices'][0]['logprobs']['token_logprobs'])):
                with torch.no_grad():
                    expected=(self.model(torch.tensor([r['prompt']+ids[:j]])).logits[0,-1].float()/.7).log_softmax(-1)[tid]
                self.assertAlmostEqual(lp,float(expected),places=5)
    def test_batch_worker_combines_requests_and_propagates_failure(self):
        sizes=[]
        def work(rows):
            sizes.append(len(rows))
            if rows[0].get('fail'):raise ValueError('test failure')
            return rows
        worker=BatchWorker(work,4,50)
        try:
            with ThreadPoolExecutor(4) as pool:results=list(pool.map(worker.submit,[{'value':i} for i in range(4)]))
            self.assertEqual([r['value'] for r in results],list(range(4)))
            self.assertIn(4,sizes)
            with self.assertRaises(ValueError):worker.submit({'fail':True})
        finally:worker.close()
    def test_t1_batch_probabilities_and_bound_engine(self):
        requests=[{**self.request([1,2,3]),'temperature':1.},
                  {**self.request([1,4],17),'temperature':1.}]
        batch=generate_batch(self.model,self.tok,requests,9)
        for request,result in zip(requests,batch):
            single=generate_batch(self.model,self.tok,[request],9)[0]
            self.assertEqual(result['choices'][0]['token_ids'],single['choices'][0]['token_ids'])
            ids=result['choices'][0]['token_ids']
            for j,(token,lp) in enumerate(zip(ids,result['choices'][0]['logprobs']['token_logprobs'])):
                with torch.no_grad():
                    expected=self.model(torch.tensor([request['prompt']+ids[:j]])).logits[0,-1].float().log_softmax(-1)[token]
                self.assertAlmostEqual(lp,float(expected),places=5)
        with tempfile.TemporaryDirectory() as folder:
            engine=make_parallel_engine(self.model,None,{'student':self.tok},9,Path(folder),temperature=1.)
            engine.accepting=True;engine.snapshot='test'
            try:
                engine.call(requests[0])
                with self.assertRaisesRegex(ValueError,'temperature'):
                    engine.call(self.request([1,2]))
            finally:engine.close_collection();engine.close()
    def test_http_reaches_real_batched_forward_and_closes(self):
        with tempfile.TemporaryDirectory() as folder:
            engine=make_parallel_engine(self.model,None,{'student':self.tok},9,Path(folder),batch_size=4,wait_ms=150)
            engine.accepting=True;engine.snapshot='test'
            server=start_server(engine)
            def request(i):
                body=json.dumps(self.request([1,2],i)).encode()
                with urllib.request.urlopen(urllib.request.Request(f'http://127.0.0.1:{server.server_port}/v1/completions',data=body),timeout=15) as r:return json.load(r)
            try:
                with ThreadPoolExecutor(4) as pool:results=list(pool.map(request,range(4)))
                self.assertEqual(len(results),4)
                self.assertTrue(any(r['batch_size']>1 for r in engine.requests))
                engine.close_collection()
                with self.assertRaises(RuntimeError):engine.call(self.request([1,2]))
            finally:server.shutdown();server.server_close();engine.close()
    def test_close_waits_for_outstanding_remote_call(self):
        entered=threading.Event();release=threading.Event()
        class Remote:
            def score_support(self):pass
            def call(self,*args):
                entered.set();release.wait(5)
                return {'usage':{'prompt_tokens':1,'completion_tokens':1}}
        with tempfile.TemporaryDirectory() as folder:
            engine=make_parallel_engine(None,Remote(),{},9,Path(folder));engine.accepting=True
            with ThreadPoolExecutor(2) as pool:
                result=pool.submit(engine.call,{'model':'teacher'});self.assertTrue(entered.wait(2))
                closing=pool.submit(engine.close_collection);time.sleep(.03);self.assertFalse(closing.done())
                release.set();result.result();closing.result()
            self.assertEqual(engine.active,0);self.assertFalse(engine.accepting);engine.close()

class Qwen35ParallelTests(ParallelTests):
    def setUp(self):
        try:
            from transformers import Qwen3_5TextConfig, Qwen3_5ForCausalLM
        except ImportError:
            self.skipTest('Qwen3.5 requires Transformers 5.x')
        # These tiny numerical tests deliberately run on CPU. Installed optional
        # CUDA kernels must not override the documented PyTorch reference path.
        import inspect
        from unittest.mock import patch
        from transformers.models.qwen3_5 import modeling_qwen3_5 as module
        for name in ('causal_conv1d_fn','causal_conv1d_update',
                     'torch_recurrent_gated_delta_rule','torch_chunk_gated_delta_rule'):
            guard=patch.object(module,name,None if name.startswith('causal_conv1d') else inspect.unwrap(getattr(module,name)))
            guard.start();self.addCleanup(guard.stop)
        for name in ('FusedRMSNormGated','chunk_gated_delta_rule','fused_recurrent_gated_delta_rule'):
            guard=patch.object(module,name,None)
            guard.start();self.addCleanup(guard.stop)
        torch.set_num_threads(1)
        torch.manual_seed(7)
        cfg=Qwen3_5TextConfig(vocab_size=23,hidden_size=32,intermediate_size=64,
            num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=1,head_dim=16,
            linear_key_head_dim=16,linear_value_head_dim=16,linear_num_key_heads=2,
            linear_num_value_heads=2,layer_types=['linear_attention','full_attention'],
            max_position_embeddings=128,bos_token_id=1,eos_token_id=22,pad_token_id=0)
        self.model=Qwen3_5ForCausalLM(cfg).eval()
        self.tok=SimpleNamespace(pad_token_id=0)

class Qwen35ScoreTests(Qwen35ParallelTests):
    def test_response_only_logits_equal_full_forward(self):
        from budgetsi.top16 import response_logits
        prompt, target = [1,2,3], [4,5,22]
        with torch.no_grad():
            full=self.model(torch.tensor([prompt+target]),use_cache=False).logits[0,2:5]
            selected=response_logits(self.model,prompt,target)
        torch.testing.assert_close(selected,full,rtol=1e-5,atol=1e-7)  # GEMM shape changes FP32 rounding


class RemoteBatchTests(unittest.TestCase):
    def test_teacher_rpc_support_and_parallel_generation(self):
        from budgetsi.remote_teacher import TeacherService, serve, RemoteTeacher, compact_scores
        from budgetsi.variant_spec import get_variant
        torch.set_num_threads(1)
        model=GPT2LMHeadModel(GPT2Config(vocab_size=23,n_positions=40,n_embd=16,n_layer=1,n_head=2,
                                      bos_token_id=1,eos_token_id=22,pad_token_id=0)).eval().requires_grad_(False)
        variant=get_variant('student_top16')
        cfg=dict(schema_version='budgetsi_upstream_variants_online_v1',opd_variant=variant.name,
                 opd=variant.contract(),context=9,collection_parallel={'batch_size':4,'wait_ms':100})
        with tempfile.TemporaryDirectory() as folder:
            service=TeacherService(model,SimpleNamespace(pad_token_id=0),cfg,{'test':'binding'},{},Path(folder))
            server=serve(service,0)
            thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
            try:
                remote=RemoteTeacher(f'http://127.0.0.1:{server.server_port}/',{'test':'binding'})
                def generate(seed):
                    return remote.call('collect',dict(model='teacher',prompt=[1,2],seed=seed,max_tokens=7,
                                temperature=.7,top_p=1,top_k=-1,stop_token_ids=[22]))
                with ThreadPoolExecutor(4) as pool:list(pool.map(generate,range(4)))
                self.assertTrue(any(r['batch_size']>1 for r in service.engine.requests))
                action=SimpleNamespace(teacher_prompt_ids=(1,2),target_ids=(3,22))
                ids=torch.arange(16).expand(2,-1)
                actual=remote.score_support(action,ids,16)
                expected=compact_scores(model,[1,2],[3,22],ids.tolist(),16)
                for key in actual:torch.testing.assert_close(actual[key],torch.tensor(expected[key]),rtol=0,atol=0)
                self.assertTrue(remote.assert_frozen()['frozen'])
            finally:
                server.shutdown();server.server_close();service.engine.close();thread.join()

if __name__=='__main__':unittest.main()

class TeacherBurstTests(unittest.TestCase):
    def test_teacher_burst_is_bounded_and_each_request_runs_once(self):
        class Remote:
            def __init__(self):
                self.lock=threading.Lock();self.active=0;self.peak=0;self.seen=[]
            def score_support(self):pass
            def call(self,operation,data):
                with self.lock:
                    self.active+=1;self.peak=max(self.peak,self.active);self.seen.append(data['id'])
                try:
                    time.sleep(.03)
                    if data['id']==7:raise ValueError('upstream failure')
                    return {'usage':{'prompt_tokens':1,'completion_tokens':1}}
                finally:
                    with self.lock:self.active-=1
        teacher=Remote()
        with tempfile.TemporaryDirectory() as folder:
            engine=make_parallel_engine(None,teacher,{},9,Path(folder));engine.accepting=True
            try:
                with ThreadPoolExecutor(32) as pool:
                    futures=[pool.submit(engine.call,{'model':'teacher','id':i}) for i in range(64)]
                    for i,future in enumerate(futures):
                        if i==7:
                            with self.assertRaisesRegex(ValueError,'upstream failure'):future.result()
                        else:future.result()
                self.assertGreater(teacher.peak,1);self.assertLessEqual(teacher.peak,8)
                self.assertEqual(sorted(teacher.seen),list(range(64)));self.assertEqual(engine.active,0)
            finally:engine.close()
