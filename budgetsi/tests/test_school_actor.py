"""Run pinned upstream batching/update/loss bodies on CPU tensor fixtures.

Only distributed containers and the model forward are replaced. This is not a
Ray/FSDP/model-runtime acceptance test; that must run on the destination GPUs.
"""
import copy
import heapq
import unittest
from types import SimpleNamespace

import torch
from omegaconf import OmegaConf

from budgetsi.audit.source_probe import function, numerical_functions
from budgetsi.top16 import actor_overrides, school_actor_config, validate_actor


class Data:
    def __init__(self, tensors, non_tensors=None, meta_info=None):
        self.batch=tensors
        self.non_tensor_batch=non_tensors or {}
        self.meta_info=meta_info or {}

    @classmethod
    def from_dict(cls, tensors, non_tensors=None, meta_info=None):
        return cls(tensors,non_tensors,meta_info)

    def select(self,batch_keys,non_tensor_batch_keys):
        return Data({key:self.batch[key] for key in batch_keys},
                    {key:self.non_tensor_batch[key] for key in non_tensor_batch_keys},self.meta_info)

    def split(self,count):
        n=len(self.batch['input_ids'])
        return [Data({key:value[i:i+count] for key,value in self.batch.items()},meta_info=self.meta_info)
                for i in range(0,n,count)]

    def to(self,device):
        return self


def upstream():
    ns=numerical_functions()
    ns.update(copy=copy,heapq=heapq,DataProto=Data,
              dist=SimpleNamespace(is_initialized=lambda:False),get_device_name=lambda:'cpu',
              get_device_id=lambda:'cpu',
              tu=SimpleNamespace(index_select_tensor_dict=lambda batch,idx:{k:v[idx] for k,v in batch.items()}),
              append_to_dict=lambda target,values:[target.setdefault(k,[]).append(v) for k,v in values.items()])
    path='verl/verl/utils/seqlen_balancing.py'
    for name in ('calculate_workload','karmarkar_karp','get_seqlen_balanced_partitions',
                 'ceildiv','roundup_divisible','rearrange_micro_batches','prepare_dynamic_batch'):
        function(path,name,ns,True)
    ns['get_policy_loss_fn']=lambda name:ns['compute_policy_loss_vanilla'] if name=='vanilla' else None
    function('verl/verl/workers/actor/dp_actor.py','update_policy',ns,True,parent='DataParallelPPOActor')
    return ns


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.logits=torch.nn.Parameter(torch.arange(36,dtype=torch.float64).reshape(3,3,4)/19)

    def scores(self,inputs):
        rows=inputs['input_ids'][:,0]
        return self.logits[rows].log_softmax(-1).gather(-1,inputs['responses'][...,None]).squeeze(-1)


def fixture(model,ns):
    inputs=torch.tensor([[0,1,1,1,1,1],[1,1,1,0,0,0],[2,1,1,1,0,0]])
    tensors=dict(input_ids=inputs,attention_mask=torch.tensor([[1]*6,[1,1,1,0,0,0],[1,1,1,1,0,0]]),
                 position_ids=torch.arange(6).expand(3,-1),responses=torch.tensor([[1,2,3],[0,0,0],[2,1,0]]),
                 response_mask=torch.tensor([[1,1,1],[1,0,0],[1,1,0]],dtype=torch.float64))
    old=model.scores(tensors).detach()
    reward=torch.tensor([[.2,-.1,.3],[-.4,0,0],[.1,.2,0]],dtype=torch.float64)
    advantages,_=ns['compute_token_reward_direct_advantage'](reward,tensors['response_mask'])
    tensors.update(old_log_probs=old,advantages=advantages)
    return Data(tensors,meta_info={'temperature':1.,'top_k':0})


class SchoolActor(unittest.TestCase):
    def test_dynamic_config_retains_old_recipe_and_guards_context(self):
        legacy=actor_overrides(12)
        self.assertNotIn('use_dynamic_bsz',legacy)
        self.assertEqual(legacy['ppo_micro_batch_size_per_gpu'],1)
        cfg=school_actor_config(12,40960)
        validate_actor(cfg,12)
        self.assertTrue(cfg.use_dynamic_bsz)
        self.assertEqual(cfg.ppo_max_token_len_per_gpu,40960)
        with self.assertRaises(ValueError):school_actor_config(12,40960,32768)
        cfg.ppo_max_token_len_per_gpu=32768
        with self.assertRaises(ValueError):validate_actor(cfg,12)
        with self.assertRaises(ValueError):validate_actor(school_actor_config(12),12,world_size=2)

    def test_saved_tensor_cpu_offload_preserves_pinned_update_math(self):
        # CPU checks numerical semantics only; the failing production GPU batch
        # must separately establish that offloading actually reduces peak VRAM.
        with torch.autograd.graph.save_on_cpu(pin_memory=True):
            self.test_pinned_dynamic_actor_gradient_and_adamw_update_match()

    def test_pinned_dynamic_actor_gradient_and_adamw_update_match(self):
        ns=upstream()
        model=Model();reference=copy.deepcopy(model)
        data=fixture(model,ns);cfg=school_actor_config(3,max_context=6,max_token_len_per_gpu=8)
        batches,indices=ns['prepare_dynamic_batch'](data,max_token_len=8)
        self.assertEqual(sorted(i for group in indices for i in group),[0,1,2])
        self.assertEqual(len(batches),2)
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-5,weight_decay=.01)
        ref_optimizer=torch.optim.AdamW(reference.parameters(),lr=1e-5,weight_decay=.01)
        expected_loss=0
        for batch in batches:
            lp=reference.scores(batch.batch)
            loss,_=ns['compute_policy_loss_vanilla'](lp.detach(),lp,batch.batch['advantages'],
                    batch.batch['response_mask'],loss_agg_mode='token-mean',config=cfg)
            expected_loss=expected_loss+loss*len(batch.batch['responses'])/3
        expected_loss.backward()
        torch.nn.utils.clip_grad_norm_(reference.parameters(),1.)
        expected_gradient=reference.logits.grad.detach().clone()
        ref_optimizer.step()
        counts={'optimizer_steps':0,'forward_rows':[],'policy_calls':0}
        gradients=[]
        actual_loss=ns['compute_policy_loss_vanilla']
        def tracked_loss(**kwargs):
            counts['policy_calls']+=1
            return actual_loss(**kwargs)
        ns['get_policy_loss_fn']=lambda name:tracked_loss
        def forward(inputs,**kwargs):
            self.assertEqual(kwargs['temperature'],1.)
            counts['forward_rows'].extend(inputs['input_ids'][:,0].tolist())
            return None,model.scores(inputs),None,None
        def step():
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            gradients.append(model.logits.grad.detach().clone())
            optimizer.step();counts['optimizer_steps']+=1
            return norm
        actor=SimpleNamespace(actor_module=model,actor_optimizer=optimizer,config=cfg,
                              ulysses_sequence_parallel_size=1,_forward_micro_batch=forward,_optimizer_step=step)
        ns['update_policy'](actor,data)
        self.assertEqual(counts['optimizer_steps'],1)
        self.assertEqual(counts['policy_calls'],len(batches))
        self.assertEqual(sorted(counts['forward_rows']),[0,1,2])
        torch.testing.assert_close(gradients[0],expected_gradient,rtol=1e-12,atol=1e-12)
        torch.testing.assert_close(model.logits,reference.logits,rtol=1e-12,atol=1e-12)
        self.assertTrue(all(int(s['step'])==1 for s in optimizer.state.values()))

    def test_dynamic_weighting_is_not_claimed_as_global_token_mean(self):
        ns=upstream();model=Model();data=fixture(model,ns)
        cfg=school_actor_config(3,max_context=6,max_token_len_per_gpu=8)
        batches,_=ns['prepare_dynamic_batch'](data,max_token_len=8)
        losses=[]
        for batch in batches:
            lp=model.scores(batch.batch)
            losses.append(ns['compute_policy_loss_vanilla'](lp.detach(),lp,batch.batch['advantages'],
                           batch.batch['response_mask'],config=cfg)[0]*len(batch.batch['responses'])/3)
        dynamic_gradient=torch.autograd.grad(sum(losses),model.logits)[0]
        lp=model.scores(data.batch)
        full_loss=ns['compute_policy_loss_vanilla'](lp.detach(),lp,data.batch['advantages'],data.batch['response_mask'],config=cfg)[0]
        global_gradient=torch.autograd.grad(full_loss,model.logits)[0]
        self.assertFalse(torch.allclose(dynamic_gradient,global_gradient))


if __name__=='__main__':unittest.main()
