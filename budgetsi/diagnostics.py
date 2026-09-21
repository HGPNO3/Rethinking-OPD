"""Full-vocabulary entropy; explicit student Top-K+tail coarse-grained KL/JS."""
import torch

def student_summary(logits,temperature=.7,k=16):
 if temperature<=0 or k<1 or k>logits.shape[-1]:raise ValueError('Invalid diagnostic temperature/K')
 lp=(logits.float()/temperature).log_softmax(-1)
 vals,ids=lp.topk(k,-1)
 return {'ids':ids.cpu(),'logp':vals.cpu(),'entropy':(-(lp.exp()*lp).sum(-1)).cpu(),'vocab_size':lp.shape[-1]}

@torch.no_grad()
def compare(summary,teacher_logits,temperature=1.):
 if temperature<=0:raise ValueError('Invalid teacher temperature')
 q=(teacher_logits.float()/temperature).log_softmax(-1)
 if q.shape[-1]!=summary['vocab_size'] or q.shape[0]!=summary['ids'].shape[0]:raise ValueError('Vocabulary or response position mismatch')
 ids=summary['ids'].to(q.device);lp=summary['logp'].to(q.device);p=lp.exp();lq=q.gather(-1,ids);probq=lq.exp()
 ti=q.topk(ids.shape[-1],-1).indices
 overlap=(ids.unsqueeze(-1)==ti.unsqueeze(-2)).any(-1)
 n=overlap.sum(-1);adv=lq-lp
 # Partition vocabulary into student Top-K singletons plus one tail bucket.
 P=torch.cat([p,(1-p.sum(-1,keepdim=True)).clamp_min(0)],-1)
 Q=torch.cat([probq,(1-probq.sum(-1,keepdim=True)).clamp_min(0)],-1)
 P=P/P.sum(-1,keepdim=True);Q=Q/Q.sum(-1,keepdim=True);M=(P+Q)/2
 def kl(a,b):return torch.where(a>0,a*(a.clamp_min(1e-30).log()-b.clamp_min(1e-30).log()),0).sum(-1)
 hp=summary['entropy'].to(q.device);hq=-(q.exp()*q).sum(-1)
 return {'overlap_ratio':n.float()/ids.shape[-1],
 'student_overlap_mass':(p*overlap).sum(-1),'teacher_overlap_mass':(probq*overlap).sum(-1),
 'student_topk_mass':p.sum(-1),'teacher_mass_on_student_topk':probq.sum(-1),
 'student_entropy':hp,'teacher_entropy':hq,'teacher_minus_student_entropy':hq-hp,
 'student_topk_tail_kl_pq':kl(P,Q),'student_topk_tail_js':(kl(P,M)+kl(Q,M))/2,
 'overlap_logprob_gap_sum':(adv*overlap).sum(-1),'overlap_token_count':n.float(),
 'empty_overlap':(n==0).float()}

class Accumulator:
 def __init__(self):self.sums={};self.tokens=0
 def add(self,values):
  n=len(values['overlap_ratio']);self.tokens+=n
  for k,v in values.items():
   if len(v)!=n or not torch.isfinite(v).all():raise ValueError('Invalid diagnostic values')
   self.sums[k]=self.sums.get(k,0.)+v.double().sum().item()
 def result(self):
  if not self.tokens:raise ValueError('Empty diagnostic set')
  out={k:v/self.tokens for k,v in self.sums.items() if k not in ('overlap_logprob_gap_sum','overlap_token_count')}
  # Explicit project metric, not falsely labelled as upstream weighted advantage.
  count=self.sums['overlap_token_count'];out['overlap_logprob_gap_mean']=self.sums['overlap_logprob_gap_sum']/count if count else None
  out['overlap_token_count']=count;out['response_tokens']=self.tokens
  return out
