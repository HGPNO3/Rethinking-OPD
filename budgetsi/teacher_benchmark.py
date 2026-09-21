"""Gated teacher-only benchmark on frozen real prompts; never a training run."""
import argparse
import concurrent.futures
import json
import statistics
import time
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--approval',required=True)
    p.add_argument('--requests',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    from budgetsi.formal_gate import check_launch
    from budgetsi.protocol_gate import sha256_file
    cfg=json.loads(Path(a.config).read_text());out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    gate=check_launch(a.config,a.approval,Path(__file__).resolve().parents[1])
    (out/'gate.json').write_text(json.dumps(gate,indent=2))
    spec=cfg['teacher_benchmark']
    assert spec['requests_sha256']==sha256_file(Path(a.requests))
    requests=json.loads(Path(a.requests).read_text());assert len(requests)==spec['count']
    for path,h in cfg['files_sha256'].items():
        if Path(path).parent==Path(cfg['teacher']):assert sha256_file(Path(path))==h,path
    from budgetsi.teacher_inference import VLLMTeacher
    backend=None;sampler=None
    load_started=time.monotonic()
    try:
        if spec['backend']=='vllm':
            backend=VLLMTeacher(cfg,out);backend.start();call=backend.call
        elif spec['backend']=='hf':
            from budgetsi.model_runtime import load_model
            from transformers import AutoTokenizer
            from budgetsi.parallel_collect import make_parallel_engine
            model=load_model(cfg['teacher'],'balanced').requires_grad_(False).eval()
            tok=AutoTokenizer.from_pretrained(cfg['teacher'],local_files_only=True)
            backend=make_parallel_engine(None,model,{'teacher':tok},cfg['context'],out,batch_size=4,wait_ms=10)
            backend.accepting=True;call=backend.call
        else:raise ValueError('Unknown benchmark backend')
        load_seconds=time.monotonic()-load_started
        call(requests[0]['data']) # same warmup request, excluded from measured throughput
        def run(item):
            start=time.monotonic();result=call(item['data']);end=time.monotonic()
            return dict(id=item['id'],seconds=end-start,started=start,finished=end,result=result)
        from budgetsi.gpu_sampling import GPUSampler
        sampler=GPUSampler();sampler.start()
        start=time.monotonic()
        with concurrent.futures.ThreadPoolExecutor(max_workers=spec['concurrency']) as pool:
            rows=list(pool.map(run,requests))
        elapsed=time.monotonic()-start
        sampler.close()
        (out/"gpu_samples.json").write_text(json.dumps(sampler.rows))
        tokens=sum(r['result']['usage']['completion_tokens'] for r in rows)
        generated=sum(r['result']['usage']['completion_tokens'] for r in rows if 'choices' in r['result'])
        differences=[]
        for item,row in zip(requests,rows):
            if 'expected_raw_logprobs' in item:
                actual=row['result']['raw_logprobs'];expected=item['expected_raw_logprobs']
                if len(actual)!=len(expected):raise ValueError('Score alignment differs from saved HF fixture')
                differences.extend(abs(x-y) for x,y in zip(actual,expected))
        numeric=dict(tokens=len(differences),mean_abs=sum(differences)/len(differences),max_abs=max(differences),mean_abs_tolerance=.1) if differences else None
        if numeric:
            (out/'cross_engine.json').write_text(json.dumps(numeric))
            assert numeric['mean_abs']<.1,'Prompt logprobs disagree with recorded HF reference'

        lat=sorted(r['seconds'] for r in rows)
        report=dict(status='passed',scope='teacher_only_real_prompt_replay_not_training',elapsed_seconds=elapsed,load_seconds=load_seconds,
                    useful_tokens=tokens,tokens_per_second=tokens/elapsed,generated_tokens_per_second=generated/elapsed,scored_tokens=tokens-generated,cross_engine=numeric,p50_seconds=statistics.median(lat),
                    p95_seconds=lat[min(len(lat)-1,int(.95*len(lat)))],
                    truncated=sum(r['result']['choices'][0]['finish_reason']!='stop' for r in rows if 'choices' in r['result']),
                    benchmark=spec,teacher_inference=cfg.get('teacher_inference'),rows=rows)
        (out/'result.json').write_text(json.dumps(report))
        print(json.dumps({k:v for k,v in report.items() if k!='rows'}),flush=True)
    finally:
        if sampler and sampler.thread.is_alive():sampler.close()
        if backend:backend.close()

if __name__=='__main__':main()
