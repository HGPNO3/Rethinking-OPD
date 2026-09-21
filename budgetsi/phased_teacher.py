"""Shared teacher: continuous generation, centrally coordinated exact HF scoring.

Each backend is a separate process, never two resident 27B GPU copies. No student
may individually switch it. Existing remote_teacher remains the exact reference.
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from types import SimpleNamespace
from budgetsi.remote_teacher import contract,digest,serve
from budgetsi.teacher_inference import VLLMTeacher,post
from budgetsi.teacher_schedule import FairScheduler,Cohort


class PhasedTeacher:
    def __init__(self,configs,bindings,config_paths,approval_paths,assets,out):
        self.configs={digest(b):c for b,c in zip(bindings,configs)}
        self.bindings={digest(b):b for b in bindings};self.assets=assets;self.out=out
        self.config_paths,self.approval_paths=config_paths,approval_paths
        self.backend=None;self.hf=None;self.hf_log=None;self.mode=None;self.closing=False
        self.engine=SimpleNamespace(parallel=True)
        self.scheduler=FairScheduler(self.bindings,self.execute,out/'scheduler.jsonl')
        self.cohort=Cohort(self.bindings,self.switch)
        try:self.switch('vllm')
        except BaseException:
            self.stop_backend();self.scheduler.close();raise

    def stop_backend(self):
        if self.backend:self.backend.close();self.backend=None
        if self.hf:
            try:os.killpg(self.hf.pid,signal.SIGTERM)
            except ProcessLookupError:pass
            try:self.hf.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(self.hf.pid,signal.SIGKILL);self.hf.wait()
            self.hf=None;self.hf_log.close()

    def switch(self,target):
        if not self.scheduler.idle():raise RuntimeError('Cannot switch with queued/inflight requests')
        start=time.time();self.stop_backend();self.mode=None
        if self.closing:raise RuntimeError('Teacher is closing')
        if target=='vllm':
            self.backend=VLLMTeacher(next(iter(self.configs.values())),self.out)
            self.backend.start()
        elif target=='hf':
            child_out=self.out/f'exact_hf_{time.time_ns()}'
            cmd=[sys.executable,'-u','-m','budgetsi.remote_teacher','--port','18742','--output',str(child_out)]
            for c,a in zip(self.config_paths,self.approval_paths):cmd+=['--config',c,'--approval',a]
            self.hf_log=(self.out/f'hf_{time.time_ns()}.log').open('w')
            self.hf=subprocess.Popen(cmd,stdout=self.hf_log,stderr=subprocess.STDOUT,start_new_session=True)
            deadline=time.monotonic()+900
            while not (child_out/'ready.json').exists():
                if self.hf.poll() is not None:raise RuntimeError('Exact HF teacher exited')
                if time.monotonic()>deadline:raise TimeoutError('Exact HF teacher startup timeout')
                time.sleep(1)
        elif target!='closed':raise ValueError('Unknown backend')
        if self.closing:
            self.stop_backend();raise RuntimeError("Teacher closed during backend load")
        self.mode=target
        with (self.out/'phases.jsonl').open('a') as f:
            f.write(json.dumps(dict(target=target,started_at=start,completed_at=time.time(),switch_seconds=time.time()-start))+'\n')

    def execute(self,request):
        if self.mode=='vllm' and request['operation']=='collect':
            return self.backend.call(request['payload'])
        if self.mode=='hf' and request['operation'] in {'opd','diagnostics','validate_raw'}:
            result=post('http://127.0.0.1:18742/',request)
            if result.get('request_sha256')!=digest(request) or result.get('binding')!=request['binding']:
                raise ValueError('Exact teacher response mismatch')
            return result['result']
        raise ValueError('Backend/operation mismatch')

    def dispatch(self,request):
        key=digest(request['binding'])
        if self.bindings.get(key)!=request['binding']:raise ValueError('Unregistered teacher binding')
        op,data=request['operation'],request['payload']
        timing={}
        if op=='status':
            if self.cohort.error:raise RuntimeError(self.cohort.error)
            if self.mode=='hf':
                result=post('http://127.0.0.1:18742/',request)['result']
            else:
                if self.mode=='vllm' and self.backend.process.poll() is not None:
                    self.cohort.abort('vLLM process exited');raise RuntimeError('vLLM process exited')
                result=dict(frozen=True,assets=self.assets)
            result.update(backend=self.mode,cohort_states=self.cohort.states.copy())
        elif op=='phase':result=self.cohort.arrive(key,data['action'],data.get('done',False))
        elif op=='abort':
            self.cohort.abort('Experiment aborted: '+key);result=dict(aborted=True)
        elif op in {'collect','opd','diagnostics','validate_raw'}:
            self.cohort.check(key,op)
            try:result,timing=self.scheduler.submit(key,request,with_timing=True)
            except BaseException as exc:self.cohort.abort(repr(exc));raise
        else:raise ValueError('Unknown operation')
        return dict(result=result,binding=request['binding'],request_sha256=digest(request),timing=timing)

    def close(self):
        self.closing=True;self.cohort.abort('Teacher shutdown');self.stop_backend();self.scheduler.close()


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',action='append',required=True)
    p.add_argument('--approval',action='append',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    from budgetsi.formal_gate import check_launch
    from budgetsi.protocol_gate import sha256_file
    if len(a.config)!=len(a.approval) or not 1<=len(a.config)<=3:raise ValueError('Explicit cohort required')
    configs=[json.loads(Path(c).read_text()) for c in a.config]
    gates=[check_launch(c,p,Path(__file__).resolve().parents[1]) for c,p in zip(a.config,a.approval)]
    cfg=configs[0]
    for c in configs:
        if any(c.get(k)!=cfg.get(k) for k in ('teacher','teacher_inference','context','temperature','teacher_temperature')):
            raise ValueError('Cohort inference contract mismatch')
    assets={p:sha256_file(Path(p)) for p in cfg['files_sha256'] if Path(p).parent==Path(cfg['teacher'])}
    if any(c['files_sha256'].get(p)!=h for c in configs for p,h in assets.items()):raise ValueError('Teacher asset mismatch')
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    (out/'gate.json').write_text(json.dumps(gates,indent=2))
    bindings=[contract(c,g['verified_bindings']['git_commit']) for c,g in zip(configs,gates)]
    service=PhasedTeacher(configs,bindings,a.config,a.approval,assets,out)
    server=serve(service,18740)
    def stop(*_):raise KeyboardInterrupt('User stop')
    signal.signal(signal.SIGTERM,stop)
    (out/'ready.json').write_text(json.dumps(dict(bindings=bindings,assets=assets,port=18740)))
    try:server.serve_forever()
    finally:server.server_close();service.close()

if __name__=='__main__':main()
