"""Gate and verify frozen teacher before serving on a bound private interface."""
import argparse,hashlib,json,os,subprocess,time
from pathlib import Path
from protocol_gate import load_json,verify_approval
from train import atomic,digest
ROOT=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--approval',required=True);p.add_argument('--output',required=True);a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
cfg=load_json(ROOT/'config.json');gate=verify_approval(load_json(Path(a.approval)),ROOT/'manifest.json',ROOT/'config.json',ROOT);atomic(out/'gate.json',gate)
assert gate['ok'],gate['errors']
r=cfg['runtime'];folder=Path(r['teacher_model_path'])
for name,expected in cfg['teacher_files_sha256'].items():assert digest(folder/name)==expected,name
cmd=[r['teacher_vllm'],'serve',str(folder),'--served-model-name','teacher','--host',r['teacher_bind'],'--port','18014','--tensor-parallel-size',str(r['teacher_tp']),'--dtype','bfloat16','--max-model-len',str(cfg['context']),'--max-num-seqs','16','--max-num-batched-tokens','4096','--gpu-memory-utilization','0.88','--generation-config','vllm','--logprobs-mode','processed_logprobs','--enforce-eager','--no-async-scheduling','--seed',str(cfg['seed'])]
atomic(out/'launch.json',{'pid':os.getpid(),'time':time.time(),'command':cmd,'verified_teacher_files':cfg['teacher_files_sha256'],'git_commit':gate['verified_bindings']['git_commit']})
os.execv(cmd[0],cmd)
