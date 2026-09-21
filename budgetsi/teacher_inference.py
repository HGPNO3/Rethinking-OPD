"""Continuous teacher generation transport; exact OPD support stays in HF.

No model defaults, prompt rewriting, implicit retry or teacher-token training.
"""
import json
import math
import os
import signal
import subprocess
import time
import urllib.request
from pathlib import Path


def post(endpoint, payload):
    req = urllib.request.Request(endpoint, data=json.dumps(payload).encode(),
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=21600) as response:
        return json.load(response)


class VLLMTeacher:
    def __init__(self, config, output, port=18741):
        self.config, self.output, self.port = config, Path(output), port
        self.process = None
        self.endpoint = f'http://127.0.0.1:{port}/v1/completions'

    def start(self):
        cfg, runtime = self.config, self.config['teacher_inference']
        import sys
        command = [runtime.get('python', sys.executable), '-m', 'vllm.entrypoints.openai.api_server',
                   '--model', cfg['teacher'], '--served-model-name', 'teacher',
                   '--host', '127.0.0.1', '--port', str(self.port),
                   '--tensor-parallel-size', str(runtime['tp']),
                   '--pipeline-parallel-size', str(runtime['pp']),
                   '--dtype', 'bfloat16', '--max-model-len', str(cfg['context']),
                   '--max-num-seqs', str(runtime['max_num_seqs']),
                   '--max-num-batched-tokens', str(runtime['max_num_batched_tokens']),
                   '--gpu-memory-utilization', str(runtime['gpu_memory_utilization']),
                   '--generation-config', 'vllm', '--logprobs-mode', 'processed_logprobs',
                   '--enforce-eager', '--no-async-scheduling', '--enable-chunked-prefill', '--seed', str(cfg['seed'])]
        if runtime.get('language_model_only', False):
            command.append('--language-model-only')
        self.output.mkdir(parents=True, exist_ok=True)
        self.log = (self.output / f'vllm_{time.time_ns()}.log').open('w')
        env = os.environ.copy()
        if 'cuda_visible_devices' in runtime:
            env['CUDA_VISIBLE_DEVICES'] = runtime['cuda_visible_devices']
        self.process = subprocess.Popen(command, stdout=self.log, stderr=subprocess.STDOUT,
                                        start_new_session=True, env=env)
        deadline = time.monotonic() + 900
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(f'vLLM exited {self.process.returncode}; see {self.log.name}')
            try:
                with urllib.request.urlopen(f'http://127.0.0.1:{self.port}/health', timeout=1):
                    return
            except OSError:
                time.sleep(1)
        self.close()
        raise TimeoutError('vLLM startup timeout')

    def call(self, data):
        if data.get('model') != 'teacher':
            raise ValueError('Teacher model required')
        ctx = self.config['context']
        if data.get('operation') == 'score':
            prompt, target = data['prompt_ids'], data['target_ids']
            if not prompt or not target or len(prompt)+len(target)+1 > ctx:
                raise ValueError('Score context mismatch')
            result = post(self.endpoint, dict(model='teacher', prompt=prompt+target,
                          max_tokens=1, temperature=0., seed=0, prompt_logprobs=0,
                          return_token_ids=True))
            rows = result['choices'][0]['prompt_logprobs']
            if len(rows) != len(prompt)+len(target):
                raise ValueError('Prompt score alignment mismatch')
            values = [row[str(t)]['logprob'] for row,t in zip(rows[len(prompt):],target)]
            if len(values) != len(target) or any(not math.isfinite(x) or x > 0 for x in values):
                raise ValueError('Invalid prompt score')
            return dict(raw_logprobs=values, usage=dict(prompt_tokens=len(prompt),completion_tokens=len(target)))
        prompt = data['prompt']
        temperature = self.config.get('temperature')
        if not isinstance(temperature, (int, float)) or isinstance(temperature, bool) or not math.isfinite(temperature) or temperature <= 0:
            raise ValueError('Explicit positive generation temperature required')
        if (not prompt or data['max_tokens'] != ctx-len(prompt) or data['max_tokens'] < 1 or
            (data['temperature'],data['top_p'],data['top_k']) != (temperature,1,-1) or not data['stop_token_ids']):
            raise ValueError('Generation contract mismatch')
        result = post(self.endpoint, dict(model='teacher', prompt=prompt,
                      max_tokens=data['max_tokens'], temperature=temperature, top_p=1., top_k=-1,
                      repetition_penalty=1., presence_penalty=0., frequency_penalty=0.,
                      seed=data['seed'], stop_token_ids=data['stop_token_ids'], logprobs=0,
                      return_token_ids=True, skip_special_tokens=False))
        choice = result['choices'][0]
        ids, values = choice['token_ids'], choice['logprobs']['token_logprobs']
        if not ids or len(ids) != len(values) or any(x is None or not math.isfinite(x) or x > 0 for x in values):
            raise ValueError('Invalid generation probabilities')
        if choice['finish_reason'] == 'stop' and ids[-1] not in data['stop_token_ids']:
            raise ValueError('Unexpected stop without EOS')
        return dict(choices=[choice], usage=dict(prompt_tokens=len(prompt), completion_tokens=len(ids)))

    def close(self):
        if self.process:
            # Child exit and NVML context release need not occur simultaneously.
            owned=set()
            for line in subprocess.check_output(['ps','-eo','pid=,pgid='],text=True).splitlines():
                pid,pgid=map(int,line.split())
                if pgid==self.process.pid:owned.add(pid)
            if self.process.poll() is None:
                try:
                    with urllib.request.urlopen(f'http://127.0.0.1:{self.port}/metrics',timeout=3) as r:
                        (self.output/f'vllm_metrics_{time.time_ns()}.prom').write_bytes(r.read())
                except OSError:
                    pass
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait()
            deadline=time.monotonic()+30
            while owned:
                raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True)
                active={int(line.strip()) for line in raw.splitlines() if line.strip().isdigit()}
                if not active.intersection(owned):break
                if time.monotonic()>deadline:raise TimeoutError('Owned GPU contexts did not release')
                time.sleep(1)
            self.log.close()
            self.process = None
