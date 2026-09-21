"""Two frozen teacher backends resident on explicitly disjoint GPU sets.

Generation and exact HF scoring have independent bounded queues. Each bound
experiment advances its own phase; completion of one never unloads the shared
teacher required by another. Numerical scoring implementation is unchanged.
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from budgetsi.remote_teacher import contract, digest, serve
from budgetsi.teacher_inference import VLLMTeacher, post
from budgetsi.teacher_schedule import FairScheduler


class IndependentPhases:
    def __init__(self, bindings):
        self.states = {b: 'collecting' for b in bindings}
        self.rounds = {b: 0 for b in bindings}
        self.active = {b: 0 for b in bindings}
        self.error = None
        self.lock = threading.RLock()

    def check(self, binding, operation):
        if self.error:
            raise RuntimeError(self.error)
        expected = 'collecting' if operation == 'collect' else 'updating'
        if self.states.get(binding) != expected:
            raise RuntimeError('Request outside experiment phase')

    def enter(self, binding, operation):
        with self.lock:
            self.check(binding, operation)
            self.active[binding] += 1

    def leave(self, binding):
        with self.lock:
            self.active[binding] -= 1

    def arrive(self, binding, action, done=False):
        with self.lock:
            if self.error:
                raise RuntimeError(self.error)
            if type(done) is not bool or (done and action != 'updated'):
                raise ValueError('Invalid completion flag')
            operation = 'collect' if action == 'collected' else 'opd' if action == 'updated' else None
            if operation is None:
                raise ValueError('Invalid phase action')
            self.check(binding, operation)
            if self.active[binding]:
                raise RuntimeError('Cannot advance with outstanding requests')
            if action == 'collected':
                self.states[binding] = 'updating'
            else:
                self.rounds[binding] += 1
                self.states[binding] = 'finished' if done else 'collecting'
            return dict(round=self.rounds[binding], state=self.states[binding],
                        all_finished=all(s == 'finished' for s in self.states.values()))

    def abort(self, reason):
        with self.lock:
            self.error = str(reason)


class PersistentTeacher:
    def __init__(self, configs, bindings, config_paths, approval_paths, assets, out):
        self.configs = {digest(b): c for b, c in zip(bindings, configs)}
        self.bindings = {digest(b): b for b in bindings}
        if len(self.bindings) != len(bindings):
            raise ValueError('Duplicate teacher bindings')
        self.assets, self.out = assets, Path(out)
        self.runtime = configs[0]['teacher_inference']
        self.validate_runtime(self.runtime)
        self.backend = self.hf = self.hf_log = None
        self.closing = False
        self.stop_lock = threading.Lock()
        self.engine = SimpleNamespace(parallel=True)
        self.phases = IndependentPhases(self.bindings)
        # HF serializes its exact forward under its existing model execution lock.
        # A separate queue prevents these requests occupying generation capacity.
        self.generation = FairScheduler(self.bindings, self.execute, out/'generation_scheduler.jsonl', max_active=16, per_run=8)
        self.scoring = FairScheduler(self.bindings, self.execute, out/'scoring_scheduler.jsonl', max_active=2, per_run=1)
        self.config_paths, self.approval_paths = config_paths, approval_paths
        try:
            self.start_backends(configs[0])
        except BaseException:
            self.close()
            raise

    @staticmethod
    def validate_runtime(runtime):
        if runtime.get('backend') != 'vllm_persistent_hf_v1':
            raise ValueError('Persistent backend must be explicitly configured')
        for key in ('python', 'hf_python'):
            if not Path(runtime.get(key, '')).is_absolute():
                raise ValueError('Explicit absolute backend Python paths required')
        groups = []
        for key in ('cuda_visible_devices', 'hf_cuda_visible_devices'):
            value = runtime.get(key)
            if not isinstance(value, str) or not value:
                raise ValueError('Explicit GPU allocation required')
            ids = value.split(',')
            if any(not x.isdigit() for x in ids) or len(set(ids)) != len(ids):
                raise ValueError('Invalid GPU allocation')
            groups.append(set(ids))
        if groups[0] & groups[1] or len(groups[0]) != runtime['tp']*runtime['pp'] or len(groups[1]) != 2:
            raise ValueError('Teacher backends require disjoint GPU allocations')
        ports = [runtime.get('port', 18740), runtime.get('vllm_port', 18741), runtime.get('hf_port', 18742)]
        if len(set(ports)) != 3 or any(type(p) is not int or not 1024 <= p <= 65535 for p in ports):
            raise ValueError('Teacher ports must be distinct')

    def start_backends(self, config):
        child_out = self.out/'exact_hf'
        command = [self.runtime['hf_python'], '-u', '-m', 'budgetsi.remote_teacher',
                   '--port', str(self.runtime.get('hf_port', 18742)), '--output', str(child_out)]
        for c, a in zip(self.config_paths, self.approval_paths):
            command += ['--config', c, '--approval', a]
        self.hf_log = (self.out/'exact_hf.log').open('w')
        env = os.environ.copy(); env['CUDA_VISIBLE_DEVICES'] = self.runtime['hf_cuda_visible_devices']
        self.hf = subprocess.Popen(command, stdout=self.hf_log, stderr=subprocess.STDOUT,
                                   env=env, start_new_session=True)
        self.backend = VLLMTeacher(config, self.out, self.runtime.get('vllm_port', 18741))
        self.backend.start()
        deadline = time.monotonic()+900
        while not (child_out/'ready.json').exists():
            if self.hf.poll() is not None:
                raise RuntimeError('Exact HF teacher exited during startup')
            if time.monotonic() > deadline:
                raise TimeoutError('Exact HF teacher startup timeout')
            time.sleep(1)
        info = json.loads((child_out/'ready.json').read_text())
        if info.get('assets') != self.assets or {digest(b) for b in info.get('bindings', [])} != set(self.bindings):
            raise ValueError('Exact HF startup binding mismatch')
        self.check_alive()
        (self.out/'backend_launch.json').write_text(json.dumps(dict(
            generation_pid=self.backend.process.pid, exact_hf_pid=self.hf.pid,
            generation_gpus=self.runtime['cuda_visible_devices'], exact_hf_gpus=self.runtime['hf_cuda_visible_devices'],
            started_at=time.time()), indent=2))

    def check_alive(self):
        if self.closing or self.phases.error:
            raise RuntimeError(self.phases.error or 'Teacher closing')
        if self.backend is None or self.backend.process.poll() is not None or self.hf is None or self.hf.poll() is not None:
            self.phases.abort('Teacher backend exited')
            raise RuntimeError('Teacher backend exited')

    def hf_call(self, request):
        response = post(f"http://127.0.0.1:{self.runtime.get('hf_port', 18742)}/", request)
        if response.get('binding') != request['binding'] or response.get('request_sha256') != digest(request):
            raise ValueError('Exact teacher response mismatch')
        return response['result']

    def execute(self, request):
        self.check_alive()
        if request['operation'] == 'collect':
            return self.backend.call(request['payload'])
        if request['operation'] in {'opd', 'diagnostics', 'validate_raw'}:
            return self.hf_call(request)
        raise ValueError('Unsupported backend operation')

    def dispatch(self, request):
        key = digest(request['binding'])
        if self.bindings.get(key) != request['binding']:
            raise ValueError('Unregistered teacher binding')
        operation, data = request['operation'], request['payload']
        timing = {}
        if operation == 'abort':
            self.phases.abort('Experiment aborted: '+key)
            result = dict(aborted=True)
        elif operation == 'phase':
            self.check_alive()
            result = self.phases.arrive(key, data['action'], data.get('done', False))
            if result['all_finished']:
                # This transition is possible only after both runs drained requests.
                self.stop_backends()
        elif operation == 'status':
            if all(s == 'finished' for s in self.phases.states.values()):
                result = dict(frozen=True, assets=self.assets, backend='closed')
            else:
                self.check_alive()
                result = self.hf_call(request)
                result['backend'] = 'vllm_persistent_hf_v1'
            result['cohort_states'] = self.phases.states.copy()
        elif operation in {'collect', 'opd', 'diagnostics', 'validate_raw'}:
            self.phases.enter(key, operation)
            try:
                scheduler = self.generation if operation == 'collect' else self.scoring
                result, timing = scheduler.submit(key, request, with_timing=True)
            except BaseException as error:
                self.phases.abort(repr(error))
                raise
            finally:
                self.phases.leave(key)
        else:
            raise ValueError('Unknown operation')
        return dict(result=result, binding=request['binding'], request_sha256=digest(request), timing=timing)

    def stop_backends(self):
        with self.stop_lock:
            try:
                if self.backend is not None:
                    self.backend.close(); self.backend = None
            finally:
                # A generation cleanup error must never orphan the HF worker.
                if self.hf is not None:
                    try: os.killpg(self.hf.pid, signal.SIGTERM)
                    except ProcessLookupError: pass
                    try: self.hf.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(self.hf.pid, signal.SIGKILL); self.hf.wait()
                    self.hf = None
                if self.hf_log is not None:
                    self.hf_log.close(); self.hf_log = None

    def close(self):
        self.closing = True
        self.phases.abort('Teacher shutdown')
        try: self.stop_backends()
        finally:
            self.generation.close(); self.scoring.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', action='append', required=True)
    parser.add_argument('--approval', action='append', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    from budgetsi.formal_gate import check_launch
    from budgetsi.protocol_gate import sha256_file
    if len(args.config) != len(args.approval) or not 1 <= len(args.config) <= 3:
        raise ValueError('Explicit paired cohort required')
    configs = [json.loads(Path(p).read_text()) for p in args.config]
    gates = [check_launch(c, a, Path(__file__).resolve().parents[1]) for c,a in zip(args.config,args.approval)]
    cfg = configs[0]
    for c in configs:
        if any(c.get(k) != cfg.get(k) for k in ('teacher','teacher_inference','context','temperature','teacher_temperature','collection_parallel')):
            raise ValueError('Cohort inference contract mismatch')
    assets = {p: sha256_file(Path(p)) for p in cfg['files_sha256'] if Path(p).parent == Path(cfg['teacher'])}
    if any(c['files_sha256'].get(p) != h for c in configs for p,h in assets.items()):
        raise ValueError('Teacher asset mismatch')
    out = Path(args.output); out.mkdir(parents=True, exist_ok=False)
    (out/'gate.json').write_text(json.dumps(gates, indent=2))
    bindings = [contract(c, g['verified_bindings']['git_commit']) for c,g in zip(configs,gates)]
    def stop(*_): raise KeyboardInterrupt('User stop')
    signal.signal(signal.SIGTERM,stop)
    service = PersistentTeacher(configs,bindings,args.config,args.approval,assets,out)
    server = None
    try:
        server = serve(service, cfg['teacher_inference'].get('port',18740))
        (out/'ready.json').write_text(json.dumps(dict(bindings=bindings,assets=assets,port=server.server_port)))
        server.serve_forever()
    finally:
        if server is not None: server.server_close()
        service.close()

if __name__ == '__main__': main()
