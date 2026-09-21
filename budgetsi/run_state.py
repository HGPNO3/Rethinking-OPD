"""CPU-only quota, deterministic scene ordering, durable checkpoint ledger."""
import copy
import hashlib
import json
import os
import random
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def durable_json(path, value):
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    with tmp.open('w') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def choose_quota(records, used, target):
    ids = [r['id'] for r in records]
    if len(ids) != len(set(ids)) or set(ids) & set(used):
        raise ValueError('Duplicate or previously used node')
    if not 0 <= len(used) <= target:
        raise ValueError('Invalid used-node quota')
    ordered = sorted(records, key=lambda r: r['id'])
    n = target - len(used)
    return ordered[:n], [r['id'] for r in ordered[n:]]


def batch_scenes(pool, index, n, seed):
    if len(pool) != 100 or len({s['id'] for s in pool}) != 100:
        raise ValueError('Expected100 unique source scenes')
    order = sorted(pool, key=lambda s: hashlib.sha256(('split20260915' + s['id']).encode()).hexdigest())[:80]
    random.Random(seed).shuffle(order)
    ans = []
    for j in range(n):
        absolute = index * n + j
        scene = copy.deepcopy(order[absolute % len(order)])
        scene['id'] = f'formal:b{index:04d}:j{j:02d}:' + scene['id']
        scene['seed'] = seed + absolute
        ans.append(scene)
    return {'schema_version': 'fresh_online_batch_v1', 'scenes': ans}


class Ledger:
    def __init__(self, out, config, commit, resume):
        self.out, self.target = Path(out), config['target_nodes']
        self.binding = dict(config_sha256=hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(), git_commit=commit)
        self.path = self.out / 'state.json'
        if resume:
            self.state = json.loads(self.path.read_text())
            if self.state['binding'] != self.binding:
                raise ValueError('Resume code/config binding changed')
        else:
            if self.path.exists():
                raise ValueError('Existing run requires --resume')
            self.state = dict(binding=self.binding, next_batch=0, updates=[], batches=[])
            durable_json(self.path, self.state)
        self.validate(self.state)

    @property
    def used(self):
        return {node for r in self.state['updates'] for node in r['nodes']}

    def validate(self, state, verify_files=True):
        used, previous = set(), None
        if state['next_batch'] != len(state['batches']):
            raise ValueError('Batch progress mismatch')
        if [b['batch'] for b in state['batches']] != list(range(state['next_batch'])):
            raise ValueError('Missing/out-of-order batch')
        last_batch = -1
        for step, r in enumerate(state['updates'], 1):
            if r['status'] != 'passed' or r['optimizer_step'] != step or not last_batch < r['batch'] < state['next_batch']:
                raise ValueError('Invalid committed update')
            last_batch = r['batch']
            if previous is not None and r['snapshot_before'] != previous:
                raise ValueError('Broken snapshot chain')
            if not r['nodes'] or len(r['nodes']) != len(set(r['nodes'])) or used.intersection(r['nodes']):
                raise ValueError('Duplicate/empty committed nodes')
            used.update(r['nodes'])
            previous = r['snapshot_after']
            folder = (self.out / r['checkpoint']).resolve()
            if not folder.is_relative_to(self.out.resolve()):
                raise ValueError('Checkpoint outside run')
            if verify_files:
                for name, field in [('adapter/adapter_model.safetensors', 'adapter_file_sha256'), ('optimizer.pt', 'optimizer_file_sha256')]:
                    if digest(folder / name) != r[field]:
                        raise ValueError('Checkpoint hash mismatch')
                if json.loads((folder / 'result.json').read_text()) != r:
                    raise ValueError('Checkpoint receipt mismatch')
        if len(used) > self.target:
            raise ValueError('Committed quota overflow')

    def commit(self, batch_index, batch, receipt):
        if batch_index != self.state['next_batch']:
            raise ValueError('Out-of-order commit')
        candidate = copy.deepcopy(self.state)
        candidate['next_batch'] += 1
        candidate['batches'].append(batch)
        if receipt:
            folder = self.out / receipt['checkpoint']
            # Flush checkpoint content before atomically committing the ledger.
            for path in folder.rglob('*'):
                if path.is_file():
                    with path.open('rb') as f:
                        os.fsync(f.fileno())
            for path in [folder / 'adapter', folder, folder.parent, self.out]:
                fd = os.open(path, os.O_RDONLY)
                try:
                    os.fsync(fd)
                finally:
                    os.close(fd)
            candidate['updates'].append(receipt)
        self.validate(candidate, verify_files=False)
        durable_json(self.path, candidate)
        self.state = candidate
