"""CPU-only spawn/Pipe fixtures; no model load or CUDA use."""
import hashlib
import json
import os
import time
import unittest
from concurrent.futures import ThreadPoolExecutor

from budgetsi.student_replica import StudentReplica, ReplicaError


def fixture_hash(state):
    return hashlib.sha256(json.dumps(state, sort_keys=True).encode()).hexdigest()


class FixtureBackend:
    def __init__(self, config):
        if config.get('fail_load'):
            raise ValueError('fixture load failure')
        self.config = config
        self.state = None

    def sync_adapter(self, state):
        self.state = dict(state)
        if self.config.get('corrupt_load'):
            self.state['value'] += 1
        return self.state

    def generate_batch(self, requests):
        if requests[0].get('error'):
            raise ValueError('fixture generation failure')
        if requests[0].get('sleep'):
            time.sleep(requests[0]['sleep'])
        return [{'seed': r['seed'], 'temperature': r['temperature'],
                 'prompt': r['prompt'], 'value': self.state['value'],
                 'device': os.environ['CUDA_VISIBLE_DEVICES'], 'pid': os.getpid()}
                for r in requests]

    def score(self, request):
        if request.get('sleep'):
            time.sleep(request['sleep'])
        return {'raw_logprobs': [float(self.state['value'])] * len(request['target_ids']),
                'usage': {'prompt_tokens': len(request['prompt_ids']), 'completion_tokens': len(request['target_ids'])}}


def replica(config=None, **kwargs):
    return StudentReplica(config or {}, '6', _backend_factory=FixtureBackend,
                          _state_hash=fixture_hash, **kwargs)


def request(seed=42):
    return {'prompt': [11, 22], 'seed': seed, 'temperature': 1.0}


class StudentReplicaTests(unittest.TestCase):
    def test_generation_requires_sync_and_preserves_inputs(self):
        parent_device = os.environ.get('CUDA_VISIBLE_DEVICES')
        with replica() as child:
            with self.assertRaisesRegex(ReplicaError, 'synchronized'):
                child.generate_batch([request()])
            state = {'value': 3}; snapshot = fixture_hash(state)
            self.assertEqual(child.sync_adapter(state, snapshot), snapshot)
            state['value'] = 999
            result = child.generate_batch([request()], snapshot=snapshot)[0]
            self.assertEqual(result['value'], 3)
            self.assertEqual(result['seed'], 42)
            self.assertEqual(result['temperature'], 1.0)
            self.assertEqual(result['prompt'], [11, 22])
            self.assertEqual(result['device'], '6')
            self.assertNotEqual(result['pid'], os.getpid())
            self.assertEqual(child.last_receipt['snapshot'], snapshot)
        self.assertFalse(child._process.is_alive())
        self.assertEqual(os.environ.get('CUDA_VISIBLE_DEVICES'), parent_device)

    def test_adapter_crosses_pipe_as_one_byte_payload(self):
        with replica() as child:
            original = child._rpc
            observed = []
            def inspect(kind, **payload):
                if kind == 'sync':
                    self.assertNotIn('state', payload)
                    self.assertIsInstance(payload['state_bytes'], bytes)
                    observed.append(len(payload['state_bytes']))
                return original(kind, **payload)
            child._rpc = inspect
            state = {'value': 5}
            child.sync_adapter(state, fixture_hash(state))
            self.assertEqual(len(observed), 1)
            self.assertGreater(observed[0], 0)

    def test_source_and_loaded_adapter_hashes_both_checked(self):
        with replica() as child:
            with self.assertRaisesRegex(ReplicaError, 'Source adapter hash'):
                child.sync_adapter({'value': 1}, 'wrong')
        with replica({'corrupt_load': True}) as child:
            state = {'value': 1}
            with self.assertRaisesRegex(ReplicaError, 'Loaded replica adapter hash'):
                child.sync_adapter(state, fixture_hash(state))
            self.assertIsNone(child.snapshot)

    def test_sync_replaces_snapshot_and_rejects_stale_requests(self):
        with replica() as child:
            a, b = {'value': 1}, {'value': 2}
            child.sync_adapter(a, fixture_hash(a))
            child.sync_adapter(b, fixture_hash(b))
            with self.assertRaisesRegex(ReplicaError, 'synchronized'):
                child.generate_batch([request()], snapshot=fixture_hash(a))
            stale = {**request(), 'snapshot': fixture_hash(a)}
            with self.assertRaisesRegex(ReplicaError, 'Request snapshot'):
                child.generate_batch([stale])
            self.assertEqual(child.generate_batch([request()])[0]['value'], 2)

    def test_parallel_callers_keep_pipe_responses_bound(self):
        with replica() as child:
            state = {'value': 7}; child.sync_adapter(state, fixture_hash(state))
            with ThreadPoolExecutor(4) as pool:
                outputs = list(pool.map(lambda seed: child.generate_batch([request(seed)])[0], range(12)))
            self.assertEqual([r['seed'] for r in outputs], list(range(12)))
            self.assertTrue(all(r['value'] == 7 for r in outputs))

    def test_child_errors_explicit_and_timeout_cleans_owned_child(self):
        with self.assertRaisesRegex(ReplicaError, 'fixture load failure'):
            replica({'fail_load': True})
        with replica(timeout=2, close_timeout=.2) as child:
            state = {'value': 1}; child.sync_adapter(state, fixture_hash(state))
            with self.assertRaisesRegex(ReplicaError, 'fixture generation failure'):
                child.generate_batch([{**request(), 'error': True}])
            with self.assertRaisesRegex(ReplicaError, 'timed out'):
                child.generate_batch([{**request(), 'sleep': 5}])
            self.assertFalse(child._process.is_alive())
        child.close()

    def test_close_bounds_wait_for_busy_caller(self):
        child = replica(timeout=20, close_timeout=.2)
        state = {'value': 1}; child.sync_adapter(state, fixture_hash(state))
        with ThreadPoolExecutor(1) as pool:
            future = pool.submit(child.generate_batch, [{**request(), 'sleep': 5}])
            time.sleep(.1)
            start = time.monotonic(); child.close()
            self.assertLess(time.monotonic()-start, 2)
            self.assertFalse(child._process.is_alive())
            with self.assertRaises(ReplicaError):
                future.result()

    def test_one_physical_device_required(self):
        for device in ('', '6,7', '-1', 6):
            with self.assertRaises(ValueError):
                StudentReplica({}, device, _backend_factory=FixtureBackend, _state_hash=fixture_hash)

    def test_score_requires_current_snapshot_and_reloads_after_update(self):
        data={'model':'student','operation':'score','prompt_ids':[1,2],'target_ids':[3,4,5]}
        with replica() as child:
            with self.assertRaisesRegex(ReplicaError,'synchronized'):
                child.score(data,snapshot='absent')
            for value in (2,7):
                state={'value':value};snapshot=fixture_hash(state)
                child.sync_adapter(state,snapshot)
                self.assertEqual(child.score(data,snapshot=snapshot)['raw_logprobs'],[float(value)]*3)
            with self.assertRaisesRegex(ReplicaError,'synchronized'):
                child.score(data,snapshot=fixture_hash({'value':2}))


if __name__ == '__main__':
    unittest.main()
