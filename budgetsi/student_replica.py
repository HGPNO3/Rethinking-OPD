"""A collection-only student replica in an owned spawned process.

The caller must close collection, sync the current adapter, wait for the hash
acknowledgement, then reopen collection. This module does not schedule or split
microbatches and never changes the existing seeded generation algorithm.
"""
import copy
import multiprocessing
import os
import pickle
import threading
import time


class ReplicaError(RuntimeError):
    pass


def _adapter_hash(state):
    from budgetsi.social_loop import state_hash
    return state_hash(state)


class _TorchBackend:
    def __init__(self, config):
        import torch
        from transformers import AutoTokenizer
        from peft import LoraConfig, get_peft_model
        from budgetsi.model_runtime import load_model, lora_targets, text_config
        if torch.cuda.device_count() != 1:
            raise ReplicaError('Replica must see exactly one assigned GPU')
        torch.cuda.set_device(0)
        torch.manual_seed(config['seed'])
        self.tokenizer = AutoTokenizer.from_pretrained(config['student'], local_files_only=True)
        model = load_model(config['student'], {'': 0})
        self.model = get_peft_model(model, LoraConfig(
            r=config['optimizer']['r'], lora_alpha=config['optimizer']['alpha'],
            lora_dropout=0.0, target_modules=lora_targets(model), task_type='CAUSAL_LM'))
        # Match the training student's generation setup; generate_batch controls
        # use_cache in its explicit GenerationConfig.
        self.model.config.use_cache = False
        text_config(self.model).use_cache = False
        self.model.requires_grad_(False)
        self.model.eval()
        self.max_context = config['context']
        self.temperature = config['temperature']

    def sync_adapter(self, state):
        from peft import set_peft_model_state_dict
        from budgetsi.gpu_smoke import adapter_state
        set_peft_model_state_dict(self.model, state)
        return adapter_state(self.model)

    def generate_batch(self, requests):
        from budgetsi.parallel_collect import generate_batch
        if any(r['temperature'] != self.temperature for r in requests):
            raise ReplicaError('Replica request temperature differs from bound configuration')
        return generate_batch(self.model, self.tokenizer, requests, self.max_context)


def _worker(connection, config, device, factory, hash_state):
    # Spawn, never fork a CUDA process. Set visibility before backend imports or
    # device initialization; physical device IDs are not inherited from parent.
    os.environ['CUDA_VISIBLE_DEVICES'] = device
    snapshot = None
    try:
        backend = factory(config)
        connection.send({'kind': 'ready', 'pid': os.getpid(), 'device': device})
        while True:
            message = connection.recv()
            kind = message['kind']
            if kind == 'close':
                connection.send({'kind': 'closed', 'id': message['id']})
                return
            try:
                if kind == 'sync':
                    snapshot = None
                    state = pickle.loads(message['state_bytes'])
                    if hash_state(state) != message['snapshot']:
                        raise ReplicaError('Transferred adapter hash mismatch')
                    actual = hash_state(backend.sync_adapter(state))
                    if actual != message['snapshot']:
                        raise ReplicaError('Loaded replica adapter hash mismatch')
                    snapshot = actual
                    result = {'snapshot': actual}
                elif kind == 'generate':
                    if snapshot is None or message['snapshot'] != snapshot:
                        raise ReplicaError('Generation requested with stale or unsynchronized adapter')
                    requests = message['requests']
                    if not requests:
                        raise ReplicaError('Empty replica batch')
                    if any(r.get('snapshot', snapshot) != snapshot for r in requests):
                        raise ReplicaError('Request snapshot differs from replica snapshot')
                    result = backend.generate_batch(requests)
                    if len(result) != len(requests):
                        raise ReplicaError('Replica response count mismatch')
                elif kind == 'score':
                    if snapshot is None or message['snapshot'] != snapshot:
                        raise ReplicaError('Scoring requested with stale or unsynchronized adapter')
                    request = message['request']
                    if request.get('snapshot') != snapshot:
                        raise ReplicaError('Score request snapshot differs from replica snapshot')
                    result = backend.score(request)
                    if len(result['raw_logprobs']) != len(request['target_ids']):
                        raise ReplicaError('Score response count mismatch')
                else:
                    raise ReplicaError('Unknown replica operation')
                connection.send({'kind': 'result', 'id': message['id'], 'snapshot': snapshot,
                                 'result': result})
            except Exception as error:
                # No silent fallback, retry, or skipped training sample.
                connection.send({'kind': 'error', 'id': message['id'],
                                 'error_type': type(error).__name__, 'error': str(error)})
    except EOFError:
        pass
    except Exception as error:
        try:
            connection.send({'kind': 'error', 'error_type': type(error).__name__, 'error': str(error)})
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        connection.close()


class StudentReplica:
    """One GPU's frozen collection snapshot; serial RPC transport is thread-safe.

    ``sync_adapter(adapter_state(student), snapshot)`` must complete before
    generation. ``generate_batch(requests, snapshot=...)`` can explicitly assert
    the caller's snapshot; omitting it uses the last acknowledged snapshot.
    ``_backend_factory`` and ``_state_hash`` are CPU fixture injection points.
    """
    def __init__(self, config, cuda_visible_devices, *, timeout=900, close_timeout=10,
                 _backend_factory=None, _state_hash=None):
        if (not isinstance(cuda_visible_devices, str) or
                not cuda_visible_devices.isdecimal() or
                timeout <= 0 or close_timeout <= 0):
            raise ValueError('Assign exactly one physical GPU and positive timeouts')
        self.config = copy.deepcopy(config)
        self.device = cuda_visible_devices
        self.timeout, self.close_timeout = timeout, close_timeout
        self._hash = _state_hash or _adapter_hash
        self._lock = threading.RLock()
        self._snapshot = None
        self._closed = False
        self._serial = 0
        self.last_receipt = None
        context = multiprocessing.get_context('spawn')
        self._connection, child_connection = context.Pipe()
        self._process = context.Process(target=_worker, args=(
            child_connection, self.config, self.device, _backend_factory or _TorchBackend, self._hash),
            name='budgetsi-student-replica', daemon=True)
        try:
            self._process.start()
            child_connection.close()
            ready = self._receive(timeout)
            if (ready.get('kind') != 'ready' or ready.get('pid') != self._process.pid or
                    ready.get('device') != self.device):
                raise ReplicaError('Replica initialization failed: ' + str(ready))
        except BaseException:
            child_connection.close()
            self._terminate_owned()
            self._connection.close()
            self._closed = True
            raise

    @property
    def snapshot(self):
        return self._snapshot

    @property
    def pid(self):
        return self._process.pid

    def _receive(self, timeout):
        if not self._connection.poll(timeout):
            self._terminate_owned()
            self._closed = True
            self._connection.close()
            raise ReplicaError('Replica timed out; owned child terminated')
        try:
            return self._connection.recv()
        except (EOFError, OSError) as error:
            raise ReplicaError('Replica exited or transport failed') from error

    def _rpc(self, kind, **payload):
        if self._closed or not self._process.is_alive():
            raise ReplicaError('Replica is closed or has exited')
        self._serial += 1
        try:
            self._connection.send({'kind': kind, 'id': self._serial, **payload})
        except (BrokenPipeError, OSError) as error:
            raise ReplicaError('Replica transport failed') from error
        response = self._receive(self.timeout)
        if response.get('id') != self._serial:
            raise ReplicaError('Replica response ID mismatch')
        if response.get('kind') == 'error':
            raise ReplicaError(f"Replica {response.get('error_type')}: {response.get('error')}")
        if response.get('kind') != 'result':
            raise ReplicaError('Unexpected replica response')
        return response

    def sync_adapter(self, state, snapshot):
        with self._lock:
            self._snapshot = None
            # Clone CPU adapter tensors so a caller cannot mutate the object
            # while its acknowledged version is being transferred.
            state = copy.deepcopy(state)
            if not isinstance(snapshot, str) or not snapshot or self._hash(state) != snapshot:
                raise ReplicaError('Source adapter hash mismatch')
            # A single trusted-process byte payload bypasses torch's per-tensor
            # ForkingPickler shared-FD reduction (hundreds of LoRA tensors).
            state_bytes = pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL)
            response = self._rpc('sync', state_bytes=state_bytes, snapshot=snapshot)
            if response.get('snapshot') != snapshot or response['result']['snapshot'] != snapshot:
                raise ReplicaError('Adapter acknowledgement mismatch')
            self._snapshot = snapshot
            return snapshot

    def generate_batch(self, requests, *, snapshot=None):
        with self._lock:
            expected = self._snapshot if snapshot is None else snapshot
            if expected is None or expected != self._snapshot:
                raise ReplicaError('No synchronized replica for requested snapshot')
            start = time.monotonic()
            response = self._rpc('generate', requests=requests, snapshot=expected)
            if response.get('snapshot') != expected:
                raise ReplicaError('Generation response snapshot mismatch')
            self.last_receipt = {'pid': self.pid, 'device': self.device, 'snapshot': expected,
                                 'batch_size': len(requests), 'seconds': time.monotonic()-start}
            return response['result']

    def _terminate_owned(self):
        # Process object identifies only the child created by this instance.
        if self._process.pid is None:
            return
        if self._process.is_alive():
            self._process.terminate()
        self._process.join(self.close_timeout)
        if self._process.is_alive():
            self._process.kill()
            self._process.join(self.close_timeout)

    def score(self, request, *, snapshot):
        with self._lock:
            if snapshot is None or snapshot != self._snapshot:
                raise ReplicaError('No synchronized replica for requested score snapshot')
            response = self._rpc('score', request={**request, 'snapshot': snapshot}, snapshot=snapshot)
            if response.get('snapshot') != snapshot:
                raise ReplicaError('Score response snapshot mismatch')
            return response['result']

    def close(self):
        # Even if a collection caller is hung inside an RPC, closing has a
        # bounded wait and can terminate only this object's spawned process.
        acquired = self._lock.acquire(timeout=self.close_timeout)
        if not acquired:
            self._closed = True
            self._terminate_owned()
            self._connection.close()
            self._snapshot = None
            return
        try:
            if self._closed:
                return
            self._closed = True
            try:
                if self._process.is_alive():
                    self._serial += 1
                    self._connection.send({'kind': 'close', 'id': self._serial})
                    if self._connection.poll(self.close_timeout):
                        self._connection.recv()
                    self._process.join(self.close_timeout)
            except (EOFError, OSError):
                pass
            finally:
                self._terminate_owned()
                self._connection.close()
                self._snapshot = None
        finally:
            self._lock.release()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
