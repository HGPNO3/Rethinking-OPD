"""Bounded per-model inference workers; batched generation, isolated RNG streams.

Training is phase-separated via close_collection(). This is an opt-in backend;
GPU/model-family acceptance is required before a formal run.
"""
import json
import queue
import threading
import time
from concurrent.futures import Future

import torch
from transformers import GenerationConfig, LogitsProcessor, LogitsProcessorList


class RowSampler(LogitsProcessor):
    """Sample each row at its requested temperature using its own generator, then force that token.

    The recorded probability is BEFORE forcing. Ended/length-limited rows are
    ignored in receipts; a synthetic stop used to retire a row is never stored.
    """
    def __init__(self, requests, device):
        self.requests = requests
        self.generators = None
        self.ids = [[] for _ in requests]
        self.logps = [[] for _ in requests]
        self.done = [False] * len(requests)

    def __call__(self, input_ids, scores):
        if self.generators is None:
            self.generators = [torch.Generator(device=scores.device).manual_seed(r["seed"]) for r in self.requests]
        forced = torch.full_like(scores, -torch.inf)
        for i, request in enumerate(self.requests):
            if self.done[i]:
                token = request['stop_token_ids'][0]
            else:
                logp = (scores[i].float() / request["temperature"]).log_softmax(-1)
                token = int(torch.multinomial(logp.exp(), 1, generator=self.generators[i]).item())
                self.ids[i].append(token)
                self.logps[i].append(float(logp[token].item()))
                self.done[i] = token in request['stop_token_ids'] or len(self.ids[i]) == request['max_tokens']
            forced[i, token] = 0
        return forced


@torch.no_grad()
def generate_batch(model, tokenizer, requests, max_context):
    if not requests:
        raise ValueError('Empty batch')
    stops = requests[0]['stop_token_ids']
    for r in requests:
        if (not r['prompt'] or not stops or r['stop_token_ids'] != stops or
            r['max_tokens'] != max_context - len(r['prompt']) or r['max_tokens'] < 1 or
            (r['temperature'] not in (.7, 1.0) or (r['top_p'], r['top_k']) != (1, -1))):
            raise ValueError('Generation contract mismatch')
    device = model.get_input_embeddings().weight.device
    width = max(len(r['prompt']) for r in requests)
    pad = tokenizer.pad_token_id
    if pad is None:
        raise ValueError('Explicit pad token required')
    inputs = torch.full((len(requests), width), pad, dtype=torch.long, device=device)
    mask = torch.zeros_like(inputs)
    for i, r in enumerate(requests):
        inputs[i, -len(r['prompt']):] = torch.tensor(r['prompt'], device=device)
        mask[i, -len(r['prompt']):] = 1
    sampler = RowSampler(requests, device)
    model.eval()
    # Greedy selection AFTER our seeded sampler. Do not inherit nucleus/top-k,
    # repetition, forced-EOS, or other processors from model defaults.
    cfg = GenerationConfig(do_sample=False, num_beams=1, repetition_penalty=1.,
                           max_new_tokens=max(r['max_tokens'] for r in requests),
                           eos_token_id=stops, pad_token_id=pad, use_cache=True)
    previous_config = model.generation_config
    try:
        # One worker owns each model. Neutralize HF 4.x/5.x default merging
        # without relying on the removed use_model_defaults argument.
        model.generation_config = cfg
        model.generate(input_ids=inputs, attention_mask=mask, generation_config=cfg,
                       logits_processor=LogitsProcessorList([sampler]))
    finally:
        model.generation_config = previous_config
    results = []
    for r, ids, logps in zip(requests, sampler.ids, sampler.logps):
        results.append({'choices': [{'token_ids': ids,
                        'finish_reason': 'stop' if ids[-1] in stops else 'length',
                        'logprobs': {'token_logprobs': logps}}],
                        'usage': {'prompt_tokens': len(r['prompt']), 'completion_tokens': len(ids)}})
    return results


class BatchWorker:
    def __init__(self, execute, batch_size=4, wait_ms=10, shared_queue=None, trace=None):
        if type(batch_size) is not int or batch_size < 1 or wait_ms < 0:
            raise ValueError('Invalid batching limits')
        self.execute, self.batch_size, self.wait = execute, batch_size, wait_ms / 1000
        self.trace = trace
        self.queue = shared_queue if shared_queue is not None else queue.Queue(maxsize=256)
        self.execution_lock = threading.Lock()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def submit(self, data):
        future = Future()
        future.submitted_at = time.monotonic()
        self.queue.put((data, future))
        return future.result()

    def run(self):
        pending = None
        while True:
            first = pending if pending is not None else self.queue.get()
            pending = None
            if first is None:
                return
            batch = [first]
            key = lambda r: (r.get('operation', 'generate'), tuple(r.get('stop_token_ids', [])))
            deadline = time.monotonic() + self.wait
            # Score calls stay unbatched; each model has one owner. Different
            # model workers and remote RPCs can nevertheless overlap.
            if key(first[0])[0] == 'generate':
                while len(batch) < self.batch_size:
                    try:
                        item = self.queue.get(timeout=max(0, deadline-time.monotonic()))
                    except queue.Empty:
                        break
                    if item is None or key(item[0]) != key(first[0]):
                        pending = item
                        if item is None:
                            self.queue.put(None)
                        break
                    batch.append(item)
            try:
                with self.execution_lock:
                    dispatched = time.monotonic()
                    results = self.execute([r for r, _ in batch])
                    finished = time.monotonic()
                if self.trace is not None:
                    for request, future in batch:
                        self.trace(dict(kind=request.get("profile_kind"), model=request.get("model"),
                                        operation=request.get("operation", "generate"), batch_size=len(batch),
                                        enqueued=future.submitted_at, dispatched=dispatched, completed=finished,
                                        queue_seconds=dispatched-future.submitted_at, service_seconds=finished-dispatched))
                if len(results) != len(batch):
                    raise ValueError('Batch response count mismatch')
                for (_, future), result in zip(batch, results):
                    future.set_result(result)
            except Exception as error:
                for _, future in batch:
                    future.set_exception(error)

    def close(self):
        self.queue.put(None)
        self.thread.join()


class WorkerPool:
    """Two independent executors pull complete microbatches from one bounded queue."""
    def __init__(self, executors, batch_size, wait_ms, trace=None):
        shared = queue.Queue(maxsize=256)
        self.workers = [BatchWorker(fn, batch_size, wait_ms, shared, trace=trace) for fn in executors]
        self.queue = shared

    def submit(self, data):
        return self.workers[0].submit(data)

    def close(self):
        for _ in self.workers:
            self.queue.put(None)
        for worker in self.workers:
            worker.thread.join()


def make_parallel_engine(student, teacher, tokenizers, max_context, out, *, batch_size=4, wait_ms=10, temperature=0.7, student_replica=None, student_score_replica=None):
    from budgetsi.social_loop import Engine

    class ParallelEngine(Engine):
        parallel = True

        def __init__(self):
            super().__init__(student, teacher, tokenizers, max_context, out, temperature=temperature)
            self.condition = threading.Condition()
            self.active = 0
            self.closing = False
            self.receipt_lock = threading.Lock()
            self.local_model_lock = threading.RLock()
            self.model_admission = threading.Condition()
            self.pending_scores = 0
            # Match the teacher scheduler per-run capacity; queue locally, never retry sampling.
            self.teacher_rpc_slots = threading.BoundedSemaphore(8)
            self.workers = {name: BatchWorker(lambda rs, name=name: self.execute(name, rs), batch_size, wait_ms, trace=self.trace)
                            for name, model in self.models.items()
                            if model is not None and not hasattr(model, 'score_support')}
            if student_replica is not None:
                self.workers['student'].close()
                self.workers['student'] = WorkerPool([
                    lambda rs: self.execute('student', rs),
                    lambda rs: self.execute('student', rs, replica=student_replica)], batch_size, wait_ms, trace=self.trace)
            # Score tasks never occupy replica generation workers. Scores retain
            # the original unbatched primary-model implementation and lock.
            self.score_workers = {name: BatchWorker(lambda rs, name=name: self.execute(name, rs),
                                                   1, 0, trace=self.trace) for name in self.workers}

        def trace(self, row):
            with self.receipt_lock:
                with (self.out / 'request_profile.jsonl').open('a') as handle:
                    handle.write(json.dumps(dict(snapshot=self.snapshot, logged_at=time.time(), **row))+'\n')

        def execute(self, name, requests, replica=None):
            start = time.monotonic()
            lock_wait = 0.0
            if requests[0].get('operation', 'generate') == 'generate':
                if any(r['temperature'] != self.temperature for r in requests):
                    raise ValueError('Generation temperature differs from bound configuration')
                if replica is not None:
                    if replica.snapshot != self.snapshot:
                        raise RuntimeError('Student replica snapshot is stale')
                    results = replica.generate_batch(requests)
                else:
                    lock_start = time.monotonic()
                    # Non-preemptive priority at generation microbatch boundaries.
                    # Drain already admitted cheap score requests before starting
                    # another long primary generation; replica generation is free.
                    with self.model_admission:
                        self.model_admission.wait_for(lambda: self.pending_scores == 0)
                        self.local_model_lock.acquire()
                    try:
                        lock_wait = time.monotonic()-lock_start
                        results = generate_batch(self.models[name], self.tokenizers[name], requests, self.max_context)
                    finally:
                        self.local_model_lock.release()
            else:
                if name == 'student' and student_score_replica is not None:
                    result = student_score_replica.score(requests[0], snapshot=self.snapshot)
                    end = time.monotonic()
                    with self.receipt_lock:
                        receipt = dict(model=name, operation='score', snapshot=self.snapshot,
                                       seconds=end-start, backend='hf_frozen_score_replica_v1',
                                       execution_device=student_score_replica.device, **result['usage'])
                        self.requests.append(receipt)
                        with (self.out / 'engine_requests.jsonl').open('a') as handle:
                            handle.write(json.dumps(receipt)+'\n')
                    self.trace(dict(kind=requests[0].get('profile_kind'), model=name,
                                    operation='score_replica', compute_and_lease_seconds=end-start,
                                    batch_size=1, execution_device=student_score_replica.device))
                    return [result]
                # Base score path has no RNG mutation. Protect its old log writer.
                lock_start = time.monotonic()
                with self.local_model_lock:
                    lock_acquired = time.monotonic()
                    with self.receipt_lock:
                        results = [super(ParallelEngine, self).call(requests[0])]
                self.trace(dict(kind=requests[0].get('profile_kind'), model=name, operation='score_lock',
                                lock_seconds=lock_acquired-lock_start, compute_seconds=time.monotonic()-lock_acquired))
                return results
            end = time.monotonic()
            with self.receipt_lock:
                with (self.out / 'engine_requests.jsonl').open('a') as handle:
                    for request, result in zip(requests, results):
                        receipt = dict(experiment_binding=request.get("experiment_binding"),model=name, operation='generate', snapshot=self.snapshot,
                                       seconds=end-start, start_monotonic=start, end_monotonic=end,
                                       batch_size=len(requests), lock_seconds=lock_wait, backend='hf_seeded_batch_v1', execution_device='replica' if replica is not None else 'primary', **result['usage'])
                        self.requests.append(receipt)
                        handle.write(json.dumps(receipt)+'\n')
            return results

        def call(self, data):
            with self.condition:
                if not self.accepting or self.closing:
                    raise RuntimeError('Collection closed during policy update')
                self.active += 1
            try:
                name = data['model']
                if name in self.workers:
                    is_score = data.get('operation') == 'score'
                    worker = self.score_workers[name] if is_score else self.workers[name]
                    needs_primary = is_score and not (name == 'student' and student_score_replica is not None)
                    if needs_primary:
                        with self.model_admission:
                            self.pending_scores += 1
                    try:
                        return worker.submit(data)
                    finally:
                        if needs_primary:
                            with self.model_admission:
                                self.pending_scores -= 1
                                self.model_admission.notify_all()
                # Remote requests must not hold a GPU or global lock while waiting.
                start = time.monotonic()
                with self.teacher_rpc_slots:
                    acquired = time.monotonic()
                    result = self.models[name].call('collect', data)
                self.trace(dict(model=name, kind=data.get('profile_kind'), operation=data.get('operation','generate'),
                                queue_seconds=acquired-start, service_seconds=time.monotonic()-acquired, batch_size=1))
                with self.receipt_lock:
                    receipt = dict(model=name, operation=data.get('operation', 'generate'),
                                   snapshot=self.snapshot, start_monotonic=start,
                                   end_monotonic=time.monotonic(), **result['usage'])
                    self.requests.append(receipt)
                    with (self.out / 'engine_requests.jsonl').open('a') as handle:
                        handle.write(json.dumps(receipt)+'\n')
                return result
            finally:
                with self.condition:
                    self.active -= 1
                    self.condition.notify_all()

        def close_collection(self):
            with self.condition:
                self.closing = True
                self.condition.wait_for(lambda: self.active == 0)
                self.accepting = False
                self.closing = False

        def close(self):
            self.close_collection()
            for worker in list(self.workers.values()) + list(self.score_workers.values()):
                worker.close()

    return ParallelEngine()
