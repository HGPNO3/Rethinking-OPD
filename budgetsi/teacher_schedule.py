"""Bounded fair admission plus fail-closed cohort phase coordination (no GPU)."""
import collections
import concurrent.futures
import json
import threading
import time
import uuid


class FairScheduler:
    def __init__(self, bindings, execute, output, max_active=24, per_run=8, max_queued=128):
        self.execute, self.output = execute, output
        self.bindings = list(bindings)
        self.queues = {b: collections.deque() for b in bindings}
        self.active = collections.Counter()
        self.score_streak = collections.Counter()
        self.max_active, self.per_run, self.max_queued = max_active, per_run, max_queued
        self.cv = threading.Condition(); self.cursor = 0; self.closed = False
        self.pool = concurrent.futures.ThreadPoolExecutor(max_workers=max_active)
        self.thread = threading.Thread(target=self.run, daemon=True); self.thread.start()

    def submit(self, binding, request, with_timing=False):
        received = time.time(); future = concurrent.futures.Future()
        with self.cv:
            if self.closed or binding not in self.queues:
                raise RuntimeError('Scheduler closed or unknown experiment')
            if len(self.queues[binding]) >= self.max_queued:
                raise RuntimeError('Bounded teacher queue full; no implicit retry')
            item = dict(id=request.get("request_id", uuid.uuid4().hex), binding=binding, request=request, future=future,
                        received_at=received, enqueued_at=time.time())
            self.queues[binding].append(item); self.cv.notify_all()
        result, timing = future.result(timeout=21600)
        return (result, timing) if with_timing else result

    def idle(self):
        with self.cv:
            return not any(self.queues.values()) and not sum(self.active.values())

    def pop_fair(self,binding):
        queue=self.queues[binding]
        is_score=lambda item: isinstance(item['request'].get('payload'),dict) and item['request']['payload'].get('operation')=='score'
        scores=[i for i,item in enumerate(queue) if is_score(item)]
        generation=[i for i,item in enumerate(queue) if not is_score(item)]
        if scores and (self.score_streak[binding]<2 or not generation):
            index=scores[0];self.score_streak[binding]+=1
        else:
            index=generation[0] if generation else 0;self.score_streak[binding]=0
        item=queue[index];del queue[index];return item

    def run(self):
        while True:
            with self.cv:
                item = None
                while item is None:
                    if self.closed and not any(self.queues.values()):return
                    if sum(self.active.values()) < self.max_active:
                        for _ in self.bindings:
                            b=self.bindings[self.cursor];self.cursor=(self.cursor+1)%len(self.bindings)
                            if self.queues[b] and self.active[b]<self.per_run:
                                # At most two queued short scores before a waiting generation.
                                item=self.pop_fair(b);self.active[b]+=1;break
                    if item is None:self.cv.wait()
                item['dispatched_at']=time.time()
            self.pool.submit(self.finish,item)

    def finish(self,item):
        started=time.monotonic();error=None;result=None
        try:result=self.execute(item['request'])
        except BaseException as exc:error=exc
        row={k:item[k] for k in ('id','binding','received_at','enqueued_at','dispatched_at')}
        row.update(completed_at=time.time(),backend_roundtrip_seconds=time.monotonic()-started,
                   admission_queue_seconds=item['dispatched_at']-item['enqueued_at'],
                   operation=item['request']['operation'],status='failed' if error else 'completed')
        # backend_roundtrip includes vLLM internal scheduling; never label it GPU execution.
        with self.cv:
            try:
                with self.output.open('a') as f:f.write(json.dumps(row)+'\n')
            except BaseException as exc:error=error or exc
            self.active[item['binding']]-=1;self.cv.notify_all()
        if error:item['future'].set_exception(error)
        else:item['future'].set_result((result,row))

    def close(self):
        with self.cv:self.closed=True;self.cv.notify_all()
        self.thread.join();self.pool.shutdown(wait=True)


class Cohort:
    """No switch until every live experiment has drained its current phase."""
    def __init__(self,bindings,switch,timeout=21600):
        self.states={b:'collecting' for b in bindings};self.round=0
        self.switch=switch;self.timeout=timeout;self.error=None;self.cv=threading.Condition()

    def check(self,binding,operation):
        with self.cv:
            if self.error:raise RuntimeError(self.error)
            expected='collecting' if operation=='collect' else 'updating'
            if self.states.get(binding)!=expected:raise RuntimeError('Request outside cohort phase')

    def abort(self,reason):
        with self.cv:self.error=str(reason);self.cv.notify_all()

    def arrive(self,binding,action,done=False):
        deadline=time.monotonic()+self.timeout
        with self.cv:
            if self.error:raise RuntimeError(self.error)
            if binding not in self.states:raise ValueError('Unknown cohort member')
            phase='collecting' if action=='collected' else 'updating' if action=='updated' else None
            if phase is None or self.states[binding]!=phase:raise ValueError('Invalid cohort transition')
            before=self.round
            self.states[binding]='finished' if done and action=='updated' else ('collected' if action=='collected' else 'updated')
            waiting='collected' if action=='collected' else 'updated'
            if all(s in {waiting,'finished'} for s in self.states.values()):
                target='hf' if action=='collected' else 'vllm'
                if all(s=='finished' for s in self.states.values()):target='closed'
                # Slow model loads must not prevent another member from reporting
                # failure or the operator from waking barrier waiters.
                self.cv.release()
                try:
                    self.switch(target)
                except BaseException as exc:
                    self.cv.acquire();self.error=repr(exc);self.cv.notify_all();raise
                else:
                    self.cv.acquire()
                if self.error:self.cv.notify_all();raise RuntimeError(self.error)
                for b,s in self.states.items():
                    if s!='finished':self.states[b]='updating' if action=='collected' else 'collecting'
                if action=='updated':self.round+=1
                self.cv.notify_all()
            while (self.states[binding]==waiting or (self.states[binding]=='finished' and self.round==before)) and not self.error:
                remaining=deadline-time.monotonic()
                if remaining<=0:self.error='Cohort barrier timed out';self.cv.notify_all();break
                self.cv.wait(remaining)
            if self.error:raise RuntimeError(self.error)
            return dict(round=self.round,state=self.states[binding])
