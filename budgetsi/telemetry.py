"""Formal-run W&B lifecycle with a durable, scalar-only recovery ledger."""
import hashlib
import json
import math
import os
from pathlib import Path


class Telemetry:
    def __init__(self, root, mode='offline', entity=None, project='budgetsi-opd', config=None):
        if mode not in ('offline', 'online', 'disabled'):
            raise ValueError('Invalid wandb mode')
        entity = entity or os.environ.get('WANDB_ENTITY')
        if mode == 'online' and not entity:
            raise ValueError('Online logging requires an explicit WANDB_ENTITY or entity')
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.mode, self.run, self.seen = mode, None, {}
        self.path = self.root / 'metrics.jsonl'
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                record = json.loads(line)
                key = record['event_id']
                if key in self.seen and self.seen[key] != record:
                    raise ValueError('Conflicting ledger event')
                self.seen[key] = record
        run_id = hashlib.sha256(str(self.root.resolve()).encode()).hexdigest()[:16]
        kw = dict(entity=entity, project=project, config=config or {})
        if config and isinstance(config.get('variant'), dict):
            kw.update(name=config['variant']['name']+'-'+run_id[:6],
                      group='qwen35-27b-to-4b-opd')
        self.status = dict(mode=mode, entity=entity, project=project,
                           cloud_upload_enabled=mode == 'online', status='starting')
        try:
            if mode != 'disabled':
                import wandb
                # Offline resume is unsupported: each restart creates a complete export
                # from the durable ledger. Upload only the latest complete export.
                self.run = wandb.init(
                    **kw, mode=mode, dir=str(self.root), save_code=False,
                    settings=wandb.Settings(disable_git=True, disable_job_creation=True, console='off'),
                    **({'id':run_id, 'resume':'allow'} if mode == 'online' else {}))
                self.run.define_metric('optimizer_step')
                self.run.define_metric('*', step_metric='optimizer_step')
                self.status.update(run_id=self.run.id, directory=str(Path(self.run.dir).parent),
                                   url=self.run.url if mode == 'online' else None)
                # Online resume restores the next W&B history step. Offline starts at 0.
                next_step = self.run.step if mode == 'online' else 0
                for index, record in enumerate(self.seen.values()):
                    if index >= next_step:
                        self.run.log(record['metrics'], step=index)
        except Exception as exc:
            self._sdk_failed(exc)
        self.status['status'] = 'running'
        self._status()

    def _sdk_failed(self, exc):
        self.status.update(sdk_error_type=type(exc).__name__, cloud_upload_enabled=False, recovery="durable_scalar_ledger")
        self.run = None

    def _status(self):
        path = self.root / 'wandb_status.json'
        temp = path.with_suffix('.tmp')
        temp.write_text(json.dumps(self.status, indent=2))
        temp.replace(path)

    def log(self, event_id, values):
        if any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in values.values()):
            raise ValueError('Only finite numeric telemetry permitted')
        record = dict(event_id=event_id, metrics=values)
        if event_id in self.seen:
            if self.seen[event_id] != record:
                raise ValueError('Conflicting replayed telemetry')
            return
        with self.path.open('a') as stream:
            stream.write(json.dumps(record)+'\n')
            stream.flush()
            os.fsync(stream.fileno())
        index = len(self.seen)
        self.seen[event_id] = record
        if self.run:
            try:
                self.run.log(values, step=index)
            except Exception as exc:
                self._sdk_failed(exc)
                self._status()

    def finish(self, exit_code=0):
        if self.run:
            try:
                self.run.finish(exit_code=exit_code)
            except Exception as exc:
                self._sdk_failed(exc)
        self.status.update(status='finished' if exit_code == 0 else 'failed',
                           events=len(self.seen), exit_code=exit_code)
        self._status()
