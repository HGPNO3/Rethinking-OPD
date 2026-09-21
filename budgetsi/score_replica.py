"""Frozen student scoring on spare teacher-GPU capacity, using the original Engine."""
import tempfile
from pathlib import Path
from budgetsi.student_replica import StudentReplica, _TorchBackend, ReplicaError
from budgetsi.gpu_lease import device_lease, release_transient_cuda


class ScoreBackend(_TorchBackend):
    def __init__(self, config):
        self.lease_path = config['student_score_replica']['lease_path']
        with device_lease(self.lease_path):
            super().__init__(config)
            release_transient_cuda()
        from budgetsi.social_loop import Engine
        self.scratch = tempfile.TemporaryDirectory(prefix='budgetsi-score-replica-')
        self.engine = Engine(self.model, None, {'student': self.tokenizer},
                             self.max_context, Path(self.scratch.name), temperature=self.temperature)
        self.engine.accepting = True

    def sync_adapter(self, state):
        with device_lease(self.lease_path):
            try:
                return super().sync_adapter(state)
            finally:
                release_transient_cuda()

    def generate_batch(self, requests):
        raise ReplicaError('Scoring replica cannot generate')

    def score(self, request):
        if request.get('model') != 'student' or request.get('operation') != 'score':
            raise ReplicaError('Scoring replica accepts only student probability requests')
        self.engine.snapshot = request['snapshot']
        with device_lease(self.lease_path):
            try:
                return self.engine.call(request)
            finally:
                release_transient_cuda()


class StudentScoreReplica(StudentReplica):
    def __init__(self, config, **kwargs):
        settings = config['student_score_replica']
        super().__init__(config, settings['cuda_visible_devices'], _backend_factory=ScoreBackend, **kwargs)
