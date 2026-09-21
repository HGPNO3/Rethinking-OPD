"""Cooperative forward-memory lease for the two exact-teacher GPUs.

Student scoring readers may run together on disjoint GPUs. An exact-teacher
writer excludes both readers and has priority over newly arriving readers.
Only transient CUDA allocations are released; resident weights stay loaded.
"""
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
import fcntl


@contextmanager
def device_lease(path, *, exclusive=False):
    path = Path(path)
    if not path.is_absolute():
        raise ValueError('GPU lease requires an absolute experiment path')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix('.turnstile').open('a') as turn, path.open('a') as resource:
        fcntl.flock(turn, fcntl.LOCK_EX)
        try:
            fcntl.flock(resource, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
            if not exclusive:
                fcntl.flock(turn, fcntl.LOCK_UN)
            try:
                yield
            finally:
                fcntl.flock(resource, fcntl.LOCK_UN)
        finally:
            fcntl.flock(turn, fcntl.LOCK_UN)


def release_transient_cuda():
    import torch
    if torch.cuda.is_initialized():
        for device in range(torch.cuda.device_count()):
            with torch.cuda.device(device):
                torch.cuda.synchronize()
                torch.cuda.empty_cache()


def teacher_scoring_lease(dispatch):
    @wraps(dispatch)
    def guarded(self, request):
        config = self.resolve_config(request['binding'])
        settings = config.get('student_score_replica')
        if not settings or request['operation'] not in {'opd', 'diagnostics', 'validate_raw'}:
            return dispatch(self, request)
        with device_lease(settings['lease_path'], exclusive=True):
            try:
                return dispatch(self, request)
            finally:
                release_transient_cuda()
    return guarded
