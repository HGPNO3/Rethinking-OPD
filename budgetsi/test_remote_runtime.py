"""Actual upstream actor: localhost JSON transport versus local teacher."""

import copy
import tempfile
import threading
from pathlib import Path
from urllib.error import HTTPError

import torch

from budgetsi.test_variant_runtime import VariantRuntime
from budgetsi.remote_teacher import RemoteTeacher, TeacherService, serve
from budgetsi.variant_spec import get_variant
from budgetsi.variant_bridge import score_with_actor, update_actor


class RemoteRuntime(VariantRuntime):
    def test_local_remote_exact_scores_and_update(self):
        for name in ("student_top16", "union_student", "sampled_token"):
            variant = get_variant(name)
            student, teacher, actions, optimizer, cfg, worker = self.models()
            with tempfile.TemporaryDirectory() as folder:
                config = dict(
                    schema_version="budgetsi_upstream_variants_online_v1",
                    opd_variant=name,
                    opd=variant.contract(),
                    context=32,
                )
                binding = {"test": name}
                service = TeacherService(
                    teacher, None, config, binding, {"test": "hash"}, Path(folder)
                )
                server = serve(service, 0)
                thread = threading.Thread(target=server.serve_forever, daemon=True)
                thread.start()
                try:
                    remote = RemoteTeacher(
                        f"http://127.0.0.1:{server.server_port}/", binding
                    )
                    kwargs = dict(
                        variant=variant, snapshot_id="snapshot", eos_ids={2}, pad_id=0
                    )
                    local_scores = score_with_actor(worker, teacher, actions, **kwargs)
                    remote_scores = score_with_actor(worker, remote, actions, **kwargs)
                    for a, b in zip(local_scores, remote_scores):
                        self.assertEqual(a.keys(), b.keys())
                        for key in a:
                            if isinstance(a[key], torch.Tensor):
                                torch.testing.assert_close(
                                    a[key], b[key], rtol=0, atol=0
                                )
                            else:
                                self.assertEqual(a[key], b[key])
                    initial = copy.deepcopy(student.state_dict())
                    update_actor(
                        worker, actions, local_scores, actor_config=cfg, **kwargs
                    )
                    expected = copy.deepcopy(student.state_dict())
                    student.load_state_dict(initial)
                    optimizer.state.clear()
                    # Regenerate actor revision after restoring parameters.
                    scores = score_with_actor(worker, remote, actions, **kwargs)
                    update_actor(worker, actions, scores, actor_config=cfg, **kwargs)
                    for key, value in student.state_dict().items():
                        torch.testing.assert_close(value, expected[key], rtol=0, atol=0)
                    self.assertTrue(remote.assert_frozen()["frozen"])
                    with self.assertRaises(HTTPError):
                        RemoteTeacher(remote.endpoint, {"wrong": "binding"})
                    with torch.no_grad():
                        next(teacher.parameters()).add_(1)
                    with self.assertRaises(HTTPError):
                        remote.assert_frozen()
                finally:
                    server.shutdown()
                    server.server_close()
                    thread.join()


if __name__ == "__main__":
    import unittest

    unittest.main()
