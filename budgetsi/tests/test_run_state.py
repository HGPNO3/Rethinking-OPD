import copy
import json
import tempfile
import unittest
from pathlib import Path
from budgetsi.run_state import Ledger, batch_scenes, choose_quota, digest


class RunStateTests(unittest.TestCase):
    def test_quota_is_used_actions_and_sorted_tail(self):
        records = [{'id': f'n{i:04d}'} for i in reversed(range(990, 1005))]
        taken, excluded = choose_quota(records, {f'n{i:04d}' for i in range(990)}, 1000)
        self.assertEqual([r['id'] for r in taken], [f'n{i:04d}' for i in range(990, 1000)])
        self.assertEqual(excluded, [f'n{i:04d}' for i in range(1000, 1005)])
        with self.assertRaises(ValueError):
            choose_quota([{'id': 'x'}, {'id': 'x'}], set(), 1000)
        with self.assertRaises(ValueError):
            choose_quota([{'id': 'x'}], {'x'}, 1000)

    def receipt(self, out, step, nodes, before, after):
        folder = out / f'attempt{step}/update'
        (folder / 'adapter').mkdir(parents=True)
        (folder / 'adapter/adapter_model.safetensors').write_bytes(b'adapter')
        (folder / 'optimizer.pt').write_bytes(b'optimizer')
        r = dict(status='passed', checkpoint=str(folder.relative_to(out)), batch=step-1,
                 optimizer_step=step, nodes=nodes, snapshot_before=before, snapshot_after=after,
                 adapter_file_sha256=digest(folder/'adapter/adapter_model.safetensors'),
                 optimizer_file_sha256=digest(folder/'optimizer.pt'))
        (folder / 'result.json').write_text(json.dumps(r))
        return r

    def test_crash_before_commit_does_not_count_orphan_then_resume_exactly(self):
        with tempfile.TemporaryDirectory() as t:
            out = Path(t)
            cfg = {'target_nodes': 3}
            ledger = Ledger(out, cfg, 'commit', False)
            r = self.receipt(out, 1, ['a', 'b'], 'base', 'one')
            resumed = Ledger(out, cfg, 'commit', True)
            self.assertEqual(resumed.used, set())  # saved checkpoint alone is not a committed update
            resumed.commit(0, {'batch': 0}, r)
            restored = Ledger(out, cfg, 'commit', True)
            self.assertEqual(restored.used, {'a', 'b'})
            r2 = self.receipt(out, 2, ['c'], 'one', 'two')
            restored.commit(1, {'batch': 1}, r2)
            self.assertEqual(Ledger(out, cfg, 'commit', True).used, {'a', 'b', 'c'})
            with self.assertRaises(ValueError):
                Ledger(out, cfg, 'changed_commit', True)
            (out / r2['checkpoint'] / 'optimizer.pt').write_bytes(b'corrupted')
            with self.assertRaises(ValueError):
                Ledger(out, cfg, 'commit', True)

    def test_empty_batch_progress_and_overflow_fail_closed(self):
        with tempfile.TemporaryDirectory() as t:
            out = Path(t); cfg = {'target_nodes': 1}
            ledger = Ledger(out, cfg, 'commit', False)
            ledger.commit(0, {'batch': 0}, None)
            self.assertEqual(Ledger(out, cfg, 'commit', True).state['next_batch'], 1)
            r = self.receipt(out, 2, ['a', 'b'], 'base', 'one'); r['optimizer_step'] = 1
            with self.assertRaises(ValueError):
                ledger.commit(1, {'batch': 1}, r)
            self.assertEqual(Ledger(out, cfg, 'commit', True).used, set())

    def test_scene_cycles_preserve_reserved_split_and_unique_keys(self):
        pool = [{'id': str(i)} for i in range(100)]
        source = copy.deepcopy(pool)
        batches = [batch_scenes(pool, i, 16, 20360915) for i in range(10)]
        ids = [s['id'] for b in batches for s in b['scenes']]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(len({s.rsplit(':', 1)[-1] for s in ids}), 80)
        self.assertEqual(pool, source)
        self.assertEqual(batch_scenes(pool, 2, 16, 20360915), batches[2])


if __name__ == '__main__':
    unittest.main()
