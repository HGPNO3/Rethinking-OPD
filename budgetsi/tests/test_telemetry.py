import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from budgetsi.telemetry import Telemetry

class TestTelemetry(unittest.TestCase):
    def fake(self, root, step=0):
        calls=[]
        run=SimpleNamespace(id='test',dir=str(root/'wandb/test/files'),url='https://example.invalid',step=step,
            define_metric=lambda *a,**k:None,log=lambda values,**kw:calls.append((values,kw)),finish=lambda **k:None)
        inits=[]
        return SimpleNamespace(init=lambda **kw:(inits.append(kw) or run),Settings=lambda **kw:kw),calls,inits
    def test_eager_and_offline_recovery(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);fake,calls,inits=self.fake(root)
            with patch.dict('sys.modules',wandb=fake):
                t=Telemetry(root);self.assertEqual(len(inits),1)
                t.log('pre_0',{'optimizer_step':0,'overlap':.5});t.finish()
                t=Telemetry(root);self.assertEqual(len(calls),2)
                t.log('pre_0',{'optimizer_step':0,'overlap':.5});self.assertEqual(len(calls),2)
                with self.assertRaises(ValueError):t.log('pre_0',{'optimizer_step':1})
                with self.assertRaises(ValueError):t.log('bad',{'x':float('nan')})
                t.finish(exit_code=1)
                self.assertEqual(json.loads((root/'wandb_status.json').read_text())['status'],'failed')
                self.assertNotIn('resume',inits[0])
    def test_online_resume_skips_acknowledged_history(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);t=Telemetry(root,mode='disabled')
            t.log('a',{'optimizer_step':0});t.log('b',{'optimizer_step':1})
            fake,calls,inits=self.fake(root,step=1)
            with patch.dict('sys.modules',wandb=fake):
                t=Telemetry(root,mode='online',entity='explicit')
                self.assertEqual(calls,[({'optimizer_step':1},{'step':1})])
                self.assertEqual(inits[0]['resume'],'allow')
    def test_sdk_failure_preserves_ledger(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);fake,calls,inits=self.fake(root)
            fake.init=lambda **kw: (_ for _ in ()).throw(RuntimeError('sdk unavailable'))
            with patch.dict('sys.modules',wandb=fake):
                t=Telemetry(root);t.log('update_1',{'optimizer_step':1});t.finish()
            self.assertEqual(json.loads(t.path.read_text())['event_id'],'update_1')
            self.assertEqual(json.loads((root/'wandb_status.json').read_text())['sdk_error_type'],'RuntimeError')
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);fake,calls,inits=self.fake(root)
            with patch.dict('sys.modules',wandb=fake):
                t=Telemetry(root)
                t.run.log=lambda *a,**kw: (_ for _ in ()).throw(RuntimeError('sdk unavailable'))
                t.log('update_1',{'optimizer_step':1});t.log('update_2',{'optimizer_step':2});t.finish()
            self.assertEqual(len(t.path.read_text().splitlines()),2)

    def test_online_requires_target(self):
        with tempfile.TemporaryDirectory() as d,patch.dict('os.environ',{},clear=True):
            with self.assertRaises(ValueError):Telemetry(d,mode='online')
if __name__=='__main__':unittest.main()
