import json,math,tempfile,unittest
from pathlib import Path
import torch
from budgetsi.diagnostics import student_summary,compare,Accumulator
from budgetsi.diagnostic_run import publish
from budgetsi.telemetry import Telemetry

class DiagnosticsTests(unittest.TestCase):
    def test_identical_distributions_zero_divergence_and_exact_entropy(self):
        logits=torch.tensor([[2.,1.,0.,-1.],[0.,1.,2.,3.]])
        summary=student_summary(logits,.7,2)
        values=compare(summary,logits/.7,1.)
        self.assertTrue(torch.all(values['overlap_ratio']==1))
        torch.testing.assert_close(values['student_topk_tail_kl_pq'],torch.zeros(2),atol=1e-7,rtol=0)
        torch.testing.assert_close(values['student_topk_tail_js'],torch.zeros(2),atol=1e-7,rtol=0)
        torch.testing.assert_close(values['teacher_minus_student_entropy'],torch.zeros(2),atol=1e-7,rtol=0)
        p=(logits/.7).softmax(-1)
        torch.testing.assert_close(values['student_entropy'],-(p*p.log()).sum(-1))
        torch.testing.assert_close(values['student_overlap_mass'],p.topk(2,-1).values.sum(-1))
    def test_disjoint_support_and_token_weighted_aggregation(self):
        p=torch.tensor([[4.,3.,0.,-1.]])
        q=p.flip(-1)
        v=compare(student_summary(p,1.,2),q)
        self.assertEqual(v['overlap_ratio'].item(),0)
        self.assertEqual(v['student_overlap_mass'].item(),0)
        self.assertGreater(v['student_topk_tail_js'].item(),0)
        acc=Accumulator();acc.add(v);acc.add({k:torch.cat([x,x]) for k,x in v.items()})
        r=acc.result();self.assertEqual(r['response_tokens'],3)
        self.assertIsNone(r['overlap_logprob_gap_mean'])
    def test_step_zero_and_committed_update_replay(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);batch=root/'batch';batch.mkdir()
            (batch/'diagnostics.json').write_text(json.dumps(dict(optimizer_step=0,snapshot='a',selected_node_ids=['n'],seconds=.1,metrics={'overlap_ratio':.5})))
            r=dict(optimizer_step=1,snapshot_before='a',nodes=['n'],update_and_restore_seconds=.2,
                   metrics={'actor/pg_loss':[.1,.3]},peak_allocated_gib=[12.])
            t=Telemetry(root/'logs',mode='disabled')
            publish(t,r,batch,{'optimizer':{'lr':1e-6}});publish(t,r,batch,{'optimizer':{'lr':1e-6}})
            rows=[json.loads(x) for x in t.path.read_text().splitlines()]
            self.assertEqual(len(rows),2);self.assertEqual(rows[0]['metrics']['optimizer_step'],0)
            self.assertEqual(rows[1]['metrics']['upstream/actor/pg_loss'],.2)
            with self.assertRaises(ValueError):publish(t,{**r,'snapshot_before':'other'},batch,{'optimizer':{'lr':1e-6}})

class ActualWandbTests(unittest.TestCase):
    def test_actual_offline_wandb_file_and_scalar_ledger(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            t=Telemetry(root,mode='offline',project='budgetsi-diagnostics-cpu-test',config={'variant':{'name':'sampled_token'}})
            try:
                t.log('pre_0',{'optimizer_step':0,'diagnostics/overlap_ratio':.25})
                t.log('update_1',{'optimizer_step':1,'train/learning_rate':1e-6})
            finally:t.finish()
            files=list(root.glob('wandb/offline-run-*/run-*.wandb'))
            self.assertEqual(len(files),1);self.assertGreater(files[0].stat().st_size,0)
            self.assertEqual(len(t.path.read_text().splitlines()),2)

if __name__=='__main__':unittest.main()
