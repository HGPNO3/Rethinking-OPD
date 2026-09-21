import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
from budgetsi.diagnostic_run import diagnose, publish


class DiagnosticTemperatureTests(unittest.TestCase):
    def test_t1_is_used_for_distribution_and_receipt(self):
        model=torch.nn.Linear(1,1)
        action=SimpleNamespace(node_id='n',sampling_temperature=1.,student_prompt_ids=(1,),teacher_prompt_ids=(1,),target_ids=(2,),snapshot_id='s')
        logits=torch.tensor([[2.,0.,-1.]])
        with patch('budgetsi.diagnostic_run.response_logits',return_value=logits),patch('budgetsi.diagnostic_run._binding',return_value='binding'):
            result=diagnose(model,model,[action],{'max_actions':4,'k':2},0)
        self.assertEqual(result['student_temperature'],1.)
        self.assertAlmostEqual(result['metrics']['student_topk_tail_js'],0.,places=7)
        self.assertTrue(model.training)
        action.sampling_temperature=.7
        with patch('budgetsi.diagnostic_run.response_logits',return_value=logits),patch('budgetsi.diagnostic_run._binding',return_value='binding'):
            result=diagnose(model,model,[action],{'max_actions':4,'k':2},0)
        self.assertGreater(result['metrics']['student_topk_tail_js'],0.)

    def test_publish_rejects_stale_temperature_before_logging(self):
        with tempfile.TemporaryDirectory() as d:
            batch=Path(d)
            (batch/'diagnostics.json').write_text(json.dumps({'optimizer_step':0,'snapshot':'s','selected_node_ids':['n'],'student_temperature':.7,'teacher_temperature':1.}))
            with self.assertRaisesRegex(ValueError,'temperature'):
                publish(None,{'optimizer_step':1,'snapshot_before':'s','nodes':['n']},batch,{'temperature':1.})

if __name__=='__main__':unittest.main()
