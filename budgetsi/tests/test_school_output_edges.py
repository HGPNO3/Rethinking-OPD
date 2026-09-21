"""Reviewed output failures preserve executed nodes; infrastructure stays fatal."""
import asyncio
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'social_protocol'))
import runner
from budgetsi.batch_validation import validate_batch


class SchoolOutputEdges(unittest.TestCase):
    def test_output_boundaries_preserve_only_executed_prefix(self):
        cases = [
            ('context_exhausted', {'prompt_token_ids':[1]}),
            ('truncated_or_no_eos', {'generated_token_ids':[1], 'finish_reason':'length', 'raw_text':'{'}),
            ('Generated thinking marker in nonthinking action', {'raw_text':'<think>abc</think><|im_end|>'}),
            ('Tokens after action EOS', {'raw_text':'{}<|im_end|>x<|im_end|>'}),
            ('nonfinite JSON', {'raw_text':'{"action_type":"speak","argument":NaN}<|im_end|>', 'finish_reason':'stop'}),
        ]
        for reason, receipt in cases:
            with self.subTest(reason=reason):
                dialogue={'id':'a','events':[{'role':0}],'complete':False,'failure':{'reason':reason,'receipt':receipt}}
                before=copy.deepcopy(dialogue)
                result=validate_batch({'status':'partial'},[dialogue],['a'],[{'id':'a:0:0'}])
                self.assertEqual(result['validated_selected_nodes'],1)
                self.assertEqual(dialogue,before)
                self.assertFalse(result['resampled_for_quality'])
                self.assertFalse(result['invalid_outputs_repaired'])
                with self.assertRaises(ValueError):
                    validate_batch({'status':'partial'},[dialogue],['a'],[{'id':'a:1:0'}])

    def test_wrong_action_type_is_a_schema_error_not_a_python_crash(self):
        for value in ([], {}, 3, None):
            with self.assertRaisesRegex(ValueError,'invalid action schema'):
                runner.strict_action(json.dumps({'action_type':value,'argument':'x'}))

    def test_unknown_or_infrastructure_failure_remains_fatal(self):
        for reason in ('CUDA out of memory', 'HTTP 500', 'invalid raw prompt probability', 'unknown'):
            d={'id':'a','events':[],'complete':False,'failure':{'reason':reason,'receipt':{'raw_text':'x'}}}
            with self.assertRaises(ValueError):
                validate_batch({'status':'partial'},[d],['a'],[])

    def test_score_overflow_executes_source_but_does_not_train_node(self):
        class Session:
            done=False
            active_role=0
            def messages(self, role):return [{'role':'user','content':'visible'}]
            async def step(self, action):self.done=True
        class Student:
            async def generate(self,*args):return {'action':{'action_type':'speak','argument':'valid'}}
        with tempfile.TemporaryDirectory() as out:
            pilot=runner.Pilot(SimpleNamespace(output=out,seed=1,teacher_context_mode='reference_context'),Student(),object())
            async def inspect(session,messages,gen,key):
                pilot.rows.append({'id':key,'selected':'turn'})
                raise runner.InvalidAction('score_context_exhausted',{'prompt_token_ids':[1],'target_ids':[2]})
            pilot.inspect=inspect
            with patch.dict(sys.modules,{'adapter':SimpleNamespace(create_session=lambda _:Session())}):
                asyncio.run(pilot.dialogue({'id':'a'}))
            self.assertTrue(pilot.dialogues[0]['complete'])
            self.assertEqual(len(pilot.dialogues[0]['events']),1)
            self.assertEqual(pilot.selected,[])
            self.assertIsNone(pilot.rows[0]['selected'])
            self.assertEqual(pilot.rows[0]['status'],'score_context_exhausted')


if __name__ == '__main__':
    unittest.main()
