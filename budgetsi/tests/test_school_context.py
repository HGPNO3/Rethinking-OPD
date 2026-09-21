"""P0 causal-boundary and selection-invariance checks; no model downloads."""
import copy
import asyncio
import hashlib
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

PROTOCOL = Path(__file__).resolve().parents[1] / 'social_protocol'
sys.path.insert(0, str(PROTOCOL))
import runner
import test_reference_opd as reference_fixture


class SchoolContext(unittest.TestCase):
    def collect(self, mode):
        def pilot(args, student, teacher):
            args.teacher_context_mode = mode
            return runner.Pilot(args, student, teacher)
        with patch.object(reference_fixture, 'Pilot', side_effect=pilot):
            return reference_fixture.ReferenceFlow().collect()

    def test_only_reference_block_changes_and_inputs_are_immutable(self):
        messages = [
            {'role': 'system', 'content': 'ORIGINAL_SYSTEM'},
            {'role': 'user', 'content': 'Earlier question'},
            {'role': 'assistant', 'content': 'Earlier answer'},
            {'role': 'user', 'content': 'VISIBLE_NOW'},
        ]
        original = copy.deepcopy(messages)
        reference = {'action_type': 'speak', 'argument': 'SELECTED_REFERENCE'}
        same = runner.opd_messages(messages, 'turn', reference, 'same_context')
        guided = runner.opd_messages(messages, 'turn', reference, 'reference_context')
        self.assertEqual(same, original)
        self.assertEqual(messages, original)
        self.assertEqual(guided[:-1], original[:-1])
        prefix, block = guided[-1]['content'].split('\n\n<teacher_only_reference>\n')
        self.assertEqual(prefix, original[-1]['content'])
        data = json.loads(block.removesuffix('\n</teacher_only_reference>'))
        self.assertEqual(data['teacher_only_reference']['reference_action'], reference)
        self.assertEqual(data['teacher_only_reference']['status'], 'unexecuted_alternative')
        self.assertEqual(guided, runner.opd_messages(messages, 'expression', reference, 'reference_context'))
        guided[0]['content'] = 'MUTATION'
        self.assertEqual(messages, original)

    def test_both_arms_select_identical_teacher_and_student_targets(self):
        same, same_calls = self.collect('same_context')
        guided, guided_calls = self.collect('reference_context')
        for record, mode in [(same, 'same_context'), (guided, 'reference_context')]:
            runner.validate_reference_record(record, expected_mode=mode)
        for field in ('specialty', 'reference_action', 'target_ids', 'student_prefix', 'snapshot_id', 'behavior_logprobs'):
            self.assertEqual(same[field], guided[field])
        self.assertEqual(same['teacher_messages'], same['visible_messages'])
        self.assertEqual(same['teacher_score']['prompt_token_ids'], same['student_prefix'])
        self.assertNotIn('SELECTED_REF', str(same['teacher_messages']))
        self.assertIn('SELECTED_REF', str(guided['teacher_messages']))
        self.assertEqual(
            [(k, v) for k, v in same_calls if k != 'opd_teacher_score'],
            [(k, v) for k, v in guided_calls if k != 'opd_teacher_score'],
        )

    def test_cross_arm_and_forged_prefix_records_are_rejected(self):
        same, _ = self.collect('same_context')
        with self.assertRaises(ValueError):
            runner.validate_reference_record(same, expected_mode='reference_context')
        bad = copy.deepcopy(same)
        bad['teacher_context_mode'] = 'reference_context'
        with self.assertRaises(ValueError):
            runner.validate_reference_record(bad)
        bad = copy.deepcopy(same)
        bad['teacher_score']['prompt_token_ids'] = [999]
        bad['teacher_prompt_sha256'] = hashlib.sha256(json.dumps([999]).encode()).hexdigest()
        with self.assertRaisesRegex(ValueError, 'identical'):
            runner.validate_reference_record(bad)
        bad = copy.deepcopy(same)
        bad['teacher_messages'][-1]['content'] += '\nREFERENCE_LEAK'
        with self.assertRaises(ValueError):
            runner.validate_reference_record(bad)

    def test_unknown_mode_and_unsupported_chat_shape_fail_closed(self):
        with self.assertRaises(ValueError):
            runner.teacher_context_binding('typo')
        with self.assertRaises(ValueError):
            runner.opd_messages([{'role': 'assistant', 'content': 'x'}], 'turn', {'action_type':'speak','argument':'x'})

    def test_rollout_temperature_and_probability_receipt_stay_aligned(self):
        client=runner.Client.__new__(runner.Client)
        client.max_context=20;client.model='student';client.known={2,99};client.eos={99}
        client.token_contract={};client.sampling_temperature=1.
        client.render=lambda messages:[1]
        body='{"action_type":"speak","argument":"x"}'
        client.tok=SimpleNamespace(decode=lambda ids,**kwargs:body+('<|im_end|>' if ids[-1]==99 else ''))
        calls=[]
        async def request(payload,kind):
            calls.append(payload)
            return {'choices':[{'token_ids':[2,99],'logprobs':{'token_logprobs':[-.2,-.3]},'finish_reason':'stop'}]}
        client.request=request
        with patch.dict(sys.modules,{'token_contract':SimpleNamespace(validate_action_ids=lambda *args:None)}):
            record=asyncio.run(client.generate([{'role':'user','content':'x'}],7,'source_A'))
        self.assertEqual(calls[0]['temperature'],1.)
        self.assertEqual(record['behavior_temperature'],1.)
        self.assertEqual(record['behavior_logprobs'],[-.2,-.3])
        self.assertEqual(calls[0]['seed'],7)


if __name__ == '__main__':
    unittest.main()
