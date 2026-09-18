import copy,unittest
from batch_validation import validate_batch
class TestBatch(unittest.TestCase):
 def test_valid_prefix_retained_without_repair(self):
  d={'id':'a','complete':False,'events':[{'role':0}],'failure':{'reason':'invalid action schema','receipt':{'raw_text':'null'}}}
  r=validate_batch({'status':'partial'},[d],['a'],[{'id':'a:0:0'}]);self.assertFalse(r['invalid_outputs_repaired'])
  with self.assertRaises(ValueError):validate_batch({'status':'partial'},[d],['a'],[{'id':'a:1:0'}])
  d['failure']['reason']='Generated thinking marker in nonthinking action'
  with self.assertRaises(ValueError):validate_batch({'status':'partial'},[d],['a'],[])
 def test_crash_missing_or_duplicate_fails(self):
  with self.assertRaises(ValueError):validate_batch({'status':'partial'},[],['a'],[])
  with self.assertRaises(ValueError):validate_batch({'status':'failed','fatal_error':'CUDA'},[],[],[])
  d={'id':'a','complete':True,'events':[]}
  with self.assertRaises(ValueError):validate_batch({'status':'completed'},[d,d],['a'],[])

 def test_fenced_json_is_failed_not_repaired(self):
  d={'id':'a','complete':False,'events':[{'role':0}],'failure':{'reason':'Expecting value: line 1 column 1 (char 0)','receipt':{'raw_text':'```json\n{}\n```<|im_end|>','finish_reason':'stop'}}}
  r=validate_batch({'status':'partial'},[d],['a'],[{'id':'a:0:0'}]);self.assertFalse(r['invalid_outputs_repaired'])
  for text,finish in [('<think>x</think>','stop'),('```json\n{}','length'),('','stop')]:
   d['failure']['receipt'].update(raw_text=text,finish_reason=finish)
   with self.assertRaises(ValueError):validate_batch({'status':'partial'},[d],['a'],[])

 def test_json_syntax_must_reproduce_and_preserve_receipt(self):
  import json
  body='{"action_type":"speak","argument":"Okay'
  try:json.loads(body)
  except json.JSONDecodeError as e:reason=str(e)
  d={'id':'a','complete':False,'events':[{'role':0}],'failure':{'reason':reason,'receipt':{'raw_text':body+'<|im_end|>','finish_reason':'stop'}}}
  before=copy.deepcopy(d)
  r=validate_batch({'status':'partial'},[d],['a'],[{'id':'a:0:0'}])
  self.assertEqual(before,d);self.assertFalse(r['invalid_outputs_repaired'])
  for text,finish,why in [(body+'<|im_end|>','length',reason),(body+'<|im_end|>','stop','CUDA'),('<think>'+body+'<|im_end|>','stop',reason),('{}<|im_end|>','stop',reason)]:
   d['failure']={'reason':why,'receipt':{'raw_text':text,'finish_reason':finish}}
   with self.assertRaises(ValueError):validate_batch({'status':'partial'},[d],['a'],[])
