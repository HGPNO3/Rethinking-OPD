import unittest,tempfile,json
from pathlib import Path
from train import batch_scenes,receipts,atomic,digest
from runner import PROMPT_VERSION,prompt_binding
class Check(unittest.TestCase):
 def test_scene_sampling_has_disjoint_ids_and_stable_resume(self):
  pool=[{'id':str(i),'seed':1} for i in range(100)]
  a=batch_scenes(pool,'acceptance',0,2,10);b=batch_scenes(pool,'formal',0,16,100010)
  self.assertEqual(a,batch_scenes(pool,'acceptance',0,2,10));self.assertFalse({s['id'] for s in a['scenes']} & {s['id'] for s in b['scenes']})
  self.assertEqual(pool[0],{'id':'0','seed':1})
 def test_receipts_ignore_incomplete_and_reject_duplicate(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t)
   def put(i,node,step,status='passed'):
    d=root/f'batch_{i:04d}'/'update';d.mkdir(parents=True);(d/'adapter').mkdir();(d/'adapter/adapter_model.safetensors').write_bytes(b'a');(d/'optimizer.pt').write_bytes(b'o')
    atomic(d/'result.json',dict(prompt_version=PROMPT_VERSION,prompt_binding=prompt_binding(),status=status,optimizer_step=step,adapter_before='a',adapter_after='a',used_node_ids=[node],selected_records=1,checkpoint_file_sha256=digest(d/'adapter/adapter_model.safetensors'),optimizer_file_sha256=digest(d/'optimizer.pt')))
   put(0,'one',1);put(1,'two',2,'starting');self.assertEqual(receipts(root)[0],['one'])
   p=root/'batch_0001/update/result.json';r=json.loads(p.read_text());r['status']='passed';atomic(p,r);self.assertEqual(receipts(root)[0],['one','two'])
   old=root/'batch_0000/update/result.json';old_data=json.loads(old.read_text());old_data.pop('prompt_version');atomic(old,old_data)
   with self.assertRaises(ValueError):receipts(root)
   old_data['prompt_version']=PROMPT_VERSION;atomic(old,old_data)
   put(2,'one',3)
   with self.assertRaises(AssertionError):receipts(root)
if __name__=='__main__':unittest.main()
