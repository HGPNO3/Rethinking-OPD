"""Data split, native relationship, scheduler coverage and resume acceptance."""
import collections,copy,hashlib,json,tempfile,unittest
from pathlib import Path
from budgetsi.school_data import DATA_DIR,DEFAULT_INPUTS,load_initializations,schedule_batch,dataset_identity

class SchoolDataTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.scenes=load_initializations()
 def test_exact_user_approved_split_and_native_relationships(self):
  self.assertEqual(len(self.scenes),1129)
  self.assertEqual(len({s['environment']['pk'] for s in self.scenes}),200)
  manifest=json.loads((DATA_DIR/'manifest.json').read_text())
  self.assertEqual(manifest['dev_scenarios'],0)
  self.assertEqual(manifest['test_configurations'],450)
  relations={r['pk']:r for r in json.loads((DATA_DIR/'relationship_profiles.json').read_text())['relationship_profiles']}
  for s in self.scenes:
   r=relations[s['source']['relationship_profile_ids'][0]]
   self.assertEqual([r['agent_1_id'],r['agent_2_id']],[a['pk'] for a in s['agents']])
   self.assertEqual(r['relationship'],s['environment']['relationship'])
   self.assertEqual(s['student_agent_index'],0)
 def test_balanced_distinct_batches_complete_pair_coverage_and_unique_ids(self):
  counts=collections.Counter();pairs=collections.defaultdict(set);ids=set()
  # 200*7 visits cover every published pair (4-7 per environment).
  sampled=[]
  for batch in range(88):
   rows=schedule_batch(self.scenes,batch)
   self.assertEqual(len({r['environment']['pk'] for r in rows}),16)
   sampled.extend(rows)
  for row in sampled:
   eid=row['environment']['pk'];counts[eid]+=1;pairs[eid].add(row['sampling']['initialization_id'])
   self.assertNotIn(row['id'],ids);ids.add(row['id'])
  self.assertLessEqual(max(counts.values())-min(counts.values()),1)
  self.assertEqual(set.union(*pairs.values()),{s['id'] for s in self.scenes})
 def test_resume_and_arm_schedules_identical_and_do_not_mutate_source(self):
  before=copy.deepcopy(self.scenes)
  uninterrupted=[schedule_batch(self.scenes,b) for b in range(4)]
  resumed=[schedule_batch(load_initializations(),b) for b in range(2,4)]
  self.assertEqual(uninterrupted[2:],resumed)
  uninterrupted[0][0]['agents'][0]['first_name']='MUTATION_TEST'
  self.assertEqual(self.scenes,before)
  self.assertNotEqual(schedule_batch(self.scenes,0,seed=10),schedule_batch(self.scenes,0,seed=11))
 def test_checksum_tampering_rejected(self):
  with tempfile.TemporaryDirectory() as td:
   d=Path(td);(d/'inputs_200.json').write_bytes(DEFAULT_INPUTS.read_bytes()+b' ')
   (d/'manifest.json').write_bytes((DATA_DIR/'manifest.json').read_bytes())
   with self.assertRaisesRegex(ValueError,'SHA256'):load_initializations(d/'inputs_200.json')
 def test_invalid_cursor_and_duplicate_scenes_rejected(self):
  for index in [-1,1.1,True]:
   with self.assertRaises(ValueError):schedule_batch(self.scenes,index)
  with self.assertRaises(ValueError):schedule_batch(self.scenes,0,batch_size=201)
  with self.assertRaises(ValueError):schedule_batch(self.scenes+[self.scenes[0]],0)
 def test_identity_matches_file(self):
  self.assertEqual(dataset_identity()['input_sha256'],hashlib.sha256(DEFAULT_INPUTS.read_bytes()).hexdigest())

if __name__=='__main__':unittest.main()
