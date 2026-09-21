"""Build the user-approved train200/test90 initializations from public evidence."""
import argparse,ast,collections,hashlib,json,re,sys,unicodedata,difflib
from pathlib import Path
EPISODES_SHA='1f3f9dc5d7809bbbbfbaa47179a558826ff0e45c703b449b435acb2c84e5ecd4'
DUMP_SHA='324c037f4ac3ca6605c800bfe74eb1c11b05ed8c96c26df48c7560a7dac90da2'
CANON=lambda x:json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()
H=lambda x:hashlib.sha256(x).hexdigest()
NORM=lambda x:' '.join(unicodedata.normalize('NFKC',x).casefold().split())
W=lambda p,x:p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')

def build(episodes_path,test_path,directory):
 d=Path(directory);data=Path(episodes_path).read_bytes();assert H(data)==EPISODES_SHA
 episodes=[json.loads(x) for x in data.splitlines() if x.strip()]
 eval_bytes=Path(test_path).read_bytes();test=json.loads(eval_bytes)['scenes'];testids={x['environment']['pk'] for x in test}
 agents_export=json.loads((d/'AgentProfile_public.json').read_text());env_export=json.loads((d/'EnvironmentProfile_public.json').read_text());relations_export=json.loads((d/'relationship_profiles.json').read_text())
 for e in [agents_export,env_export,relations_export]:assert e['source_dump_sha256']==DUMP_SHA
 envs={x['pk']:x for x in env_export['profiles']};agents={x['pk']:x for x in agents_export['profiles']}
 relations=collections.defaultdict(list)
 for r in relations_export['relationship_profiles']:relations[(r['agent_1_id'],r['agent_2_id'],r['relationship'])].append(r['pk'])
 trainids={x['environment_id'] for x in episodes}-testids
 assert len(trainids)==200 and len(testids)==90 and len(test)==450
 pairs=collections.defaultdict(set)
 for e in episodes:
  if e['environment_id'] in trainids:
   assert NORM(e['scenario'])==NORM(envs[e['environment_id']]['scenario'])
   pairs[(e['environment_id'],tuple(e['agent_ids']))].add(e['episode_id'])
 testhash={H(NORM(x['environment']['scenario']).encode()) for x in test}
 trainhash={H(NORM(envs[e]['scenario']).encode()) for e in trainids};assert not(trainhash&testhash)
 near=[]
 for eid in sorted(trainids):
  a=NORM(envs[eid]['scenario'])
  for tid in sorted(testids):
   b=NORM(envs[tid]['scenario'])
   if min(len(a),len(b))/max(len(a),len(b))<.8:continue
   score=difflib.SequenceMatcher(None,a,b,autojunk=False).ratio()
   if score>=.85:near.append({'train_environment_id':eid,'test_environment_id':tid,'character_similarity':score})
 if near:raise ValueError('cross-split near duplicate requires review')
 scenes=[];occupation_flags=0
 for (eid,pair),sourceids in sorted(pairs.items()):
  env=envs[eid];ap=[agents[a] for a in pair];rel=relations[(pair[0],pair[1],env['relationship'])]
  if not rel:raise ValueError('missing exact ordered native relationship')
  age=env['age_constraint'];assert isinstance(age,str)
  if age!='[(18, 70), (18, 70)]':
   bounds=ast.literal_eval(age);assert len(bounds)==2
   if any(not(lo<=a['age']<=hi) for a,(lo,hi) in zip(ap,bounds)):raise ValueError('age constraint mismatch')
  quality=[];occupation=env.get('occupation_constraint')
  if occupation and occupation.lower() not in ['nan','none']:
   allowed=ast.literal_eval(occupation)
   if any(opts and 'any' not in [str(x).casefold() for x in opts] and a['occupation'].casefold() not in [str(x).casefold() for x in opts] for a,opts in zip(ap,allowed)):
    quality.append('occupation_literal_mismatch_informational_not_an_official_sampler_filter');occupation_flags+=1
  cid='pi200_'+H(CANON([eid,pair]))[:20]
  scenes.append({'id':cid,'seed':int(H(CANON(['pi200_school_v1',eid,pair]))[:8],16)%2147483647,'student_agent_index':0,'environment':env,'agents':ap,'source':{'dataset':'cmu-lti/sotopia-pi','revision':'e583406958ff132f6749ca87a2f9aa31ae3c0fa1','source_episode_ids_for_pair_provenance':sorted(sourceids),'relationship_profile_ids':sorted(rel),'native_profiles_sha256':H(CANON({'environment':env,'agents':ap})),'scenario_sha256':H(NORM(env['scenario']).encode()),'quality_notes':quality}})
 payload={'schema_version':'budgetsi_school_pi200_v1','status':'INITIALIZATIONS_VERIFIED_NO_MODEL_GENERATION','split_policy':'all200_public_non_test_no_dev_user_approved','student_role':0,'scenes':scenes}
 W(d/'inputs_200.json',payload)
 audit={'schema_version':'school_pi200_audit_v1','train_scenarios':len(trainids),'train_ordered_configurations':len(scenes),'dev_scenarios':0,'test_scenarios':90,'test_configurations':450,'train_environment_ids':sorted(trainids),'test_environment_ids':sorted(testids),'episodes_sha256':EPISODES_SHA,'native_dump_sha256':DUMP_SHA,'test_inputs_sha256':H(eval_bytes),'inputs_200_sha256':H((d/'inputs_200.json').read_bytes()),'all_relationship_ordered_matches':len(scenes),'relationship_missing':0,'relationship_reverse_only':0,'native_relationship_records_available':len(relations_export['relationship_profiles']),'age_constraint_failures':0,'occupation_literal_review_count':occupation_flags,'occupation_filter_applied':False,'exact_normalized_train_test_scenario_overlap':0,'near_duplicate_character_threshold':.85,'near_duplicate_cross_split':near,'configs_per_environment_histogram':dict(sorted(collections.Counter(collections.Counter(s['environment']['pk'] for s in scenes).values()).items())),'input_target_policy':'initialization_metadata_only_no_historical_messages_answers_rewards','evaluation_manifest_unchanged':True}
 W(d/'manifest.json',audit)
 print(json.dumps({k:v for k,v in audit.items() if k not in ['train_environment_ids','test_environment_ids']},indent=2))
 return audit

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--episodes',required=True);p.add_argument('--test-inputs',required=True);p.add_argument('--directory',default=str(Path(__file__).parent));a=p.parse_args();build(a.episodes,a.test_inputs,a.directory)
