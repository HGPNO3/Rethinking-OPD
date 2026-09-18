"""Print a candidate upstream configuration; never launch models or Ray."""
import argparse,json,os,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--family',choices=['qwen3','qwen35'],default='qwen3')
p.add_argument('--train-file',required=True)
p.add_argument('--validation-file',required=True)
p.add_argument('--max-prompt-length',type=int,required=True)
p.add_argument('--max-response-length',type=int,required=True)
p.add_argument('--batch-size',type=int,required=True,help='Flat upstream prompt batch, NOT BudgetSI dialogue batch')
p.add_argument('--gpus',type=int,required=True)
a=p.parse_args()
c=json.loads((ROOT/'budgetsi/configs'/f'{a.family}.json').read_text())
if min(a.max_prompt_length,a.max_response_length,a.batch_size,a.gpus)<=0:p.error('sizes must be positive')
if a.max_prompt_length+a.max_response_length>c['context']:p.error('prompt + response exceeds recorded model service context')
env={**os.environ,'OPD_DRY_RUN':'1','ACTOR_MODEL_PATH':c['models']['student'],'REWARD_MODEL_PATH':c['models']['teacher'],
 'TRAIN_DATASET':a.train_file,'TRAIN_DATASET_NAME':'budgetsi_prefix_baseline','TEST_FILE':json.dumps([a.validation_file]),
 'MAX_PROMPT_LENGTH':str(a.max_prompt_length),'MAX_RESP_LENGTH':str(a.max_response_length),'MAX_VAL_RESP_LENGTH':str(a.max_response_length),
 'MINI_BATCH_SIZE':str(a.batch_size),'N_GPUS_PER_NODE':str(a.gpus),'PARALLEL_SIZE':'1','LR':str(c['optimizer']['lr']),
 'TEMPERATURE':str(c['temperature']),'TEACHER_TEMPERATURE':'1.0','MODEL_DTYPE':'bfloat16','ENABLE_THINKING':'False',
 'LORA_RANK':str(c['optimizer']['r']),'LORA_ALPHA':str(c['optimizer']['alpha']),'N_RESPONSES':'1','LOG_PROB_TOP_K':'0',
 'USE_KL':'False','ENABLE_FORMAT_REWARD':'False','IS_PLOT':'False','LOSS_AGG_MODE':'token-mean'}
# Pin relevant values rather than inheriting unintended shell overrides.
env.update(REPETITION_PENALTY='1.0',TOP_K_STRATEGY='only_stu',REWARD_WEIGHT_MODE='student_p',LR_SCHEDULER='constant')
r=subprocess.run(['bash',str(ROOT/'on_policy_distillation.sh')],cwd=ROOT,env=env,text=True,capture_output=True,check=True)
start=r.stdout.index('{');result=json.loads(r.stdout[start:])
result.update(family=a.family,scope='single-teacher fixed-prefix baseline proposal, NOT migrated social pipeline',
 caveats=['Flat prefixes do not reproduce online partner interaction or teacher selection.',
          'Batch and length split are explicit reviewer inputs, not inherited scientific approval.',
          'Qwen3.5 support in pinned verl is unverified; no compatibility claim.',
          'Printed argv is not a Hydra-resolved configuration or GPU validation.',
          'Original math reward callback/validation remain upstream baseline concerns; do not use their scores for social performance.'])
print(json.dumps(result,indent=2))
