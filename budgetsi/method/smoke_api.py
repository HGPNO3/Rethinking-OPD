import argparse,asyncio,json
from pathlib import Path
import aiohttp
from runner import Client
async def run(out):
 results=[];metrics=[];cfg=json.loads(Path(__file__).with_name('config.json').read_text());runtime=cfg['runtime']
 async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=120)) as http:
  for endpoint,name,model in [('http://127.0.0.1:18004','student',runtime['student_folder']),(runtime['teacher_endpoint'],'teacher',runtime['teacher_folder'])]:
   c=Client(endpoint,name,runtime['model_root']+model,http,cfg['context'],metrics)
   g=await c.generate([{'role':'user','content':'Reply with exactly this JSON action: {"action_type":"speak","argument":"Hello."}'}],20260915,'api_smoke')
   r=await c.score(g['prompt_token_ids'],g['generated_token_ids'],'api_score_smoke')
   assert len(r['raw_logprobs'])==len(g['generated_token_ids'])
   results.append({'model':name,'generation':g,'raw_score':r})
 Path(out).write_text(json.dumps({'passed':True,'results':results,'metrics':metrics},indent=2))
p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();asyncio.run(run(a.output))
