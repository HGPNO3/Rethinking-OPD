"""Materialize the reviewed source without starting a service or experiment."""
import argparse,hashlib,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def materialize(family,mode,destination):
    destination=Path(destination)
    if destination.exists():raise FileExistsError('Refuse to overwrite an existing runtime')
    destination.mkdir(parents=True)
    for source in [ROOT/'method',ROOT/'variants'/family/mode]:
        for p in source.glob('*.py'):shutil.copyfile(p,destination/p.name)
    cfg=json.loads((ROOT/'configs'/f'{family}.json').read_text())
    cfg.pop('review_metadata')
    cfg['target_nodes']=1000 if mode=='fresh' else 3000
    cfg['scope']='REVIEW_ONLY: bind local runtime, model artifacts, scene manifest, and external HG approval before execution'
    cfg['runtime']={}
    cfg['files_sha256']={}
    # Missing required runtime entries/approval fail closed. Never reuse historical approval.
    (destination/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    shutil.copyfile(ROOT/'configs'/f'{family}_manifest.json',destination/'manifest.json')
    receipt={'family':family,'mode':mode,'status':'source_materialized_only_not_run_authorized',
             'missing':['local runtime executables and model paths','bound model/tokenizer file hashes','exact private inputs.json matching manifest','new clean Git snapshot and external approval'],
             'source_files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(destination.glob('*.py'))}}
    if mode=='continuation':receipt['missing'].append('immutable parent checkpoint/optimizer lineage and continuation bindings')
    (destination/'MATERIALIZATION.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--family',choices=['qwen3','qwen35'],required=True)
    p.add_argument('--mode',choices=['fresh','continuation'],required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();print(json.dumps(materialize(a.family,a.mode,a.output),indent=2))
