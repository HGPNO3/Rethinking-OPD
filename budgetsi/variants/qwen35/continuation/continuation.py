"""Fail-closed immutable second-run lineage checks; no model/API calls."""
from pathlib import Path
from train import digest, receipts
from protocol_gate import load_json

def validate_continuation(cfg, out):
 c=cfg['continuation'];parent=Path(c['parent_run']);root=Path(c['parent_code']);out=Path(out)
 assert out.resolve()!=parent.resolve(), 'Never mutate parent run'
 old=load_json(parent/'config.json')
 for k in set(old)|set(cfg):
  if k not in {'scope','target_nodes','continuation'}:
   assert cfg.get(k)==old.get(k), f'Unexpected scientific/config change: {k}'
 assert cfg['target_nodes']==3000 and old['target_nodes']==1000
 for path,h in c['files_sha256'].items():
  assert digest(path)==h, f'Parent artifact changed: {path}'
 for name in c['unchanged_source_files']:
  assert digest(Path(__file__).parent/name)==digest(root/name), f'Unexpected implementation change: {name}'
 assert load_json(parent/'state.json')['status']=='completed'
 inherited=load_json(parent/'formal/result.json')
 assert inherited['used_nodes']==1000 and inherited['updates']==19
 assert (out/'acceptance').is_symlink() and (out/'acceptance').resolve()==(parent/'acceptance').resolve()
 for d in sorted((parent/'formal').glob('batch_*')):
  link=out/'formal'/d.name
  assert link.is_symlink() and link.resolve()==d.resolve(), f'Invalid historical batch link {link}'
 used,previous,updates=receipts(parent/'formal')
 assert len(used)==1000 and len(updates)==19
 assert str(previous)==inherited['checkpoint']
 return dict(status='passed',parent_run=str(parent),inherited_nodes=len(used),inherited_updates=len(updates),checkpoint=str(previous),target_nodes=3000)
