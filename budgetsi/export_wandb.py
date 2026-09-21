"""Rebuild W&B events from the durable numeric ledger; never uploads dialogues."""
import argparse,json
from pathlib import Path
from budgetsi.telemetry import Telemetry

def main():
    p=argparse.ArgumentParser();p.add_argument('--ledger',required=True);p.add_argument('--output',required=True)
    p.add_argument('--mode',choices=['offline','online'],default='offline');p.add_argument('--entity')
    p.add_argument('--project',default='budgetsi-qwen35-opd');args=p.parse_args()
    root=Path(args.output)
    if root.exists():raise ValueError('Use a fresh export directory to preserve original evidence')
    t=Telemetry(root,mode=args.mode,entity=args.entity,project=args.project)
    try:
        for line in Path(args.ledger).read_text().splitlines():
            row=json.loads(line);t.log(row['event_id'],row['metrics'])
    finally:t.finish()
if __name__=='__main__':main()
