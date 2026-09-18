"""Offline provenance, information-boundary and continuation regression checks."""
import hashlib,json,subprocess,sys,tempfile
from pathlib import Path
from prepare_runtime import materialize
ROOT=Path(__file__).resolve().parent

def main():
    m=json.loads((ROOT/'SOURCE_MANIFEST.json').read_text())
    for f in m['files']:
        assert hashlib.sha256((ROOT/f['destination']).read_bytes()).hexdigest()==f['source_sha256']==f['copied_sha256'],f['destination']
    results=[]
    for family in ['qwen3','qwen35']:
        for mode in ['fresh','continuation']:
            with tempfile.TemporaryDirectory() as tmp:
                out=Path(tmp)/'runtime';materialize(family,mode,out)
                r=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(out),'-p','test_*.py','-v'],cwd=out,text=True,capture_output=True)
                results.append({'family':family,'mode':mode,'returncode':r.returncode,'log':r.stdout+r.stderr})
    assert all(r['returncode']==0 for r in results),json.dumps(results,indent=2)
    # The export's approval gate rejects an empty or inherited-unbound packet.
    with tempfile.TemporaryDirectory() as tmp:
        out=Path(tmp)/'runtime';materialize('qwen3','fresh',out)
        r=subprocess.run([sys.executable,'-c','from protocol_gate import verify_approval; assert not verify_approval({})["ok"]'],cwd=out,capture_output=True,text=True)
        assert r.returncode==0,r.stderr
    report={'status':'passed','scope':'CPU source/method regression only; no models, APIs, training or performance evidence',
            'byte_identical_source_files':len(m['files']),'empty_approval_rejected':True,'suites':results}
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
