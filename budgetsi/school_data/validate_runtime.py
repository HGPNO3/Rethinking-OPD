"""Schema + official Sotopia reset/render acceptance; no model/API generation."""
import json,time
from pathlib import Path
from budgetsi.school_data import DATA_DIR,load_initializations,schedule_batch

def main():
    from budgetsi.social_protocol.adapter import create_session
    start=time.perf_counter();scenes=load_initializations();count=0
    for scene in scenes:
        session=create_session(scene)
        assert session.active_role==0
        for role in (0,1):
            messages=session.messages(role)
            assert messages and all(isinstance(m['content'],str) and m['content'] for m in messages)
            assert session.goal(role)==scene['environment']['agent_goals'][role]
        count+=1
    result={'status':'passed','initializations_schema_reset_render_verified':count,'train_scenarios':len({s['environment']['pk'] for s in scenes}),'roles_per_initialization':2,'model_calls':0,'seconds':time.perf_counter()-start}
    (DATA_DIR/'runtime_validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

if __name__=='__main__':main()
