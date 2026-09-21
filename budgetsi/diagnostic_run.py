"""Separate from the loss: selected student rollout positions, pre-update."""
import time
import math
import torch
from budgetsi.diagnostics import student_summary,compare,Accumulator
from budgetsi.top16 import response_logits,_binding

@torch.no_grad()
def diagnose(student,teacher,actions,settings,step):
    started=time.monotonic();acc=Accumulator();selected=sorted(actions,key=lambda a:a.node_id)[:settings['max_actions']]
    temperatures={a.sampling_temperature for a in actions}
    if not selected or len(temperatures)!=1:
        raise ValueError('Diagnostics require nonempty actions with one sampling temperature')
    temperature=temperatures.pop()
    if not math.isfinite(temperature) or temperature<=0:
        raise ValueError('Invalid diagnostic sampling temperature')
    mode=student.training;student.eval()
    try:
        for action in selected:
            summary=student_summary(response_logits(student,action.student_prompt_ids,action.target_ids),temperature,settings['k'])
            if hasattr(teacher,'diagnose'):
                values=teacher.diagnose(action,summary)
            else:
                values=compare(summary,response_logits(teacher,action.teacher_prompt_ids,action.target_ids),1.)
            acc.add(values)
    finally:student.train(mode)
    return dict(schema='opd_diagnostics_v1',optimizer_step=step,snapshot=selected[0].snapshot_id,
                action_bindings=[_binding(a) for a in selected],selected_node_ids=[a.node_id for a in selected],
                population='deterministic subset of selected original student actions',
                diagnostic_k=settings['k'],student_temperature=temperature,teacher_temperature=1.,
                metrics=acc.result(),seconds=time.monotonic()-started)


def publish(telemetry,receipt,batch,config):
    import json,math
    path=batch/'diagnostics.json';step=receipt['optimizer_step']
    if path.exists():
        d=json.loads(path.read_text())
        if d['optimizer_step']!=step-1 or d['snapshot']!=receipt['snapshot_before']:
            raise ValueError('Diagnostic/update snapshot mismatch')
        if not set(d['selected_node_ids']).issubset(receipt['nodes']):
            raise ValueError('Diagnostic nodes differ from committed update')
        if 'temperature' in config and (d.get('student_temperature')!=config['temperature'] or d.get('teacher_temperature')!=config.get('teacher_temperature',1.)):
            raise ValueError('Diagnostic temperature differs from training configuration')
        telemetry.log('pre_'+str(step-1),{'optimizer_step':step-1,'timing/diagnostics_seconds':d['seconds'],
            **{'diagnostics/'+k:v for k,v in d['metrics'].items() if v is not None}})
    values={'optimizer_step':step,'train/learning_rate':config['optimizer']['lr'],
            'train/nodes':len(receipt['nodes']),'timing/update_and_restore_seconds':receipt['update_and_restore_seconds']}
    if 'collection_seconds' in receipt:values['timing/collection_seconds']=receipt['collection_seconds']
    for k,v in receipt.get('teacher_timing',{}).items():values['timing/teacher_'+k]=v
    for k,v in receipt['metrics'].items():
        if isinstance(v,(list,tuple)) and v:v=sum(v)/len(v)
        if isinstance(v,(int,float)) and math.isfinite(v):values['upstream/'+k]=v
    for i,v in enumerate(receipt['peak_allocated_gib']):values[f'resources/gpu_{i}_peak_allocated_gib']=v
    validation_path=batch/'batch_validation.json'
    if validation_path.exists():
        v=json.loads(validation_path.read_text())
        failed=len(v['invalid_schema_dialogues']);total=failed+v['normal_dialogues']
        values.update({'rollout/format_failed_dialogues':failed,
                       'rollout/format_failure_rate':failed/total if total else 0.,
                       'rollout/validated_selected_nodes':v['validated_selected_nodes']})
    telemetry.log('update_'+str(step),values)
