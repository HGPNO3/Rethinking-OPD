"""Official Sotopia 0.1.5 environment and prompt adapter; no model/API calls.
Official environment, renderer, termination and action prompt reused_as_protocol.
"""
import ast
import copy
import inspect
from sotopia.agents.llm_agent import LLMAgent
from sotopia.database import AgentProfile, EnvironmentProfile
from sotopia.envs.parallel import ParallelSotopiaEnv, render_text_for_agent
from sotopia.envs.evaluators import RuleBasedTerminatedEvaluator
from sotopia.messages import AgentAction
from sotopia.generation_utils.generate import agenerate_action
from sotopia.generation_utils.output_parsers import PydanticOutputParser
from sotopia.utils import format_docstring

# Extract the original normal-agent prompt rather than copy a stale template.
_tree=ast.parse(inspect.getsource(agenerate_action))
_templates=[n.value.value for n in ast.walk(_tree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='template' for t in n.targets) and isinstance(n.value,ast.Constant) and isinstance(n.value.value,str)]
_TEMPLATE=next(t for t in _templates if 'Imagine you are {agent}' in t)
_FORMAT=PydanticOutputParser(pydantic_object=AgentAction).get_format_instructions()
PROMPT_PROTOCOL = "official_user_prompt_action_instance_clarifier_v2"
_ACTION_INSTANCE_CLARIFIER = (
    'Return one action JSON instance with exactly the keys "action_type" and "argument", '
    'both with string values. Choose action_type from the available action types in the user prompt. '
    'Do not output a JSON schema, properties, required, title, type, markdown, or explanatory text.'
)

class Session:
    def __init__(self, scene):
        self.id=scene['id']
        self.profiles=[AgentProfile(**p) for p in scene['agents']]
        self.names=[p.first_name+' '+p.last_name for p in self.profiles]
        assert len(set(self.names))==2
        self.env=ParallelSotopiaEnv(env_profile=EnvironmentProfile(**scene['environment']),action_order='round-robin',evaluators=[RuleBasedTerminatedEvaluator(max_turn_number=20,max_stale_turn=2)],terminal_evaluators=[])
        # Stable order avoids hash-seed changes in prompt action list.
        self.env.available_action_types=['none','speak','non-verbal communication','action','leave']
        agents={n:LLMAgent(agent_name=n,agent_profile=p,model_name='unused-no-api') for n,p in zip(self.names,self.profiles)}
        self.obs=self.env.reset(seed=scene['seed'],agents=agents,omniscient=False)
        self.history=[[self.obs[n]] for n in self.names]
        self.done=False
        self.last_info={}
    @property
    def active_role(self):
        return self.env.action_mask.index(True)
    @property
    def turn_number(self):
        return self.env.turn_number
    def clone(self):
        return copy.deepcopy(self)
    def goal(self, role):
        # Retain original role goal text for IG target; never return partner goal.
        return self.env.profile.agent_goals[role]
    def visible_history(self, role):
        return '\n'.join(o.to_natural_language() for o in self.history[role])
    def messages(self, role):
        vals={'agent':self.names[role],'turn_number':str(self.obs[self.names[role]].turn_number),'history':self.visible_history(role),'action_list':' '.join(self.obs[self.names[role]].available_actions),'format_instructions':_FORMAT}
        # Replace placeholders simultaneously, preventing profile text from being treated as template code.
        import re
        prompt=re.sub(r'\{(agent|turn_number|history|action_list|format_instructions)\}',lambda m:vals[m.group(1)],format_docstring(_TEMPLATE))
        return [{'role':'system','content':_ACTION_INSTANCE_CLARIFIER}, {'role':'user','content':prompt}]
    async def step(self, action):
        if self.done: raise ValueError('Cannot step terminated official environment')
        who=self.active_role
        obj=AgentAction(**action)
        if obj.action_type not in self.obs[self.names[who]].available_actions: raise ValueError('Unavailable action')
        actions={n:obj if i==who else AgentAction(action_type='none',argument='') for i,n in enumerate(self.names)}
        self.obs,_,terminated,truncated,self.last_info=self.env.step(actions)
        self.done=any(terminated.values()) or any(truncated.values())
        for i,n in enumerate(self.names): self.history[i].append(self.obs[n])
        return self.done

def create_session(scene):
    return Session(scene)
