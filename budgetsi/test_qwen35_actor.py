"""Actual pinned actor + tiny conditional Qwen3.5 text-only LoRA on CPU."""
import torch
from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration
from peft import LoraConfig,get_peft_model
from budgetsi.model_runtime import upstream_actor_class,lora_targets
upstream_actor_class()
from budgetsi.test_variant_runtime import VariantRuntime
from budgetsi.top16 import SelectedAction,LocalActorService
from budgetsi.social_loop import actor_config

class Qwen35ActorRuntime(VariantRuntime):
    def models(self):
        cfg=Qwen3_5Config(text_config=dict(vocab_size=64,hidden_size=32,intermediate_size=64,
            num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=1,head_dim=16,
            linear_key_head_dim=16,linear_value_head_dim=16,linear_num_key_heads=2,
            linear_num_value_heads=2,layer_types=['linear_attention','full_attention'],
            max_position_embeddings=128,bos_token_id=1,eos_token_id=2,pad_token_id=0),
            vision_config=dict(depth=1,hidden_size=32,intermediate_size=64,num_heads=2,
                               out_hidden_size=32,num_position_embeddings=16),
            image_token_id=60,video_token_id=61,vision_start_token_id=62,vision_end_token_id=63)
        torch.manual_seed(71);student=Qwen3_5ForConditionalGeneration(cfg).eval()
        student=get_peft_model(student,LoraConfig(r=4,lora_alpha=8,lora_dropout=0,
                              target_modules=lora_targets(student),task_type='CAUSAL_LM')).eval()
        torch.manual_seed(91);teacher=Qwen3_5ForConditionalGeneration(cfg).eval().requires_grad_(False)
        actions=[SelectedAction(str(i),'snapshot',p,(50,51,*p),y,'tok','tok','protocol','reference',.7)
                 for i,(p,y) in enumerate([((3,4),(7,2)),((8,),(9,10,2))])]
        optimizer=torch.optim.AdamW([p for p in student.parameters() if p.requires_grad],lr=1e-3,weight_decay=.01)
        actor_cfg=actor_config(len(actions));worker=LocalActorService(self.Actor(actor_cfg,student,optimizer),.7)
        return student,teacher,actions,optimizer,actor_cfg,worker
