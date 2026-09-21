"""Explicit text-only Qwen3/Qwen3.5 loading; no implicit family conversion."""
import json
from pathlib import Path
import torch


def text_config(model):
    return getattr(model.config, 'text_config', model.config)


def load_model(path, device_map):
    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText
    cfg = AutoConfig.from_pretrained(path, local_files_only=True)
    family = cfg.model_type
    if family not in {'qwen3', 'qwen3_5'}:
        raise ValueError(f'Unvalidated model family: {family}')
    loader = AutoModelForImageTextToText if family == 'qwen3_5' else AutoModelForCausalLM
    return loader.from_pretrained(path, local_files_only=True, torch_dtype=torch.bfloat16,
                                  attn_implementation='sdpa', device_map=device_map)


def lora_targets(model):
    if model.config.model_type != 'qwen3_5':
        return 'all-linear'
    names = [n for n, m in model.named_modules()
             if n.startswith('model.language_model.') and isinstance(m, torch.nn.Linear)]
    if not names:
        raise ValueError('Missing Qwen3.5 text-backbone LoRA targets')
    return names


def eos_contract(tokenizer, path):
    from budgetsi.social_protocol.token_contract import build_contract
    path = Path(path)
    generation = path / 'generation_config.json'
    return set(build_contract(tokenizer, json.loads((path/'config.json').read_text()),
                             json.loads(generation.read_text()) if generation.exists() else None)['eos_ids'])


def upstream_actor_class():
    """Resolve the retired HF auto-class name without editing pinned verl."""
    import transformers
    missing = not hasattr(transformers, 'AutoModelForVision2Seq')
    if missing:
        transformers.AutoModelForVision2Seq = transformers.AutoModelForImageTextToText
    try:
        from verl.workers.actor.dp_actor import DataParallelPPOActor
        return DataParallelPPOActor
    finally:
        if missing:
            delattr(transformers, 'AutoModelForVision2Seq')
