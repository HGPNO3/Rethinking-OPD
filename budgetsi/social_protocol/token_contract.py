"""Resolve action token boundaries from metadata, never from Qwen3 constants.

This module does not change logits or remove generated tokens. Template tokens
belong to the prompt; all original action tokens, including EOS, remain targets.
"""

def build_contract(tokenizer, config, generation_config=None):
    vocab = tokenizer.get_vocab()
    required = ('<|im_start|>', '<|im_end|>', '<|endoftext|>', '<think>', '</think>')
    ids = {}
    for text in required:
        encoded = tokenizer.encode(text, add_special_tokens=False)
        if text not in vocab or encoded != [vocab[text]]:
            raise ValueError(f'Boundary token is missing or not atomic: {text}')
        ids[text] = vocab[text]
    if len(set(ids.values())) != len(ids):
        raise ValueError('Boundary token IDs collide')
    text_config = config.get('text_config', config)
    width = text_config['vocab_size']
    if any(i < 0 or i >= width for i in vocab.values()):
        raise ValueError('Tokenizer ID outside output head')
    endings = {ids['<|im_end|>']}
    for source in (config, text_config, generation_config or {}):
        value = source.get('eos_token_id', [])
        endings.update(value if isinstance(value, list) else [value] if value is not None else [])
    if not endings.issubset({ids['<|im_end|>'], ids['<|endoftext|>']}):
        raise ValueError('Unexpected EOS metadata; review instead of silently accepting')
    return {'boundaries': ids, 'eos_ids': sorted(endings),
            'thinking_ids': [ids['<think>'], ids['</think>']],
            'output_vocab_size': width, 'mapped_token_count': len(set(vocab.values())),
            'unmapped_output_ids': sorted(set(range(width)) - set(vocab.values()))}

def validate_action_ids(token_ids, tokenizer, contract):
    """Validate original output without decoding/re-encoding or stripping it."""
    if not token_ids or token_ids[-1] not in contract['eos_ids']:
        raise ValueError('Missing complete action EOS')
    known = set(tokenizer.get_vocab().values())
    if any(type(i) is not int or i not in known for i in token_ids):
        raise ValueError('Unmapped original output token')
    if any(i in contract['thinking_ids'] for i in token_ids):
        raise ValueError('Generated thinking marker in nonthinking action')
    if any(i in contract['eos_ids'] for i in token_ids[:-1]):
        raise ValueError('Tokens after action EOS')
    return token_ids
