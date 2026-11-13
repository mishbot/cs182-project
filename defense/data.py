"""Defender training examples: chat-formatted prompt + target token ids."""
from config import PROMPT_TEMPLATE

def make_examples(tok, instructions, targets, max_prompt=256, max_target=64):
    """-> list of (prompt_ids, target_ids). Target ends with <|im_end|> unless truncated."""
    examples = []
    for ins, tgt in zip(instructions, targets):
        p = tok.encode(PROMPT_TEMPLATE.format(instruction=ins))
        t = tok.encode(tgt)
        t = t[:max_target] if len(t) >= max_target else t + [tok.im_end_id]
        if len(p) <= max_prompt:
            examples.append((p, t))
    return examples
