"""Prompt datasets, the disjoint splits every role uses, and batching.

Same sources and filtering as Arditi et al.'s repo (github.com/andyrdt/refusal_direction).
prepare_data() needs internet the first time; after that the JSON files in data_dir are reused.
"""
import json
import random

import torch as th

def load_instructions(path):
    """Arditi-format JSON: a list of {"instruction": ..., ...} (or plain strings)."""
    with open(path) as f:
        data = json.load(f)
    return [d["instruction"] if isinstance(d, dict) else d for d in data]

SOURCE_URLS = {
    "advbench": "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv",
    "malicious_instruct": "https://raw.githubusercontent.com/Princeton-SysML/Jailbreak_LLM/main/data/MaliciousInstruct.txt",
    "tdc2023_dev": "https://raw.githubusercontent.com/centerforaisafety/tdc2023-starter-kit/main/red_teaming/data/dev/behaviors.json",
    "tdc2023_test": "https://raw.githubusercontent.com/centerforaisafety/tdc2023-starter-kit/main/red_teaming/data/test/behaviors.json",
    "harmbench_val": "https://raw.githubusercontent.com/centerforaisafety/HarmBench/main/data/behavior_datasets/harmbench_behaviors_text_val.csv",
    "harmbench_test": "https://raw.githubusercontent.com/centerforaisafety/HarmBench/main/data/behavior_datasets/harmbench_behaviors_text_test.csv",
}

SPLITS = [
    "harmful_train",          # our fine-tuning:   AdvBench + MaliciousInstruct + TDC2023
    "harmful_val",            # our monitoring:    same pool, held out
    "harmless_train",         # our fine-tuning:   Alpaca
    "harmless_val",           # our monitoring:    Alpaca, held out
    "attack_harmful_train",   # attacker's direction estimate (Arditi: 128 from same pool)
    "attack_harmful_val",     # attacker's direction selection (Arditi: HarmBench val)
    "attack_harmless_train",  # attacker: 128 Alpaca
    "attack_harmless_val",    # attacker: 32 Alpaca
    "harmbench_test",         # final eval: harmful compliance
    "jailbreakbench",         # final eval: harmful compliance
    "harmless_test",          # final eval: utility / over-refusal
]

def _fetch(url):
    import urllib.request
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.read().decode("utf-8")

def _read_csv(url):
    import csv, io
    return list(csv.DictReader(io.StringIO(_fetch(url))))

def _norm(s):
    return " ".join(s.lower().split())

def _dedup(items, banned=()):
    """Drop duplicates (case/whitespace-insensitive) and anything in `banned`."""
    seen, out = {_norm(x["instruction"]) for x in banned}, []
    for x in items:
        n = _norm(x["instruction"])
        if n not in seen:
            seen.add(n)
            out.append(x)
    return out

def download_sources():
    from datasets import load_dataset

    def item(text, cat=None):
        return {"instruction": text.strip(), "category": cat}

    advbench = [item(r["goal"]) for r in _read_csv(SOURCE_URLS["advbench"])]
    malicious = [item(l) for l in _fetch(SOURCE_URLS["malicious_instruct"]).splitlines() if l.strip()]
    tdc = [item(b) for k in ("tdc2023_dev", "tdc2023_test") for b in json.loads(_fetch(SOURCE_URLS[k]))]

    def harmbench(split):
        rows = _read_csv(SOURCE_URLS[f"harmbench_{split}"])
        # Arditi et al. drop copyright behaviours and ones that need a context string
        return [item(r["Behavior"], r.get("SemanticCategory"))
                for r in rows
                if "copyright" not in (r.get("FunctionalCategory") or "")
                and "context" not in (r.get("Tags") or "")]

    jbb = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors", split="harmful")
    jbb = [item(r["Goal"], r["Category"]) for r in jbb]

    alpaca = load_dataset("tatsu-lab/alpaca", split="train")
    alpaca = [item(r["instruction"]) for r in alpaca if not r["input"].strip()]

    return dict(advbench=advbench, malicious=malicious, tdc=tdc,
                harmbench_val=harmbench("val"), harmbench_test=harmbench("test"),
                jbb=jbb, alpaca=alpaca)

def prepare_data(data_dir="data", seed=42, n_attack_train=128, n_attack_val=32,
                 n_monitor=128, n_harmless_test=500, overwrite=False):
    """Download the sources and write every split in SPLITS to data_dir as JSON.
    Evaluation sets are fixed first; everything else is filtered against them, so
    no prompt appears in more than one split."""
    import os
    paths = {s: os.path.join(data_dir, f"{s}.json") for s in SPLITS}
    if not overwrite and all(os.path.exists(p) for p in paths.values()):
        print(f"data already prepared in {data_dir}/")
        return paths

    print("downloading datasets...")
    src = download_sources()
    rng = random.Random(seed)

    # 1. evaluation sets (fixed)
    harmbench_test = _dedup(src["harmbench_test"])
    jbb = _dedup(src["jbb"], banned=harmbench_test)
    evals = harmbench_test + jbb

    # 2. attacker's validation set (HarmBench val, as in Arditi et al.)
    hb_val = _dedup(src["harmbench_val"], banned=evals)
    rng.shuffle(hb_val)
    attack_harmful_val = hb_val[:n_attack_val]

    # 3. harmful pool -> attacker train / our monitoring / our training
    pool = _dedup(src["advbench"] + src["malicious"] + src["tdc"],
                  banned=evals + attack_harmful_val)
    rng.shuffle(pool)
    attack_harmful_train = pool[:n_attack_train]
    harmful_val = pool[n_attack_train:n_attack_train + n_monitor]
    harmful_train = pool[n_attack_train + n_monitor:]

    # 4. Alpaca -> attacker train/val / our monitoring / utility test / our training
    alpaca = _dedup(src["alpaca"])
    rng.shuffle(alpaca)
    cuts = [n_attack_train, n_attack_val, n_monitor, n_harmless_test]
    parts, i = [], 0
    for c in cuts:
        parts.append(alpaca[i:i + c])
        i += c
    attack_harmless_train, attack_harmless_val, harmless_val, harmless_test = parts
    harmless_train = alpaca[i:]

    splits = dict(
        harmful_train=harmful_train, harmful_val=harmful_val,
        harmless_train=harmless_train, harmless_val=harmless_val,
        attack_harmful_train=attack_harmful_train, attack_harmful_val=attack_harmful_val,
        attack_harmless_train=attack_harmless_train, attack_harmless_val=attack_harmless_val,
        harmbench_test=harmbench_test, jailbreakbench=jbb, harmless_test=harmless_test,
    )
    os.makedirs(data_dir, exist_ok=True)
    for name, items in splits.items():
        with open(paths[name], "w") as f:
            json.dump(items, f, indent=1)
        print(f"  {name:22s} {len(items):6d}")
    return paths

# ---------------------------------------------------------------------------
# Batching
# ---------------------------------------------------------------------------

def collate(tok, examples, device):
    """Right-padded batch of prompt + target, with per-example prompt length."""
    seqs = [p + t for p, t in examples]
    T = max(len(s) for s in seqs)
    ids = th.full((len(seqs), T), tok.pad_token_id, dtype=th.long)
    mask = th.zeros((len(seqs), T), dtype=th.long)
    for i, s in enumerate(seqs):
        ids[i, :len(s)] = th.tensor(s)
        mask[i, :len(s)] = 1
    prompt_len = th.tensor([len(p) for p, _ in examples])
    return {"input_ids": ids.to(device), "attention_mask": mask.to(device),
            "prompt_len": prompt_len.to(device)}
