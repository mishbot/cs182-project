"""Monitoring during training, and the Q1 fresh-probe check."""
import torch as th

from common.evaluate import is_refusal
from common.models import generate, prompt_acts
from defense.losses import _scale

def evaluate(model, tok, harm_val, ben_val, cfg):
    r_h = generate(model, tok, harm_val, max_new_tokens=32)
    r_b = generate(model, tok, ben_val, max_new_tokens=32)
    acts_h = prompt_acts(model, tok, harm_val, cfg["layers"], cfg["n_eoi"])
    acts_b = prompt_acts(model, tok, ben_val, cfg["layers"], cfg["n_eoi"])
    gap = (acts_h.mean(2) - acts_b.mean(2)).pow(2).sum(-1) / _scale(acts_h, acts_b)  # [L, P]
    gap_per_layer = gap.mean(1)                                                       # [L]
    return {
        "refusal_rate_harmful": sum(map(is_refusal, r_h)) / len(r_h),
        "refusal_rate_benign": sum(map(is_refusal, r_b)) / len(r_b),
        "gap_val": gap_per_layer.mean().item(),
        "gap_per_layer": [round(g, 4) for g in gap_per_layer.tolist()],
    }

def probe_auroc_per_layer(acts_h, acts_b, pos=-1, seed=0):
    """Q1 check: fresh logistic probe per layer at one position, AUROC on a held-out
    split. acts_*: [L, P, N, d] from prompt_acts."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split
    aurocs = []
    for l in range(acts_h.size(0)):
        X = th.cat([acts_h[l, pos], acts_b[l, pos]]).numpy()
        y = [1] * acts_h.size(2) + [0] * acts_b.size(2)
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)
        clf = LogisticRegression(max_iter=2000).fit(Xtr, ytr)
        aurocs.append(roc_auc_score(yte, clf.predict_proba(Xte)[:, 1]))
    return aurocs
